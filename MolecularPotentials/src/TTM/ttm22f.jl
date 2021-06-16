include("electrostatics.jl")
include("Partridge_Schwenke.jl")

mutable struct TTM22F
    num_waters::Int

    constants::TTM21_Constants
    elec_data::Electrostatics
end

TTM22F(num_waters::Int) = TTM22F(num_waters, TTM22_Constants(), Electrostatics(zeros(Int(num_waters * 4)), TTM22_Constants()))
TTM22F() = TTM22F(0)

function compute_M_site_crd!(ttm22f::TTM21F, coords::AbstractMatrix{Float64})
    """
    Takes the 3xN coordinates you get coming in and stores the coordinates with
    M site in the TTM21F struct.
    """
    M_site_coords::Vector{SVector{3, Float64}} = []
    num_M_sites::Int = 0
    for i in 1:Int(size(coords, 2) / 3) * 4
        if i % 4 == 0
            @inbounds push!(M_site_coords, convert(SVector{3, Float64}, ttm22f.constants.γ_1 * coords[:,i-3 - num_M_sites] + ttm22f.constants.γ_2 * (coords[:,i-2 - num_M_sites] + coords[:, i-1 - num_M_sites])))
            num_M_sites += 1
        else
            @inbounds push!(M_site_coords, convert(SVector{3, Float64}, coords[:, (i - num_M_sites)]))
        end
    end
    return M_site_coords
end

#function nasa_surface()

function evaluate!(ttm22f::TTM21F, coords::AbstractMatrix{Float64}, grads::Union{Matrix{Float64}, Nothing}=nothing, use_cholesky::Bool=false)
    """
    Evaluates the ttm22f energy of the system containing water called coords.
    """
    @assert isinteger(size(coords, 2) / 3) "Number of coordinates not divisble by three. Is this water?"
    if (ttm22f.num_waters != Int(size(coords, 2) / 3))
        ttm22f = TTM21F(Int(size(coords, 2) / 3))
    end
    M_site_coords = compute_M_site_crd!(ttm22f, coords)
    # @SPEED factor all of the gradient intermediates into a different function or block
    # which is only executed if the gradients are actually requested
    grads_E = [@MVector zeros(3) for _ in 1:3 * 4 * ttm22f.num_waters] # O H H M
    grads_q = [@MArray zeros(3,3,3) for _ in 1:ttm22f.num_waters]
    # calculate the INTRA-molecular energy and Dipole Moment
    # Surface (Partridge-Schwenke)

    E_int::Float64 = 0.0
    # @SPEED make threaded
    q3 = @MVector zeros(3)
    dq3 = @MArray zeros(3,3,3)
    for n in 1:ttm22f.num_waters

        if (grads !== nothing)
            @inbounds E_int += pot_nasa(convert(SMatrix{3, 3, Float64}, @view(coords[:, (3*n-2):3*n])), @view(grads[:, (3*n-2):3*n]))
            @inbounds dms_nasa!(convert(SMatrix{3, 3, Float64}, @view(coords[:, (3*n-2):3*n])), q3, dq3)
        else
            @inbounds E_int += pot_nasa(convert(SMatrix{3, 3, Float64}, @view(coords[:, (3*n-2):3*n])), nothing)
            @inbounds dms_nasa!(convert(SMatrix{3, 3, Float64}, @view(coords[:, (3*n-2):3*n])), q3)
        end

        # TTM2.1-F assignment 
        tmp::Float64 = 0.5 * ttm22f.constants.γ_M / (1.0 - ttm22f.constants.γ_M)
        @inbounds ttm22f.elec_data.q[4*(n-1) + 1] = 0.0                                  # O
        @inbounds ttm22f.elec_data.q[4*(n-1) + 2] = q3[2] + tmp * (q3[2] + q3[3])        # H1
        @inbounds ttm22f.elec_data.q[4*(n-1) + 3] = q3[3] + tmp * (q3[2] + q3[3])        # H2
        @inbounds ttm22f.elec_data.q[4*(n-1) + 4] = q3[1] / (1.0 - ttm22f.constants.γ_M) # M

        if (grads === nothing)
            continue
        end
        # charge gradients
        @inbounds grads_q[n][:, 1, :] = dq3[:, 1, :] +   tmp*(dq3[:, 1, :] + dq3[:, 2, :])
        @inbounds grads_q[n][:, 2, :] = dq3[:, 2, :] +   tmp*(dq3[:, 2, :] + dq3[:, 1, :])
        @inbounds grads_q[n][:, 3, :] = dq3[:, 3, :] - 2*tmp*(dq3[:, 1, :] + dq3[:, 2, :])
    end
    ttm22f.elec_data.q *= CHARGECON
    
    if grads !== nothing
        grads_q *= CHARGECON
    end
    #---------------------------------------------------------------!
    # Calculate the vdW interactions for all atoms                  !
    #---------------------------------------------------------------!

    E_vdw::Float64 = 0.0
    # Loop through all of the waters for van der Waal's interactions
    for i in 1:ttm22f.num_waters-1
        for j in i+1:ttm22f.num_waters
            # Calculate Oxygen-Oxygen interactions
            @inbounds @views Rij = coords[:, 3*(i-1)+1] - coords[:, 3*(j-1)+1]
            expon = ttm22f.constants.vdwD * exp(-ttm22f.constants.vdwE * norm(Rij))

            # Add Evdw van der Waal energy for O-O interaction
            x::Float64 = 1 / (Rij ⋅ Rij)
            E_vdw += expon + x * x * x * (ttm22f.constants.vdwC + x * x * (ttm22f.constants.vdwB + x * ttm22f.constants.vdwA))

            if (grads !== nothing) 
                tmp::Float64 = (ttm22f.constants.vdwE * expon / norm(Rij) 
                + x*x*x*x*(6 * ttm22f.constants.vdwC 
                + x*x*(10 * ttm22f.constants.vdwB + 12 * x * ttm22f.constants.vdwA)))

                # add forces to indices for the M site
                @inbounds @views grads[:, 3*(i-1)+1] -= tmp * Rij
                @inbounds @views grads[:, 3*(j-1)+1] += tmp * Rij
            end
        end
    end
    
    # if no gradients then just return energy here.
    if (grads === nothing)
        return E_int + E_vdw + electrostatics(ttm22f.elec_data, M_site_coords, nothing, use_cholesky)
    end

    #-------------------------------------------------------------------------!
    # Calculate the remaining part of the derivatives                         !
    #-------------------------------------------------------------------------!

    E_elec::Float64 = electrostatics(ttm22f.elec_data, M_site_coords, grads_E, use_cholesky)

    # this will be relevant once I add in the iterative approach
    # as an alternative to cholesky factorization
    #assert(m_electrostatics.dipoles_converged())

    # add electrostatic derivatives
    for n in 1:ttm22f.num_waters
        # add O H H gradients from PS and electrostatics (which has M site)
        for k in 1:3
            @inbounds @views grads[:, 3 * (n-1) + k] += grads_E[4 * (n-1) + k]
        end

        # redistribute the M-site derivatives
        @inbounds @views grads[:, 3 * (n-1) + 1] += (1.0 - ttm22f.constants.γ_M) * grads_E[4*(n-1) + 4] # O
        @inbounds @views grads[:, 3 * (n-1) + 2] +=  0.5 * ttm22f.constants.γ_M  * grads_E[4*(n-1) + 4] # H
        @inbounds @views grads[:, 3 * (n-1) + 3] +=  0.5 * ttm22f.constants.γ_M  * grads_E[4*(n-1) + 4] # H
    end

    # derivatives from the adjustable charges of the NASA's PES
    for n in 1:ttm22f.num_waters
        i_O  = 3 * (n-1) + 1
        i_H1 = 3 * (n-1) + 2
        i_H2 = 3 * (n-1) + 3

        @inbounds grads[:, i_H1] += (grads_q[n][:, 1, 1]*ttm22f.elec_data.ϕ[4*(n-1) + 2]   # phi(h1)
                         + grads_q[n][:, 2, 1]*ttm22f.elec_data.ϕ[4*(n-1) + 3]   # phi(h2)
                         + grads_q[n][:, 3, 1]*ttm22f.elec_data.ϕ[4*(n-1) + 4])  # phi(M)

        @inbounds grads[:, i_H2] += (grads_q[n][:, 1, 2]*ttm22f.elec_data.ϕ[4*(n-1) + 2]   # phi(h1)
                         + grads_q[n][:, 2, 2]*ttm22f.elec_data.ϕ[4*(n-1) + 3]   # phi(h2)
                         + grads_q[n][:, 3, 2]*ttm22f.elec_data.ϕ[4*(n-1) + 4])  # phi(M)

        @inbounds grads[:, i_O]  += (grads_q[n][:, 1, 3]*ttm22f.elec_data.ϕ[4*(n-1) + 2]   # phi(h1)
                         + grads_q[n][:, 2, 3]*ttm22f.elec_data.ϕ[4*(n-1) + 3]   # phi(h2)
                         + grads_q[n][:, 3, 3]*ttm22f.elec_data.ϕ[4*(n-1) + 4])  # phi(M)
    end

    return E_int + E_vdw + E_elec
end
