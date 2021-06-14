include("electrostatics.jl")
include("Partridge_Schwenke.jl")

mutable struct TTM21F
    num_waters::Int

    constants::TTM21_Constants
    elec_data::Electrostatics
end

TTM21F(num_waters::Int) = TTM21F(num_waters, TTM21_Constants(), Electrostatics(zeros(Int(num_waters * 4)), TTM21_Constants()))
TTM21F() = TTM21F(0)

function compute_M_site_crd!(ttm21f::TTM21F, coords::AbstractMatrix{Float64})
    """
    Takes the 3xN coordinates you get coming in and stores the coordinates with
    M site in the TTM21F struct.
    """
    M_site_coords::Vector{SVector{3, Float64}} = []
    num_M_sites::Int = 0
    for i in 1:Int(size(coords, 2) / 3) * 4
        if i % 4 == 0
            push!(M_site_coords, convert(SVector{3, Float64}, ttm21f.constants.γ_1 * coords[:,i-3 - num_M_sites] + ttm21f.constants.γ_2 * (coords[:,i-2 - num_M_sites] + coords[:, i-1 - num_M_sites])))
            num_M_sites += 1
        else
            push!(M_site_coords, convert(SVector{3, Float64}, coords[:, (i - num_M_sites)]))
        end
    end
    return M_site_coords
end

function evaluate!(ttm21f::TTM21F, coords::AbstractMatrix{Float64}, grads::Union{Matrix{Float64}, Nothing}=nothing)
    """
    Evaluates the ttm21f energy of the system containing water called coords.
    """
    @assert isinteger(size(coords, 2) / 3) "Number of coordinates not divisble by three. Is this water?"
    if (ttm21f.num_waters != Int(size(coords, 2) / 3))
        ttm21f = TTM21F(Int(size(coords, 2) / 3))
    end
    M_site_coords = compute_M_site_crd!(ttm21f, coords)
    # @SPEED factor all of the gradient intermediates into a different function or block
    # which is only executed if the gradients are actually requested
    grads_E = zeros(3, ttm21f.num_waters * 4 * 3) # O H H M
    grads_q = zeros(3,3,3,ttm21f.num_waters)
    # calculate the INTRA-molecular energy and Dipole Moment
    # Surface (Partridge-Schwenke)

    E_int::Float64 = 0.0
    # @SPEED make threaded
    q3 = @MVector zeros(3)
    dq3 = @MArray zeros(3,3,3)
    for n in 1:ttm21f.num_waters

        if (grads !== nothing)
            E_int += pot_nasa(convert(SMatrix{3, 3, Float64}, @view(coords[:, (3*n-2):3*n])), @view(grads[:, (3*n-2):3*n]))
            dms_nasa!(convert(SMatrix{3, 3, Float64}, @view(coords[:, (3*n-2):3*n])), q3, dq3)
        else
            E_int += pot_nasa(convert(SMatrix{3, 3, Float64}, @view(coords[:, (3*n-2):3*n])), nothing)
            dms_nasa!(convert(SMatrix{3, 3, Float64}, @view(coords[:, (3*n-2):3*n])), q3)
        end

        # TTM2.1-F assignment 
        tmp::Float64 = 0.5 * ttm21f.constants.γ_M / (1.0 - ttm21f.constants.γ_M)
        ttm21f.elec_data.q[4*(n-1) + 1] = 0.0                                  # O
        ttm21f.elec_data.q[4*(n-1) + 2] = q3[2] + tmp * (q3[2] + q3[3])        # H1
        ttm21f.elec_data.q[4*(n-1) + 3] = q3[3] + tmp * (q3[2] + q3[3])        # H2
        ttm21f.elec_data.q[4*(n-1) + 4] = q3[1] / (1.0 - ttm21f.constants.γ_M) # M

        if (grads === nothing)
            continue
        end
        # charge gradients
        for k in 1:3
            grads_q[k, 1, 1, n] = dq3[k, 1, 1] + tmp*(dq3[k, 1, 1] + dq3[k, 2, 1])
            grads_q[k, 1, 2, n] = dq3[k, 1, 2] + tmp*(dq3[k, 1, 2] + dq3[k, 2, 2])
            grads_q[k, 1, 3, n] = dq3[k, 1, 3] + tmp*(dq3[k, 1, 3] + dq3[k, 2, 3])

            grads_q[k, 2, 1, n] = dq3[k, 2, 1] + tmp*(dq3[k, 2, 1] + dq3[k, 1, 1])
            grads_q[k, 2, 2, n] = dq3[k, 2, 2] + tmp*(dq3[k, 2, 2] + dq3[k, 1, 2])
            grads_q[k, 2, 3, n] = dq3[k, 2, 3] + tmp*(dq3[k, 2, 3] + dq3[k, 1, 3])

            grads_q[k, 3, 1, n] = dq3[k, 3, 1] - 2*tmp*(dq3[k, 1, 1] + dq3[k, 2, 1])
            grads_q[k, 3, 2, n] = dq3[k, 3, 2] - 2*tmp*(dq3[k, 1, 2] + dq3[k, 2, 2])
            grads_q[k, 3, 3, n] = dq3[k, 3, 3] - 2*tmp*(dq3[k, 1, 3] + dq3[k, 2, 3])
        end
    end
    ttm21f.elec_data.q *= CHARGECON
    
    if grads !== nothing
        grads_q *= CHARGECON
    end
    #---------------------------------------------------------------!
    # Calculate the vdW interactions for all atoms                  !
    #---------------------------------------------------------------!

    E_vdw::Float64 = 0.0
    # Loop through all of the waters for van der Waal's interactions
    for i in 1:ttm21f.num_waters-1
        for j in i+1:ttm21f.num_waters
            # Calculate Oxygen-Oxygen interactions
            @views Rij = coords[:, 3*(i-1)+1] - coords[:, 3*(j-1)+1]
            expon = ttm21f.constants.vdwD * exp(-ttm21f.constants.vdwE * norm(Rij))

            # Add Evdw van der Waal energy for O-O interaction
            x::Float64 = 1 / (Rij ⋅ Rij)
            E_vdw += expon + x * x * x * (ttm21f.constants.vdwC + x * x * (ttm21f.constants.vdwB + x * ttm21f.constants.vdwA))

            if (grads !== nothing) 
                tmp::Float64 = (ttm21f.constants.vdwE * expon / norm(Rij) 
                + x*x*x*x*(6 * ttm21f.constants.vdwC 
                + x*x*(10 * ttm21f.constants.vdwB + 12 * x * ttm21f.constants.vdwA)))

                # add forces to indices for the M site
                for k in 1:3
                    grads[k, 3*(i-1)+1] -= tmp * Rij[k]
                    grads[k, 3*(j-1)+1] += tmp * Rij[k]
                end
            end
        end
    end
    
    # if no gradients then just return energy here.
    if (grads === nothing)
        return E_int + E_vdw + electrostatics(ttm21f.elec_data, M_site_coords, nothing)
    end

    #-------------------------------------------------------------------------!
    # Calculate the remaining part of the derivatives                         !
    #-------------------------------------------------------------------------!

    E_elec::Float64 = electrostatics(ttm21f.elec_data, M_site_coords, grads_E)

    # this will be relevant once I add in the iterative approach
    # as an alternative to cholesky factorization
    #assert(m_electrostatics.dipoles_converged())

    # add electrostatic derivatives
    for n in 1:ttm21f.num_waters
        # add O H H gradients from PS and electrostatics (which has M site)
        for k in 1:3
            @views grads[:, 3 * (n-1) + k] += grads_E[:, 4 * (n-1) + k]
        end

        # redistribute the M-site derivatives
        @views grads[:, 3 * (n-1) + 1] += (1.0 - ttm21f.constants.γ_M) * grads_E[:, 4*(n-1) + 4] # O
        @views grads[:, 3 * (n-1) + 2] +=  0.5 * ttm21f.constants.γ_M  * grads_E[:, 4*(n-1) + 4] # H
        @views grads[:, 3 * (n-1) + 3] +=  0.5 * ttm21f.constants.γ_M  * grads_E[:, 4*(n-1) + 4] # H
    end

    # derivatives from the adjustable charges of the NASA's PES
    for n in 1:ttm21f.num_waters
        i_O  = 3 * (n-1) + 1
        i_H1 = 3 * (n-1) + 2
        i_H2 = 3 * (n-1) + 3

        for k in 1:3
            grads[k, i_H1] += (grads_q[k, 1, 1, n]*ttm21f.elec_data.ϕ[4*(n-1) + 2]   # phi(h1)
                             + grads_q[k, 2, 1, n]*ttm21f.elec_data.ϕ[4*(n-1) + 3]   # phi(h2)
                             + grads_q[k, 3, 1, n]*ttm21f.elec_data.ϕ[4*(n-1) + 4])  # phi(M)

            grads[k, i_H2] += (grads_q[k, 1, 2, n]*ttm21f.elec_data.ϕ[4*(n-1) + 2]   # phi(h1)
                             + grads_q[k, 2, 2, n]*ttm21f.elec_data.ϕ[4*(n-1) + 3]   # phi(h2)
                             + grads_q[k, 3, 2, n]*ttm21f.elec_data.ϕ[4*(n-1) + 4])  # phi(M)

            grads[k, i_O]  += (grads_q[k, 1, 3, n]*ttm21f.elec_data.ϕ[4*(n-1) + 2]   # phi(h1)
                             + grads_q[k, 2, 3, n]*ttm21f.elec_data.ϕ[4*(n-1) + 3]   # phi(h2)
                             + grads_q[k, 3, 3, n]*ttm21f.elec_data.ϕ[4*(n-1) + 4])  # phi(M)
        end
    end

    return E_int + E_vdw + E_elec
end