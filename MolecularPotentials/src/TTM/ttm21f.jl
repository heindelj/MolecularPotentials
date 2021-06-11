include("electrostatics.jl")
include("Partridge_Schwenke.jl")

mutable struct TTM21F
    num_waters::Int
    M_site_coords::Matrix{Float64} # @SPEED use StaticArrays

    constants::TTM21_Constants
    elec_data::Electrostatics
end

# You don't have to specify the number of waters ahead of time, but this will be fastest
TTM21F(num_waters::Int) = TTM21F(num_waters, zeros(3, 3 * num_waters), TTM21_Constants(), Electrostatics(zeros(Int(num_waters * 4)), TTM21_Constants()))

function compute_M_site_crd!(ttm21f::TTM21F, coords::AbstractMatrix{Float64})
    """
    Takes the 3xN coordinates you get coming in and stores the coordinates with
    M site in the TTM21F struct.
    """
    #@SPEED: Consider using Static arrays here.
    @assert isinteger(size(coords, 2) / 3) "Number of coordinates not divisble by three. Is this water?"
    nw_index::Int = 1
    for i in 1:size(new_coords, 2)
        if i % 4 == 0
            @views ttm21f.M_site_coords[:, i] = ttm21f.constants.γ_1 * ttm21f.M_site_coords[:,i-3] + ttm21f.constants.γ_2 * (ttm21f.M_site_coords[:,i-2] + ttm21f.M_site_coords[:, i-1])
        else
            @views ttm21f.M_site_coords[:, i] = coords[:, nw_index]
            nw_index += 1
        end
    end
end

function evaluate!(ttm21f::TTM21F, coords::AbstractMatrix{Float64}, grads::Union{Matrix{Float64}, Nothing}=nothing)
    """
    Evaluates the ttm21f energy of the system containing water called coords.
    """
    compute_M_site_crd!(ttm21f, coords)
    # @SPEED factor all of the gradient intermediates into a different function or block
    # which is only executed if the gradients are actually requested
    grads_E = zeros(3, ttm21f.num_waters * 4) # O H H M
    grads_q = convert(MArray{Tuple{3, 3, 3}, Float64, 3, 27}, zeros(27))
    # calculate the INTRA-molecular energy and Dipole Moment
    # Surface (Partridge-Schwenke)

    E_int::Float64 = 0.0
    # @SPEED make threaded
    for n in 1:ttm21f.num_waters
        q3 = @MVector zeros(3)
        dq3 = convert(MArray{Tuple{3, 3, 3}, Float64, 3, 27}, zeros(27))

        if (grads !== nothing)
            E_int += pot_nasa(convert(SMatrix{3, 3, Float64}, @view(coords[:, (3*n-2):3*n])), grads)
            dms_nasa(convert(SMatrix{3, 3, Float64}, coords), q3, dq3)
        else
            E_int += pot_nasa(convert(SMatrix{3, 3, Float64}, @view(coords[:, (3*n-2):3*n])), nothing)
            dms_nasa(convert(SMatrix{3, 3, Float64}, coords), q3)
        end

        # TTM2.1-F assignment 
        tmp::Float64 = 0.5*gammaM / (1.0 - gammaM)
        ttm21f.elec_data.q[4*(n-1) + 1] = 0.0                                                   # O
        ttm21f.elec_data.q[4*(n-1) + 2] = q3[2] + tmp * (q3[2] + q3[3]) # H1
        ttm21f.elec_data.q[4*(n-1) + 3] = q3[3] + tmp * (q3[2] + q3[3]) # H2
        ttm21f.elec_data.q[4*(n-1) + 4] = q3[1] / (1.0 - gammaM)                                # M

        if (grads === nothing)
            continue
        end
        # charge gradients
        for k in 1:3
            grads_q[1, 1, k] = dq3[1, 1, k] + tmp*(dq3[1, 1, k] + dq3[1, 2, k])
            grads_q[2, 1, k] = dq3[2, 1, k] + tmp*(dq3[2, 1, k] + dq3[2, 2, k])
            grads_q[3, 1, k] = dq3[3, 1, k] + tmp*(dq3[3, 1, k] + dq3[3, 2, k])

            grads_q[1, 2, k] = dq3[1, 2, k] + tmp*(dq3[1, 2, k] + dq3[1, 1, k])
            grads_q[2, 2, k] = dq3[2, 2, k] + tmp*(dq3[2, 2, k] + dq3[2, 1, k])
            grads_q[3, 2, k] = dq3[3, 2, k] + tmp*(dq3[3, 2, k] + dq3[3, 1, k])

            grads_q[1, 3, k] = dq3[1, 3, k] - 2*tmp*(dq3[1, 1, k] + dq3[1, 2, k])
            grads_q[2, 3, k] = dq3[2, 3, k] - 2*tmp*(dq3[2, 1, k] + dq3[2, 2, k])
            grads_q[3, 3, k] = dq3[3, 3, k] - 2*tmp*(dq3[3, 1, k] + dq3[3, 2, k])
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
    # Loop through all of the M sites for van der Waal's interactions
    for i in 1:ttm21f.num_waters-1
        for j in i+1:ttm21f.num_waters
            # Calculate Oxygen-Oxygen interactions
            @view Rij = ttm21f.M_site_coords[:, i] - ttm21f.M_site_coords[:, j]
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
                    grads_E[k, 4*i] -= tmp * Rij[k]
                    grads_E[k, 4*j] += tmp * Rij[k]
                end
            end
        end
    end
    
    # if no gradients then just return energy here.
    if (grads === nothing)
        return E_int + E_vdw + electrostatics(ttm21f.elec_data, coords, nothing)
    end

    #-------------------------------------------------------------------------!
    # Calculate the remaining part of the derivatives                         !
    #-------------------------------------------------------------------------!

    E_elec::Float64 = electrostatics(ttm21f.elec_data, coords, grads_E)

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
        @views grads[:, 3 * (n-1) + 1] += (1.0 - ttm21f.constants.γ_M) * dE[:, 4*(n-1) + 4] # O
        @views grads[:, 3 * (n-1) + 2] +=  0.5 * ttm21f.constants.γ_M  * dE[:, 4*(n-1) + 4] # H
        @views grads[:, 3 * (n-1) + 3] +=  0.5 * ttm21f.constants.γ_M  * dE[:, 4*(n-1) + 4] # H
    end

    # derivatives from the adjustable charges of the NASA's PES
    for n in 1:ttm21f.num_waters
        i_O  = 3 * (n-1) + 1
        i_H1 = 3 * (n-1) + 2
        i_H2 = 3 * (n-1) + 3

        for k in 1:3
            grads[k, i_H1] += grads_q[k, 1, 1]*ttm21f.elec_data.ϕ[4*n + 1]  # phi(h1)
                            + grads_q[k, 2, 1]*ttm21f.elec_data.ϕ[4*n + 2]  # phi(h2)
                            + grads_q[k, 3, 1]*ttm21f.elec_data.ϕ[4*n + 3]  # phi(M)

            grads[k, i_H2] += grads_q[k, 1, 2]*ttm21f.elec_data.ϕ[4*n + 1]  # phi(h1)
                            + grads_q[k, 2, 2]*ttm21f.elec_data.ϕ[4*n + 2]  # phi(h2)
                            + grads_q[k, 3, 2]*ttm21f.elec_data.ϕ[4*n + 3]  # phi(M)

            grads[k, i_O]  += grads_q[k, 1, 3]*ttm21f.elec_data.ϕ[4*n + 1]  # phi(h1)
                            + grads_q[k, 2, 3]*ttm21f.elec_data.ϕ[4*n + 2]  # phi(h2)
                            + grads_q[k, 3, 3]*ttm21f.elec_data.ϕ[4*n + 3]  # phi(M)
        end
    end

    return E_int + E_vdw + E_elec
end