include("electrostatics.jl")
include("Partridge_Schwenke.jl")
using HybridArrays

mutable struct TTM21F
    num_waters::Int

    M_site_coords::Vector{SVector{3, Float64}}
    constants::TTM21_Constants
    elec_data::Electrostatics
end

TTM21F(num_waters::Int) = TTM21F(num_waters, [@SVector zeros(3) for _ in 1:(num_waters > 0 ? 4*num_waters : 1)], TTM21_Constants(), Electrostatics(zeros(Int(num_waters * 4)), TTM21_Constants()))
TTM21F() = TTM21F(0)

function compute_M_site_crd!(ttm21f::TTM21F, coords::AbstractMatrix{Float64})
    """
    Takes the 3xN coordinates you get coming in and stores the coordinates with
    M site in the TTM21F struct.
    """
    num_M_sites::Int = 0
    for i in 1:Int(size(coords, 2) / 3) * 4
        if i % 4 == 0
            @inbounds ttm21f.M_site_coords[i] = convert(SVector{3, Float64}, ttm21f.constants.γ_1 * coords[:,i-3 - num_M_sites] + ttm21f.constants.γ_2 * (coords[:,i-2 - num_M_sites] + coords[:, i-1 - num_M_sites]))
            num_M_sites += 1
        else
            @inbounds ttm21f.M_site_coords[i] = convert(SVector{3, Float64}, coords[:, (i - num_M_sites)])
        end
    end
end

@inline function triu_linear_index(i::Int, j::Int, n::Int)
    return Int(n*(n-1)/2 - (n-i+1)*(n-i)/2 + j - i)
end

function distance_vectors(coords::Vector{SVector{3, Float64}})
    # make this take the storage as most time will be spent allocating
    n::Int = length(coords)
    distances = [@SVector zeros(3) for _ in 1:(n*(n-1)/2)]
    for i in 1:n-1
        for j in i+1:n
            distances[triu_linear_index(i,j,n)] = coords[i]- coords[j]
        end
    end
    return distances
end

function get_dipoles(ttm21f::TTM21F)
    dipoles = zeros(3, ttm21f.num_waters)
    for i in 1:length(ttm21f.M_site_coords)
        if i % 4 != 1 # static dipoles from Partridge-Schwenke
            dipoles[:, (i-1)÷4+1] += ttm21f.elec_data.q[i] * ttm21f.M_site_coords[i]
        else # add in the induced dipoles
            dipoles[:, i÷4+1] += ttm21f.elec_data.dipoles[i]   # O
            dipoles[:, i÷4+1] += ttm21f.elec_data.dipoles[i+1] # H
            dipoles[:, i÷4+1] += ttm21f.elec_data.dipoles[i+2] # H
        end
    end
    return dipoles / CHARGECON * DEBYE
end

function get_dipole_magnitudes(ttm21f::TTM21F)
    return norm.(eachcol(get_dipoles(ttm21f)))
end

function evaluate!(ttm21f::TTM21F, coords::AbstractMatrix{Float64}, grads::Union{Matrix{Float64}, Nothing}=nothing, use_cholesky::Bool=false)
    """
    Evaluates the ttm21f energy of the system containing water called coords.
    """
    @assert isinteger(size(coords, 2) / 3) "Number of coordinates not divisble by three. Is this water?"
    if (ttm21f.num_waters != Int(size(coords, 2) / 3))
        temp_ttm = TTM21F(Int(size(coords, 2) / 3))
        ttm21f.num_waters = temp_ttm.num_waters
        ttm21f.M_site_coords = temp_ttm.M_site_coords
        ttm21f.elec_data = temp_ttm.elec_data
    end

    compute_M_site_crd!(ttm21f, coords)
    #Rij_vectors = distance_vectors(M_site_coords)
    # @SPEED factor all of the gradient intermediates into a different function or block
    # which is only executed if the gradients are actually requested
    grads_E = [@MVector zeros(3) for _ in 1:3 * 4 * ttm21f.num_waters] # O H H M
    grads_q = [@MArray zeros(3,3,3) for _ in 1:ttm21f.num_waters]
    # calculate the INTRA-molecular energy and Dipole Moment
    # Surface (Partridge-Schwenke)

    E_int::Float64 = 0.0
    # @SPEED make threaded
    q3 = @MVector zeros(3)
    dq3 = @MArray zeros(3,3,3)
    for n in 1:ttm21f.num_waters

        if (grads !== nothing)
            @inbounds E_int += pot_nasa(convert(SMatrix{3, 3, Float64}, @view(coords[:, (3*n-2):3*n])), @view(grads[:, (3*n-2):3*n]))
            @inbounds dms_nasa!(convert(SMatrix{3, 3, Float64}, @view(coords[:, (3*n-2):3*n])), q3, dq3)
        else
            @inbounds E_int += pot_nasa(convert(SMatrix{3, 3, Float64}, @view(coords[:, (3*n-2):3*n])), nothing)
            @inbounds dms_nasa!(convert(SMatrix{3, 3, Float64}, @view(coords[:, (3*n-2):3*n])), q3)
        end

        # TTM2.1-F assignment 
        tmp::Float64 = 0.5 * ttm21f.constants.γ_M / (1.0 - ttm21f.constants.γ_M)
        @inbounds ttm21f.elec_data.q[4*(n-1) + 1] = 0.0                                  # O
        @inbounds ttm21f.elec_data.q[4*(n-1) + 2] = q3[2] + tmp * (q3[2] + q3[3])        # H1
        @inbounds ttm21f.elec_data.q[4*(n-1) + 3] = q3[3] + tmp * (q3[2] + q3[3])        # H2
        @inbounds ttm21f.elec_data.q[4*(n-1) + 4] = q3[1] / (1.0 - ttm21f.constants.γ_M) # M
        if (grads === nothing)
            continue
        end
        # charge gradients
        @inbounds grads_q[n][:, 1, :] = dq3[:, 1, :] +   tmp*(dq3[:, 1, :] + dq3[:, 2, :])
        @inbounds grads_q[n][:, 2, :] = dq3[:, 2, :] +   tmp*(dq3[:, 2, :] + dq3[:, 1, :])
        @inbounds grads_q[n][:, 3, :] = dq3[:, 3, :] - 2*tmp*(dq3[:, 1, :] + dq3[:, 2, :])
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
            @inbounds @views Rij = coords[:, 3*(i-1)+1] - coords[:, 3*(j-1)+1]
            expon = ttm21f.constants.vdwD * exp(-ttm21f.constants.vdwE * norm(Rij))

            # Add Evdw van der Waal energy for O-O interaction
            x::Float64 = 1 / (Rij ⋅ Rij)
            E_vdw += expon + x * x * x * (ttm21f.constants.vdwC + x * x * (ttm21f.constants.vdwB + x * ttm21f.constants.vdwA))

            if (grads !== nothing) 
                tmp::Float64 = (ttm21f.constants.vdwE * expon / norm(Rij) 
                + x*x*x*x*(6 * ttm21f.constants.vdwC 
                + x*x*(10 * ttm21f.constants.vdwB + 12 * x * ttm21f.constants.vdwA)))

                # add forces to indices for the M site
                @inbounds @views grads[:, 3*(i-1)+1] -= tmp * Rij
                @inbounds @views grads[:, 3*(j-1)+1] += tmp * Rij
            end
        end
    end
    
    # if no gradients then just return energy here.
    if (grads === nothing)
        return E_int + E_vdw + electrostatics(ttm21f.elec_data, ttm21f.M_site_coords, nothing, use_cholesky)
    end

    #-------------------------------------------------------------------------!
    # Calculate the remaining part of the derivatives                         !
    #-------------------------------------------------------------------------!

    E_elec::Float64 = electrostatics(ttm21f.elec_data, ttm21f.M_site_coords, grads_E, use_cholesky)

    # this will be relevant once I add in the iterative approach
    # as an alternative to cholesky factorization
    #assert(m_electrostatics.dipoles_converged())

    # add electrostatic derivatives
    for n in 1:ttm21f.num_waters
        # add O H H gradients from PS and electrostatics (which has M site)
        for k in 1:3
            @inbounds @views grads[:, 3 * (n-1) + k] += grads_E[4 * (n-1) + k]
        end

        # redistribute the M-site derivatives
        @inbounds @views grads[:, 3 * (n-1) + 1] += (1.0 - ttm21f.constants.γ_M) * grads_E[4*(n-1) + 4] # O
        @inbounds @views grads[:, 3 * (n-1) + 2] +=  0.5 * ttm21f.constants.γ_M  * grads_E[4*(n-1) + 4] # H
        @inbounds @views grads[:, 3 * (n-1) + 3] +=  0.5 * ttm21f.constants.γ_M  * grads_E[4*(n-1) + 4] # H
    end

    # derivatives from the adjustable charges of the NASA's PES
    for n in 1:ttm21f.num_waters
        i_O  = 3 * (n-1) + 1
        i_H1 = 3 * (n-1) + 2
        i_H2 = 3 * (n-1) + 3

        @inbounds grads[:, i_H1] += (grads_q[n][:, 1, 1]*ttm21f.elec_data.ϕ[4*(n-1) + 2]   # phi(h1)
                         + grads_q[n][:, 2, 1]*ttm21f.elec_data.ϕ[4*(n-1) + 3]   # phi(h2)
                         + grads_q[n][:, 3, 1]*ttm21f.elec_data.ϕ[4*(n-1) + 4])  # phi(M)

        @inbounds grads[:, i_H2] += (grads_q[n][:, 1, 2]*ttm21f.elec_data.ϕ[4*(n-1) + 2]   # phi(h1)
                         + grads_q[n][:, 2, 2]*ttm21f.elec_data.ϕ[4*(n-1) + 3]   # phi(h2)
                         + grads_q[n][:, 3, 2]*ttm21f.elec_data.ϕ[4*(n-1) + 4])  # phi(M)

        @inbounds grads[:, i_O]  += (grads_q[n][:, 1, 3]*ttm21f.elec_data.ϕ[4*(n-1) + 2]   # phi(h1)
                         + grads_q[n][:, 2, 3]*ttm21f.elec_data.ϕ[4*(n-1) + 3]   # phi(h2)
                         + grads_q[n][:, 3, 3]*ttm21f.elec_data.ϕ[4*(n-1) + 4])  # phi(M)
    end

    return E_int + E_vdw + E_elec
end
