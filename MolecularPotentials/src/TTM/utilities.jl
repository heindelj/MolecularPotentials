abstract type TTM_Potential end
include("constants.jl")

@inline function compute_M_site_crd!(ttm::TTM_Potential, coords::AbstractMatrix{Float64})
    """
    Takes the 3xN coordinates you get coming in and stores the coordinates with M site in the TTM_Potential struct.
    """
    num_M_sites::Int = 0
    for i in 1:Int(size(coords, 2) / 3) * 4
        if i % 4 == 0
            @inbounds ttm.M_site_coords[i] = convert(SVector{3, Float64}, ttm.constants.γ_1 * coords[:,i-3 - num_M_sites] + ttm.constants.γ_2 * (coords[:,i-2 - num_M_sites] + coords[:, i-1 - num_M_sites]))
            num_M_sites += 1
        else
            @inbounds ttm.M_site_coords[i] = convert(SVector{3, Float64}, coords[:, (i - num_M_sites)])
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


function get_dipoles(ttm::TTM_Potential)
    dipoles = zeros(3, ttm.num_waters)
    for i in 1:length(ttm.M_site_coords)
        if i % 4 != 1 # static dipoles from Partridge-Schwenke
            dipoles[:, (i-1)÷4+1] += ttm.elec_data.q[i] * ttm.M_site_coords[i]
        else # add in the induced dipoles
            dipoles[:, i÷4+1] += ttm.elec_data.dipoles[i]   # O
            dipoles[:, i÷4+1] += ttm.elec_data.dipoles[i+1] # H
            dipoles[:, i÷4+1] += ttm.elec_data.dipoles[i+2] # H
        end
    end
    return dipoles / CHARGECON * DEBYE
end

function get_dipole_magnitudes(ttm::TTM_Potential)
    return norm.(eachcol(get_dipoles(ttm)))
end

function PS_energies_and_gradients!(ttm::TTM_Potential, coords::AbstractMatrix{Float64}, grads_q::Vector{MArray{Tuple{3, 3, 3}, Float64, 3, 27}}, grads::Union{Matrix{Float64}, Nothing}=nothing, ttm3f::Bool=false)
    E_int::Float64 = 0.0
    q3 = @MVector zeros(3)
    dq3 = @MArray zeros(3,3,3)
    for n in 1:ttm.num_waters

        if (grads !== nothing)
            @inbounds E_int += pot_nasa(convert(SMatrix{3, 3, Float64}, @view(coords[:, (3*n-2):3*n])), @view(grads[:, (3*n-2):3*n]))
            @inbounds dms_nasa!(convert(SMatrix{3, 3, Float64}, @view(coords[:, (3*n-2):3*n])), q3, dq3, ttm3f)
        else
            @inbounds E_int += pot_nasa(convert(SMatrix{3, 3, Float64}, @view(coords[:, (3*n-2):3*n])), nothing)
            @inbounds dms_nasa!(convert(SMatrix{3, 3, Float64}, @view(coords[:, (3*n-2):3*n])), q3, nothing, ttm3f)
        end

        # TTM_Potential2.1-F assignment 
        tmp::Float64 = 0.5 * ttm.constants.γ_M / (1.0 - ttm.constants.γ_M)
        @inbounds ttm.elec_data.q[4*(n-1) + 1] = 0.0                                  # O
        @inbounds ttm.elec_data.q[4*(n-1) + 2] = q3[2] + tmp * (q3[2] + q3[3])        # H1
        @inbounds ttm.elec_data.q[4*(n-1) + 3] = q3[3] + tmp * (q3[2] + q3[3])        # H2
        @inbounds ttm.elec_data.q[4*(n-1) + 4] = q3[1] / (1.0 - ttm.constants.γ_M) # M

        if (grads === nothing)
            continue
        end
        # charge gradients
        @inbounds grads_q[n][:, 1, :] = dq3[:, 1, :] +   tmp*(dq3[:, 1, :] + dq3[:, 2, :])
        @inbounds grads_q[n][:, 2, :] = dq3[:, 2, :] +   tmp*(dq3[:, 2, :] + dq3[:, 1, :])
        @inbounds grads_q[n][:, 3, :] = dq3[:, 3, :] - 2*tmp*(dq3[:, 1, :] + dq3[:, 2, :])
    end
    
    return E_int
end

function add_electrostatic_and_dms_derivatives!(ttm::TTM_Potential, grads::AbstractMatrix{Float64}, grads_E::Vector{MVector{3, Float64}}, grads_q::Vector{MArray{Tuple{3, 3, 3}, Float64, 3, 27}})
    # add electrostatic derivatives
    for n in 1:ttm.num_waters
        # add O H H gradients from PS and electrostatics (which has M site)
        for k in 1:3
            @inbounds @views grads[:, 3 * (n-1) + k] += grads_E[4 * (n-1) + k]
        end

        # redistribute the M-site derivatives
        @inbounds @views grads[:, 3 * (n-1) + 1] += (1.0 - ttm.constants.γ_M) * grads_E[4*(n-1) + 4] # O
        @inbounds @views grads[:, 3 * (n-1) + 2] +=  0.5 * ttm.constants.γ_M  * grads_E[4*(n-1) + 4] # H
        @inbounds @views grads[:, 3 * (n-1) + 3] +=  0.5 * ttm.constants.γ_M  * grads_E[4*(n-1) + 4] # H
    end

    # derivatives from the adjustable charges of the NASA's PES
    for n in 1:ttm.num_waters
        i_O  = 3 * (n-1) + 1
        i_H1 = 3 * (n-1) + 2
        i_H2 = 3 * (n-1) + 3

        @inbounds grads[:, i_H1] += (grads_q[n][:, 1, 1]*ttm.elec_data.ϕ[4*(n-1) + 2]   # phi(h1)
                         + grads_q[n][:, 2, 1]*ttm.elec_data.ϕ[4*(n-1) + 3]   # phi(h2)
                         + grads_q[n][:, 3, 1]*ttm.elec_data.ϕ[4*(n-1) + 4])  # phi(M)

        @inbounds grads[:, i_H2] += (grads_q[n][:, 1, 2]*ttm.elec_data.ϕ[4*(n-1) + 2]   # phi(h1)
                         + grads_q[n][:, 2, 2]*ttm.elec_data.ϕ[4*(n-1) + 3]   # phi(h2)
                         + grads_q[n][:, 3, 2]*ttm.elec_data.ϕ[4*(n-1) + 4])  # phi(M)

        @inbounds grads[:, i_O]  += (grads_q[n][:, 1, 3]*ttm.elec_data.ϕ[4*(n-1) + 2]   # phi(h1)
                         + grads_q[n][:, 2, 3]*ttm.elec_data.ϕ[4*(n-1) + 3]   # phi(h2)
                         + grads_q[n][:, 3, 3]*ttm.elec_data.ϕ[4*(n-1) + 4])  # phi(M)
    end
end
