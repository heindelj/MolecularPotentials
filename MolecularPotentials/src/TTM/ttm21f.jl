include("electrostatics.jl")
include("Partridge_Schwenke.jl")
include("utilities.jl")

mutable struct TTM21F <: TTM
    num_waters::Int

    M_site_coords::Vector{SVector{3, Float64}}
    constants::TTM21_Constants
    elec_data::Electrostatics
end

TTM21F(num_waters::Int) = TTM21F(num_waters, [@SVector zeros(3) for _ in 1:(num_waters > 0 ? 4*num_waters : 1)], TTM21_Constants(), Electrostatics(zeros(Int(num_waters * 4)), TTM21_Constants()))
TTM21F() = TTM21F(0)

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
    E_int = PS_energies_and_gradients!(ttm21f, coords, grads_q, grads)
    
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
