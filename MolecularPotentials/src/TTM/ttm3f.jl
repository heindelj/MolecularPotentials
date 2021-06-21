include("electrostatics.jl")
include("Partridge_Schwenke.jl")
include("utilities.jl")

mutable struct TTM3F <: TTM_Potential
    num_waters::Int

    M_site_coords::Vector{SVector{3, Float64}}
    constants::TTM3_Constants
    elec_data::Electrostatics
end

TTM3F(num_waters::Int) = TTM3F(num_waters, [@SVector zeros(3) for _ in 1:(num_waters > 0 ? 4*num_waters : 1)], TTM3_Constants(), Electrostatics(zeros(Int(num_waters * 4)), TTM3_Constants()))
TTM3F() = TTM3F(0)

function evaluate!(ttm3f::TTM3F, coords::AbstractMatrix{Float64}, grads::Union{Matrix{Float64}, Nothing}=nothing)
    """
    Evaluates the ttm3f energy of the system containing water called coords.
    """
    @assert isinteger(size(coords, 2) / 3) "Number of coordinates not divisble by three. Is this water?"
    if (ttm3f.num_waters != Int(size(coords, 2) / 3))
        temp_ttm = TTM3F(Int(size(coords, 2) / 3))
        ttm3f.num_waters = temp_ttm.num_waters
        ttm3f.M_site_coords = temp_ttm.M_site_coords
        ttm3f.elec_data = temp_ttm.elec_data
    end

    compute_M_site_crd!(ttm3f, coords)
    #Rij_vectors = distance_vectors(M_site_coords)
    # @SPEED Ignore all non M-Site calculations in the electrostatics calculation !!!!!
    # which is only executed if the gradients are actually requested
    grads_E = [@MVector zeros(3) for _ in 1:3 * 4 * ttm3f.num_waters] # O H H M
    grads_q = [@MArray zeros(3,3,3) for _ in 1:ttm3f.num_waters]
    # calculate the INTRA-molecular energy and Dipole Moment
    # Surface (Partridge-Schwenke)
    E_int = PS_energies_and_gradients!(ttm3f, coords, grads_q, grads, true)
    
    ttm3f.elec_data.q *= CHARGECON
    if grads !== nothing
        grads_q *= CHARGECON
    end
   
    #---------------------------------------------------------------!
    # Calculate the vdW interactions for all atoms                  !
    #---------------------------------------------------------------!

    E_vdw::Float64 = 0.0
    # Loop through all of the waters for van der Waal's interactions
    for i in 1:ttm3f.num_waters-1
        for j in i+1:ttm3f.num_waters
            # Calculate Oxygen-Oxygen interactions
            @inbounds @views Rij = coords[:, 3*(i-1)+1] - coords[:, 3*(j-1)+1]
            expon = ttm3f.constants.vdwD * exp(-ttm3f.constants.vdwE * norm(Rij))

            # Add Evdw van der Waal energy for O-O interaction
            E_vdw += ttm3f.constants.vdwC / norm(Rij)^6 + expon

            if (grads !== nothing) 
                tmp::Float64 = (-(6.0 * ttm3f.constants.vdwC / norm(Rij)^6) / (Rij ⋅ Rij) - ttm3f.constants.vdwE * expon / norm(Rij))

                # add forces to indices for the M site
                @inbounds @views grads[:, 3*(i-1)+1] += tmp * Rij
                @inbounds @views grads[:, 3*(j-1)+1] -= tmp * Rij
            end
        end
    end
    
    # if no gradients then just return energy here.
    if (grads === nothing)
        return E_int + E_vdw + electrostatics(ttm3f.elec_data, ttm3f.M_site_coords, nothing, true)
    end

    #-------------------------------------------------------------------------!
    # Calculate the remaining part of the derivatives                         !
    #-------------------------------------------------------------------------!

    E_elec::Float64 = electrostatics(ttm3f.elec_data, ttm3f.M_site_coords, grads_E, true)
 
    #assert(m_electrostatics.dipoles_converged())

    # this function call might be slowing things down.
    # 
    add_electrostatic_and_dms_derivatives!(ttm3f, grads, grads_E, grads_q)

    return E_int + E_vdw + E_elec
end
