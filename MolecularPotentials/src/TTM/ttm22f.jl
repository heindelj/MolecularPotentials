include("electrostatics.jl")
include("Partridge_Schwenke.jl")
include("utilities.jl")

mutable struct TTM22F <: TTM_Potential
    num_waters::Int

    M_site_coords::Vector{SVector{3, Float64}}
    constants::TTM22_Constants
    elec_data::Electrostatics
end

TTM22F(num_waters::Int) = TTM22F(num_waters, [@SVector zeros(3) for _ in 1:(num_waters > 0 ? 4*num_waters : 1)], TTM22_Constants(), Electrostatics(zeros(Int(num_waters * 4)), TTM22_Constants()))
TTM22F() = TTM22F(0)

function evaluate!(ttm22f::TTM22F, coords::AbstractMatrix{Float64}, grads::Union{Matrix{Float64}, Nothing}=nothing)
    """
    Evaluates the ttm22f energy of the system containing water called coords.
    """
    @assert isinteger(size(coords, 2) / 3) "Number of coordinates not divisble by three. Is this water?"
    if (ttm22f.num_waters != Int(size(coords, 2) / 3))
        temp_ttm = TTM22F(Int(size(coords, 2) / 3))
        ttm22f.num_waters = temp_ttm.num_waters
        ttm22f.M_site_coords = temp_ttm.M_site_coords
        ttm22f.elec_data = temp_ttm.elec_data
    end

    compute_M_site_crd!(ttm22f, coords)
    #Rij_vectors = distance_vectors(M_site_coords)
    # @SPEED Ignore all non M-Site calculations in the electrostatics calculation !!!!!
    # which is only executed if the gradients are actually requested
    grads_E = [@MVector zeros(3) for _ in 1:3 * 4 * ttm22f.num_waters] # O H H M
    grads_q = [@MArray zeros(3,3,3) for _ in 1:ttm22f.num_waters]
    # calculate the INTRA-molecular energy and Dipole Moment
    # Surface (Partridge-Schwenke)
    E_int = PS_energies_and_gradients!(ttm22f, coords, grads_q, grads, true)
    
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
            E_vdw += ttm22f.constants.vdwC / norm(Rij)^6 + expon

            if (grads !== nothing) 
                tmp::Float64 = (-(6.0 * ttm22f.constants.vdwC / norm(Rij)^6) / (Rij ⋅ Rij) - ttm22f.constants.vdwE * expon / norm(Rij))

                # add forces to indices for the M site
                @inbounds @views grads[:, 3*(i-1)+1] += tmp * Rij
                @inbounds @views grads[:, 3*(j-1)+1] -= tmp * Rij
            end
        end
    end
    
    # if no gradients then just return energy here.
    if (grads === nothing)
        return E_int + E_vdw + electrostatics(ttm22f.elec_data, ttm22f.M_site_coords, nothing, true)
    end

    #-------------------------------------------------------------------------!
    # Calculate the remaining part of the derivatives                         !
    #-------------------------------------------------------------------------!

    E_elec::Float64 = electrostatics(ttm22f.elec_data, ttm22f.M_site_coords, grads_E, true)
 
    add_electrostatic_and_dms_derivatives!(ttm22f, grads, grads_E, grads_q)

    return E_int + E_vdw + E_elec
end
