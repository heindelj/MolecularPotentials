using StaticArrays
using LinearAlgebra
include("constants.jl")
include("smear.jl")

mutable struct Electrostatics
    natom::Int
    q::Vector{Float64} # 4 * nW
    dipoles::Vector{Float64} # 3 * natom
    damping_fac::Vector{Float64} # 4 * nW
    α::Vector{Float64} # 4 * nW
    ϕ::Vector{Float64} # natom 
    E_field_q::Vector{MVector{3, Float64}} # 3 * natom elements
    E_field_dip::Vector{Float64} # 3 * natom elements
    dip_dip_tensor::Vector{Float64} # natom * natom * 9 tensor 
    smear::Smear # smearing method
    dipoles_converged::Bool # convergence option
end

Electrostatics(charges::Vector{Float64}, C::TTM_Constants) =
Electrostatics(length(charges), charges, zeros(3 * length(charges)), repeat([C.damping_factor_O, C.damping_factor_H, C.damping_factor_H, C.damping_factor_M], length(charges)),
repeat([C.α_O, C.α_H, C.α_H, C.α_M], length(charges)), zeros(length(charges)), [@MVector zeros(3) for _ in 1:length(charges)],
zeros(3 * length(charges)), zeros(3 * 3 * length(charges) * length(charges)), get_smear_type(C.name), false)

function reset_electrostatics!(elec_data::Electrostatics)
    """
    Resets the electrostatics struct so as to not have problems repeating the calculations
    with the same container.
    """
    elec_data.ϕ -= elec_data.ϕ
    elec_data.E_field_q -= elec_data.E_field_q
    elec_data.E_field_dip -= elec_data.E_field_dip
    elec_data.dip_dip_tensor -= elec_data.dip_dip_tensor
end 

function get_α_sqrt(α::Vector{Float64})
    α_sqrt = zeros(4 * length(α))
    for i in 1:length(α)
        for j in 1:3
            @inbounds α_sqrt[3*(i-1)+j] = sqrt(α[i])
        end
    end
    return α_sqrt
end

function ij_bonded(i::Int, j::Int)
    """
    Determine if i and j are a part of the same water.
    Since everything has to be in OHHM order, we can just infer from the indices.
    Note that this won't check if your input is valid (all indices should be >= 1).
    """
    return (i-1) ÷ 4 == (j-1) ÷ 4
end

# OPTIMIZE EVERYTHING HERE
# get the properly sized matrices
# use static vectors wherever possible
# where is all the allocation happening??
# get rid of all the intermediates and loops over x,y,z
function electrostatics(elec_data::Electrostatics, coords::Vector{SVector{3, Float64}}, grads_E::Union{Vector{MVector{3, Float64}}, Nothing}=nothing, use_cholesky::Bool=false)
    if use_cholesky
        α_sqrt = get_α_sqrt(elec_data.α)
    else
        previous_dipoles = zero(elec_data.dipoles)
    end
    reset_electrostatics!(elec_data)

    # should be possible to wrap in @distributed or @Threads.threads()
    for i in 1:(elec_data.natom-1)
        dd3 = @MMatrix zeros(3, 3)
        i3::Int = 3 * (i - 1) + 1
        for j in (i+1):elec_data.natom
            j3::Int = 3 * (j - 1) + 1
            @views r_ij = coords[i] - coords[j]

            # charge-charge
            if !ij_bonded(i, j)
                ts0, ts1 = elec_data.smear.smear1(r_ij ⋅ r_ij, elec_data.damping_fac[i] * elec_data.damping_fac[j], elec_data.smear.aCC)
                
                @inbounds elec_data.ϕ[i] += ts0 * elec_data.q[j]
                @inbounds elec_data.ϕ[j] += ts0 * elec_data.q[i]
                
                for k in 1:3
                    @inbounds elec_data.E_field_q[i][k] += ts1 * elec_data.q[j] * r_ij[k]
                    @inbounds elec_data.E_field_q[j][k] -= ts1 * elec_data.q[i] * r_ij[k]
                end
            end

            # dipole-dipole tensor
            aDD::Float64 = ij_bonded(i, j) ? elec_data.smear.aDD_intramolecular : elec_data.smear.aDD_intermolecular

            ts1, ts2 = elec_data.smear.smear2(r_ij ⋅ r_ij, elec_data.damping_fac[i] * elec_data.damping_fac[j], aDD)

            @inbounds dd3[1, 1] = 3.0 * ts2 * r_ij[1] * r_ij[1] - ts1
            @inbounds dd3[2, 2] = 3.0 * ts2 * r_ij[2] * r_ij[2] - ts1
            @inbounds dd3[3, 3] = 3.0 * ts2 * r_ij[3] * r_ij[3] - ts1
            @inbounds dd3[1, 2] = 3.0 * ts2 * r_ij[1] * r_ij[2]
            @inbounds dd3[1, 3] = 3.0 * ts2 * r_ij[1] * r_ij[3]
            @inbounds dd3[2, 3] = 3.0 * ts2 * r_ij[2] * r_ij[3]
            @inbounds dd3[2, 1] = dd3[1, 2]
            @inbounds dd3[3, 1] = dd3[1, 3]
            @inbounds dd3[3, 2] = dd3[2, 3]

            if use_cholesky
                aiaj::Float64 = α_sqrt[i3] * α_sqrt[j3]
                for k in 0:2
                    for l in 0:2
                        @inbounds elec_data.dip_dip_tensor[3 * elec_data.natom * (i3 + k - 1) + j3 + l] = -aiaj * dd3[k+1, l+1]
                    end
                end
            else
                for k in 0:2
                    for l in 0:2
                        elec_data.dip_dip_tensor[3 * elec_data.natom * (i3 + k - 1) + j3 + l] = dd3[k+1, l+1]
                        elec_data.dip_dip_tensor[3 * elec_data.natom * (j3 + k - 1) + i3 + l] = dd3[k+1, l+1]
                    end
                end
            end
        end
    end

    # convert ddt into an actual tensor later on and figure out
    # how to use the library cholesky decomposition
    if use_cholesky
        # populate the diagonal
        for i in 1:(3 * elec_data.natom)
            @inbounds elec_data.dip_dip_tensor[3 * elec_data.natom*(i-1) + i] = 1.0
        end

        # perform Cholesky decomposition ddt = L*L^T
        # storing L in the lower triangle of ddt and
        # its diagonal in Efd
    
        natom3::Int = 3 * elec_data.natom
    
        for i in 0:natom3-1
            for j in i:natom3-1
                sum__::Float64 = elec_data.dip_dip_tensor[i*natom3 + j + 1]
                for k in 0:(i-1)
                    @inbounds sum__ -= elec_data.dip_dip_tensor[i*natom3 + k + 1] * elec_data.dip_dip_tensor[j*natom3 + k + 1]
                end
                if i == j
                    @inbounds elec_data.E_field_dip[i+1] = sqrt(sum__)
                else
                    @inbounds elec_data.dip_dip_tensor[j*natom3 + i + 1] = sum__ / elec_data.E_field_dip[i+1]
                end
            end
        end
    
        # solve L*y = sqrt(a)*Efq storing y in dipoles
        xyz::Int = 1
        for i in 0:natom3-1
            @inbounds sum__::Float64 = α_sqrt[i+1] * elec_data.E_field_q[(i ÷ 3) + 1][xyz]
            for k in 1:i
                @inbounds sum__ -= elec_data.dip_dip_tensor[i*natom3 + k] * elec_data.dipoles[k]
            end
            @inbounds elec_data.dipoles[i+1] = sum__ / elec_data.E_field_dip[i+1]
            xyz == 3 ? xyz = 1 : xyz += 1 
        end
    
        # solve L^T*x = y
        for i in natom3+1:-1:2
            i1::Int = i - 1
            @inbounds sum__::Float64 = elec_data.dipoles[i1]
            for k in i:natom3
            @inbounds  sum__ -= elec_data.dip_dip_tensor[(k-1)*natom3 + i1] * elec_data.dipoles[k]
            end
            @inbounds elec_data.dipoles[i1] = sum__ / elec_data.E_field_dip[i1]
        end
    
        for i in 1:natom3
            @inbounds elec_data.dipoles[i] *= α_sqrt[i]
        end
    
        elec_data.dipoles_converged = true
    else
        # Calculate induced dipoles iteratively
        for i in 1:elec_data.natom
            i3::Int = 3 * (i-1)
            for k in 1:3
                elec_data.dipoles[i3 + k] = elec_data.α[k] * elec_data.E_field_q[i][k]
                previous_dipoles[i3 + k]  = elec_data.dipoles[i3 + k]
            end
        end
    
        dmix::Float64 = 0.7
        stath::Float64 = DEBYE / CHARGECON / sqrt(elec_data.natom)
    
        elec_data.dipoles_converged = false
    
        for iter in 1:dipole_maxiter
            for k in 1:3*elec_data.natom
                for l in 1:3*elec_data.natom
                    elec_data.E_field_dip[k] += elec_data.dip_dip_tensor[3 * elec_data.natom * (k - 1) + l] * elec_data.dipoles[l]
                end
            end
    
            deltadip::Float64 = 0.0
    
            for i in 1:elec_data.natom
                i3::Int = 3 * (i - 1)
                for k in 1:3
                    elec_data.dipoles[i3 + k] = elec_data.α[i] * (elec_data.E_field_q[i][k] + elec_data.E_field_dip[i3 + k])
                    elec_data.dipoles[i3 + k] = dmix * elec_data.dipoles[i3 + k] + (1.0 - dmix) * previous_dipoles[i3 + k]
        
                    delta = elec_data.dipoles[i3 + k] - previous_dipoles[i3 + k]
                    deltadip += delta * delta
                end
            end
    
            deltadip = sqrt(deltadip) * stath
    
            if deltadip < dipole_tolerance
                elec_data.dipoles_converged = true
                break # converged!
            else
                copy!(previous_dipoles, elec_data.dipoles)
                elec_data.E_field_dip -= elec_data.E_field_dip
            end
        end
    end

    E_elec::Float64 = 0.0
    for i in 1:elec_data.natom
        @inbounds E_elec += elec_data.ϕ[i] * elec_data.q[i]
    end
    E_elec *= 0.5
    
    E_ind::Float64 = 0.0
    for i in 1:3*elec_data.natom
        @inbounds E_ind -= elec_data.dipoles[i] * elec_data.E_field_q[(i+2) ÷ 3][((i-1) % 3)+1]
    end
    E_ind *= 0.5

    # if no gradients requested, then return the energy here
    if grads_E === nothing
        return E_elec + E_ind
    end

    ### CALCULATE THE GRADIENTS ###
    # charge-charge interactions
    for i in 1:elec_data.natom
        @inbounds grads_E[i] = -elec_data.q[i] * elec_data.E_field_q[i]
    end
    
    # charge-dipole interactions
    derij::Float64 = 0.0
    for i in 1:elec_data.natom
        i3::Int = 3 * (i - 1)
        for j in 1:elec_data.natom
            @inbounds qj::Float64 = elec_data.q[j]

            skip_ij = ij_bonded(i, j)

            if (skip_ij)
                continue # skip this (i, j) pair
            end

            diR::Float64 = 0.0
            Rij = coords[i] - coords[j]
            for k in 1:3
               @inbounds diR += elec_data.dipoles[i3 + k] * Rij[k]
            end

            ts1, ts2 = elec_data.smear.smear2(Rij ⋅ Rij, elec_data.damping_fac[i] * elec_data.damping_fac[j], elec_data.smear.aCD);

            for k in 1:3
                @inbounds derij = qj * (3 * ts2 * diR * Rij[k] - ts1 * elec_data.dipoles[i3 + k])

                @inbounds grads_E[i][k] += derij
                @inbounds grads_E[j][k] -= derij
            end

            elec_data.ϕ[j] -= ts1 * diR
        end
    end

    # dipole-dipole interactions
    for i in 1:elec_data.natom-1
        i3::Int = 3 * (i - 1)
        for j in i+1:elec_data.natom
            j3::Int = 3 * (j - 1)
            
            diR::Float64 = 0.0
            djR:: Float64  = 0.0
            didj::Float64  = 0.0
            Rij = coords[i] - coords[j]
            for k in 1:3
                diR  += elec_data.dipoles[i3 + k] * Rij[k]
                djR  += elec_data.dipoles[j3 + k] * Rij[k]
                didj += elec_data.dipoles[i3 + k] * elec_data.dipoles[j3 + k]
            end

            aDD = ij_bonded(i, j) ? elec_data.smear.aDD_intramolecular : elec_data.smear.aDD_intermolecular

            ts1, ts2, ts3 = elec_data.smear.smear3(Rij ⋅ Rij, elec_data.damping_fac[i] * elec_data.damping_fac[j], aDD)
            
            for k in 1:3
                @inbounds derij = - 3 * ts2 * (didj * Rij[k] + djR * elec_data.dipoles[i3 + k] + diR * elec_data.dipoles[j3 + k]) + 15 * ts3 * diR * djR * Rij[k]
                @inbounds grads_E[i][k] += derij
                @inbounds grads_E[j][k] -= derij
            end

        end
    end
 
    return E_elec + E_ind
end