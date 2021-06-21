using StaticArrays
using LinearAlgebra
using LoopVectorization
include("constants.jl")
include("smear.jl")

mutable struct Electrostatics
    natom::Int
    q::Vector{Float64} # 4 * nW
    dipoles::Vector{MVector{3, Float64}} # 3 * natom
    damping_fac::Vector{Float64} # 4 * nW
    α::Vector{Float64} # 4 * nW
    ϕ::Vector{Float64} # natom 
    E_field_q::Vector{MVector{3, Float64}} # 3 * natom elements
    E_field_dip::Vector{MVector{3, Float64}} # 3 * natom elements
    dip_dip_tensor::Vector{Float64} # natom * natom * 9 tensor 
    previous_dipoles::Vector{MVector{3, Float64}} # 3 * natom
    smear::Smear # smearing method
    dipoles_converged::Bool # convergence option
end

Electrostatics(charges::Vector{Float64}, C::TTM_Constants) =
Electrostatics(length(charges), charges, [@MVector zeros(3) for _ in  1:length(charges)], repeat([C.damping_factor_O, C.damping_factor_H, C.damping_factor_H, C.damping_factor_M], length(charges)),
repeat([C.α_O, C.α_H, C.α_H, C.α_M], length(charges)), zeros(length(charges)), [@MVector zeros(3) for _ in 1:length(charges)],
[@MVector zeros(3) for _ in 1:length(charges)], zeros(3 * 3 * length(charges) * length(charges)), [@MVector zeros(3) for _ in  1:length(charges)], get_smear_type(C.name), false)

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

@inline function skip_ij(i::Int, j::Int, only_M_sites::Bool=false)
    """
    Determine if i and j are a part of the same water.
    Since everything has to be in OHHM order, we can just infer from the indices.
    Note that this won't check if your input is valid (all indices should be >= 1).
    We also provide an option for filtering based on if the M-site is the only polarizable considered. This is the case in the TTM3-F potential.
    """
    if only_M_sites
        return (i % 4 != 0) || (j % 4 != 0) || ((i-1) ÷ 4 == (j-1) ÷ 4)
    else
        return (i-1) ÷ 4 == (j-1) ÷ 4
    end
end

function form_dipole_dipole_tensor!(elec_data::Electrostatics, coords::Vector{SVector{3, Float64}}, only_M_sites::Bool)
    num_α_sites::Int = 1
    atom_stride::Int = 1
    if only_M_sites
        num_α_sites = 4
        atom_stride = 4
    end

    dd3 = @MMatrix zeros(3, 3)
    for i in atom_stride:atom_stride:(elec_data.natom-1)
        i3::Int = 3 * (i - 1) + 1
        for j in (i+atom_stride):atom_stride:elec_data.natom
            j3::Int = 3 * (j - 1) + 1
            @views r_ij = coords[i] - coords[j]

            aDD::Float64 = skip_ij(i, j) ? elec_data.smear.aDD_intramolecular : elec_data.smear.aDD_intermolecular

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

            # make dip dip a matrix of 3x3 MMatrixs
            # Look at the fortran code
            for k in 0:2
                for l in 0:2
                   @inbounds elec_data.dip_dip_tensor[3 * elec_data.natom * (i3 + k - 1) + j3 + l] = dd3[k+1, l+1]
                   @inbounds elec_data.dip_dip_tensor[3 * elec_data.natom * (j3 + k - 1) + i3 + l] = dd3[k+1, l+1]
                end
            end
        end
    end
end

function charge_charge_interactions!(elec_data::Electrostatics, coords::Vector{SVector{3, Float64}})
    for i in 1:(elec_data.natom-1)
        for j in (i+1):elec_data.natom
            @views r_ij = coords[i] - coords[j]

            # charge-charge
            if !skip_ij(i, j)
                ts0, ts1 = elec_data.smear.smear1(r_ij ⋅ r_ij, elec_data.damping_fac[i] * elec_data.damping_fac[j], elec_data.smear.aCC)
                
                @inbounds elec_data.ϕ[i] += ts0 * elec_data.q[j]
                @inbounds elec_data.ϕ[j] += ts0 * elec_data.q[i]
                
                @inbounds elec_data.E_field_q[i] += ts1 * elec_data.q[j] * r_ij
                @inbounds elec_data.E_field_q[j] -= ts1 * elec_data.q[i] * r_ij
            end
        end
    end
end

# OPTIMIZE EVERYTHING HERE
# get the properly sized matrices
# use static vectors wherever possible
# where is all the allocation happening??
# get rid of all the intermediates and loops over x,y,z
#
# Some of the interactions are completely independent. For instance, try separating the charge-charge interactions and formation of the ddt into separate tasks handled by their own threads.
function electrostatics(elec_data::Electrostatics, coords::Vector{SVector{3, Float64}}, grads_E::Union{Vector{MVector{3, Float64}}, Nothing}=nothing, only_M_sites::Bool=false)
    reset_electrostatics!(elec_data)

    num_α_sites::Int = 1
    atom_stride::Int = 1
    if only_M_sites
        num_α_sites = 4
        atom_stride = 4
    end

    # @SPEED make these separate tasks that run independently
    # and then join afterwards
    charge_charge_interactions!(elec_data, coords)
    form_dipole_dipole_tensor!(elec_data, coords, only_M_sites)

    # This seems to be a not much better guess than just
    # starting from zeros. 
    # Using the dipoles from last call to the potential is the best
    # when one can be sure the dipoles are similar from call to call.
    # Perhaps there should be "MD/MC Mode" to hint that the
    # dipoles will be similar from call to call.

    # Calculate induced dipoles iteratively
    #for i in 1:elec_data.natom
    #    @inbounds elec_data.dipoles[i] = elec_data.α[1:3] .* elec_data.E_field_q[i]
    #    @inbounds elec_data.previous_dipoles[i] = elec_data.α[1:3] .* elec_data.E_field_q[i]
    #end
    
    dmix::Float64 = 0.75
    stath::Float64 = DEBYE / CHARGECON / sqrt(elec_data.natom)
    
    elec_data.dipoles_converged = false
    
    for iter in 1:dipole_maxiter
        xyz = 1
        xyz_outer = 1
        # @SPEED this whole thing is possible by multiplying the tensor blocks by the dipole matrix or something like that. Should be much faster than it is now
        for k in 1:3*elec_data.natom
            for l in 1:3*elec_data.natom
                @inbounds elec_data.E_field_dip[(k-1)÷3+1][(k+2)%3+1] += elec_data.dip_dip_tensor[3 * elec_data.natom * (k - 1) + l] * elec_data.dipoles[(l-1)÷3+1][(l+2)%3+1]
                xyz == 3 ? xyz = 1 : xyz += 1 
            end
            xyz_outer == 3 ? xyz_outer = 1 : xyz_outer += 1 
        end
    
        deltadip::Float64 = 0.0
    
        for i in 1:elec_data.natom
            @inbounds elec_data.dipoles[i] = elec_data.α[i] * (elec_data.E_field_q[i] + elec_data.E_field_dip[i])
        end

        @inbounds elec_data.dipoles = dmix * elec_data.dipoles + (1.0 - dmix) * elec_data.previous_dipoles
        delta = sum(sum.(elec_data.dipoles - elec_data.previous_dipoles))
    
        deltadip = sqrt(delta^2) * stath
    
        if deltadip < dipole_tolerance
            elec_data.dipoles_converged = true
            break # converged!
        else
            copy!.(elec_data.previous_dipoles, elec_data.dipoles)
            elec_data.E_field_dip -= elec_data.E_field_dip
        end
    end

    E_elec::Float64 = 0.0
    for i in 1:elec_data.natom
        @inbounds E_elec += elec_data.ϕ[i] * elec_data.q[i]
    end
    E_elec *= 0.5
    
    E_ind::Float64 = 0.0
    for i in 1:elec_data.natom
        @inbounds E_ind -= elec_data.dipoles[i] ⋅ elec_data.E_field_q[i]
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
    for i in 1:elec_data.natom
        for j in 1:elec_data.natom
            if (skip_ij(i, j))
                continue # skip this (i, j) pair
            end

            Rij = coords[i] - coords[j]
            diR = elec_data.dipoles[i] ⋅ Rij

            @inbounds ts1, ts2 = elec_data.smear.smear2(Rij ⋅ Rij, elec_data.damping_fac[i] * elec_data.damping_fac[j], elec_data.smear.aCD);

            @inbounds derij = elec_data.q[j] * (3 * ts2 * diR * Rij - ts1 * elec_data.dipoles[i])
            
            @inbounds grads_E[i] += derij
            @inbounds grads_E[j] -= derij

            elec_data.ϕ[j] -= ts1 * diR
        end
    end

    # dipole-dipole interactions
    for i in 1:elec_data.natom-1
        for j in i+1:elec_data.natom
            Rij = coords[i] - coords[j]
            @inbounds diR  = elec_data.dipoles[i] ⋅ Rij
            @inbounds djR  = elec_data.dipoles[j] ⋅ Rij
            @inbounds didj = elec_data.dipoles[i] ⋅ elec_data.dipoles[j]

            aDD = skip_ij(i, j) ? elec_data.smear.aDD_intramolecular : elec_data.smear.aDD_intermolecular

            _, ts2, ts3 = elec_data.smear.smear3(Rij ⋅ Rij, elec_data.damping_fac[i] * elec_data.damping_fac[j], aDD)
            
            @inbounds derij = - 3 * ts2 * (didj * Rij + djR * elec_data.dipoles[i] + diR * elec_data.dipoles[j]) + 15 * ts3 * diR * djR * Rij
            @inbounds grads_E[i] += derij
            @inbounds grads_E[j] -= derij
        end
    end
    return E_elec + E_ind
end
