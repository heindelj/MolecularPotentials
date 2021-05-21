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
    ϕ::Vector{Float64} # natom ?
    E_field_q::Vector{MVector{3, Float64}} # 3 * natom elements
    E_field_dip::Vector{MVector{3, Float64}} # 3 * natom elements
    dip_dip_tensor::Vector{Float64} # natom * natom * 9 tensor 
    smear::Smear # smearing method
    dipoles_converged::Bool # convergence option
end

Electrostatics(charges::Vector{Float64}, C::TTM_Constants) =
Electrostatics(length(charges), charges, zeros(3 * length(charges)), repeat([C.damping_factor_O, C.damping_factor_H, C.damping_factor_H, C.damping_factor_M], length(charges)),
repeat([C.α_O, C.α_H, C.α_H, C.α_M], length(charges)), zeros(length(charges)), [@MVector zeros(3) for _ in 1:length(charges)],
[@MVector zeros(3) for _ in 1:length(charges)], zeros(3 * 3 * length(charges) * length(charges)), get_smear_type(C.name), false)

function get_α_sqrt(α::Vector{Float64})
    α_sqrt = zeros(4 * length(α))
    for i in 1:length(α)
        for j in 1:4
            α_sqrt[4*(i-1)+j] = sqrt(α[i])
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

function electrostatics(elec_data::Electrostatics, coords::Matrix{Float64}, grads::Union{Matrix{Float64}, Nothing}=nothing)
    α_sqrt = get_α_sqrt(elec_data.α)

    # should be possible to wrap in @distributed or @Threads.threads()
    for i in 1:(elec_data.natom-1)
        i3::Int = 3 * (i - 1)
        for j in (i+1):elec_data.natom
            j3::Int = 3 * (j - 1)
            @views r_ij::Vector{Float64} = coords[:, i] - coords[:, j]
            r_ij_sq::Float64 = r_ij ⋅ r_ij

            # charge-charge
            if !ij_bonded(i, j)
                ts0, ts1 = elec_data.smear.smear1(r_ij_sq, elec_data.damping_fac[i] * elec_data.damping_fac[j], elec_data.smear.aCC)

                elec_data.ϕ[i] += ts0 * elec_data.q[j]
                elec_data.ϕ[j] += ts0 * elec_data.q[i]

                for k in 1:3
                    elec_data.E_field_q[i][k] += ts1 * elec_data.q[j] * r_ij[k]
                    elec_data.E_field_q[j][k] -= ts1 * elec_data.q[i] * r_ij[k]
                end
            end

            println(elec_data.E_field_q[j][:])

            # dipole-dipole tensor
            aDD::Float64 = ij_bonded(i, j) ? elec_data.smear.aDD_intramolecular : elec_data.smear.aDD_intermolecular

            ts1, ts2 = elec_data.smear.smear2(r_ij_sq, elec_data.damping_fac[i] * elec_data.damping_fac[j], aDD)

            dd3 = @MMatrix zeros(3, 3)

            dd3[1, 1] = 3.0 * ts2 * r_ij[1] * r_ij[1] - ts1
            dd3[2, 2] = 3.0 * ts2 * r_ij[2] * r_ij[2] - ts1
            dd3[3, 3] = 3.0 * ts2 * r_ij[3] * r_ij[3] - ts1
            dd3[1, 2] = 3.0 * ts2 * r_ij[1] * r_ij[2]
            dd3[1, 3] = 3.0 * ts2 * r_ij[1] * r_ij[3]
            dd3[2, 3] = 3.0 * ts2 * r_ij[2] * r_ij[3]
            dd3[2, 1] = dd3[1, 2]
            dd3[3, 1] = dd3[1, 3]
            dd3[3, 2] = dd3[2, 3]

            aiaj::Float64 = α_sqrt[i] * α_sqrt[j]
            for k in 1:3
                for l in 1:3
                    elec_data.dip_dip_tensor[3 * elec_data.natom * (i3 + k) + j3 + l] = -aiaj * dd3[k, l]
                end
            end
        end
    end

    # convert ddt into an actual tensor later on and figure out
    # how to use the library cholesky decomposition

    # populate the diagonal
    for i in 1:(3 * elec_data.natom)
        elec_data.dip_dip_tensor[3 * elec_data.natom*(i-1) + i] = 1.0
    end

    # perform Cholesky decomposition ddt = L*L^T
    # storing L in the lower triangle of ddt and
    # its diagonal in Efd

    natom3::Int = 3 * elec_data.natom

    for i in 1:natom3
        for j in i:natom3
            sum__::Float64 = elec_data.dip_dip_tensor[i*natom3 + j]
            for k in 1:i
                sum__ -= elec_data.dip_dip_tensor[i*natom3 + k] * elec_data.dip_dip_tensor[j*natom3 + k]
            end
            if i == j
                @assert sum__ >= 0.0 "Sum is not greater than zero."
                elec_data.E_field_dip[i] = sqrt(sum__)
            else
                elec_data.dip_dip_tensor[j*natom3 + i] = sum__ / elec_data.E_field_dip[i]
            end
        end
    end

    # solve L*y = sqrt(a)*Efq storing y in dipole[]
    for i in 1:natom3
        sum__::Float64 = α_sqrt[i] * elec_data.E_field_q[i]
        for k in 1:i
            sum__ -= elec_data.dip_dip_tensor[i*natom3 + k] * elec_data.dipole[k]
        end

        elec_data.dipole[i] = sum__ / elec_data.E_field_dip[i]
    end

    # solve L^T*x = y
    for i in natom3:-1:1
        i1::Int = i - 1
        sum__::Float64 = elec_data.dipole[i1]
        for k in i:natom3
            sum__ -= elec_data.dip_dip_tensor[k*natom3 + i1] * elec_data.dipole[k]
        end
        elec_data.dipole[i1] = sum__ / elec_data.E_field_dip[i1]
    end

    for i in 1:natom3
        elec_data.dipole[i] *= polar_sqrt[i]
    end

    elec_data.dipoles_converged = true

    E_elec::Float64 = 0.0
    for i in 1:elec_data.natom
        E_elec += elec_data.ϕ[i] * elec_data.q[i]
    end
    E_elec *= 0.5

    E_ind::Float64 = 0.0
    for i in 1:3*elec_data.natom
        E_ind -= elec_data.dipole[i] * elec_data.E_field_q[i]
    end
    E_ind *= 0.5

    # if no gradients requested, then return the energy here
    if grads === nothing
        return E_elec + E_ind
    end

    # do the gradients

end