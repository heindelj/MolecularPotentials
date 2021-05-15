using StaticArrays
include("constants.jl")
include("smear.jl")

mutable struct Electrostatics
    natom::Int
    q::MVector{natom, Float64}
    damping_fac::MVector{natom, Float64}
    α::Vector{MVector{3, Float64}}
    ϕ::Vector{MVector{3, Float64}}
    E_field_q::MVector{natom, Float64}
    E_field_dip::Vector{MVector{3, Float64}}
    dip_dip_tensor::Vector{MVector{3, Float64}}
    
    # storage
    α_sqrt::Vector{MVector{3, Float64}}
    dipoles_converged::Bool
    smear::Smear
end

function get_α_sqrt(α::Vector{MVector{3, Float64}})
    return [sqrt.(α[i]) for i in 1:length(α)]
end

function ij_bonded(i::Int, j::Int)
    """
    Determine if i and j are a part of the same water.
    Since everything has to be in OHH order, we can just infer from the indices.
    """
    (i-1) ÷ 3 == (j-1) ÷ 3
end

function electrostatics(coords::SMatrix{3, 3, Float64}, grads::Union{MMatrix{3, 3, Float64}, Nothing}=nothing, elec_data::Electrostatics)
    elec_data.α_sqrt = get_α_sqrt(elec_data.α)

    for i in 1:(elec_data.natom-1)
        for j in (i+1):elec_data.natom
            @views r_ij_sq::Float64 = norm(coords[:, i] - coords[:, j])^2

            # charge-charge

        end
    end
end