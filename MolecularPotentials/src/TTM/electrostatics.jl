using StaticArrays
include("constants.jl")
include("smear.jl")

struct Electrostatics
    natom::Int
    q::SVector{natom, Float64}
    damping_fac::SVector{natom, Float64}
    α::Vector{SVector{3, Float64}}
    ϕ::Vector{SVector{3, Float64}}
    E_field_q::SVector{natom, Float64}
    E_field_dip::Vector{SVector{3, Float64}}
    dip_dip_tensor::Vector{SVector{3, Float64}}
    
    # storage
    α_sqrt::Vector{SVector{3, Float64}}
    dipoles_converged::Bool
    smear::Smear
end

function get_α_sqrt(α::Vector{SVector{3, Float64}})
    return [sqrt.(α[i]) for i in 1:length(α)]
end

