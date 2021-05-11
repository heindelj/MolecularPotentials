using MolecularPotentials

mutable struct Water{T} <: AbstractMolecule
    coords::SMatrix{3, 3, Float64}
end