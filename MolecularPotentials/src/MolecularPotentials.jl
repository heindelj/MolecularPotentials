module MolecularPotentials
    using StaticArrays
    using Parameters
    include("AbstractMolecule.jl")
    include("AbstractPotential.jl")
    include("/home/heindelj/dev/julia_development/wally/src/molecule_tools/units.jl")
    include("/home/heindelj/dev/julia_development/wally/src/molecule_tools/read_xyz.jl")
end