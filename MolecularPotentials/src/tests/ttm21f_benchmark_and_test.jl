include("../TTM/ttm21f.jl")
include("/home/heindelj/dev/julia_development/wally/src/molecule_tools/read_xyz.jl")
using BenchmarkTools, DelimitedFiles, Test

header, labels, geoms = read_xyz("/home/heindelj/dev/julia_development/MolecularPotentials/Reference_Implementation/ttm-water-models-04-Jul-2013/hexamers/W20_fused_prisms.xyz")
ttm21f = TTM21F(20)
grads = zeros(3, 60)
energy = evaluate!(ttm21f, geoms[1], grads)
@btime evaluate!(ttm21f, geoms[1], grads)

correct_energy = -215.22312443016003 
correct_grads  = readdlm("/home/heindelj/OneDrive/Documents/Coding_Projects/julia_development/MolecularPotentials/MolecularPotentials/src/tests/w20_grads_correct.txt", Float64)
Test.@test energy ≈ correct_energy
Test.@test grads  ≈ correct_grads