include("smearing_functions.jl")
abstract type Smear end

struct Smear_TTM3 <: Smear
    aCC::Float64
    aCD::Float64
    aDD_intermolecular::Float64
    aDD_intramolecular::Float64
    smear1::Function
    smear2::Function
    smear3::Function
    Smear_TTM3() = new(0.175, 0.175, 0.175, 0.175, smear1_ttm3, smear2_ttm3, smear3_ttm3)
end

struct Smear_TTM21 <: Smear
    aCC::Float64
    aCD::Float64
    aDD_intermolecular::Float64
    aDD_intramolecular::Float64
    smear1::Function
    smear2::Function
    smear3::Function
    Smear_TTM21() = new(0.2, 0.2, 0.3, 0.3, smear1_ttm3, smear2_ttm3, smear3_ttm3)
end

struct Smear_TTM4 <: Smear
    aCC::Float64
    aCD::Float64
    aDD_intermolecular::Float64
    aDD_intramolecular::Float64
    smear1::Function
    smear2::Function
    smear3::Function
    Smear_TTM4() = new(0.4, 0.4, 0.055, 0.626)
end