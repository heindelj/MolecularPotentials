include("electrostatics.jl")
include("Partridge_Schwenke.jl")

struct TTM21F
    num_waters::Int
    constants::TTM21_Constants
    elec_data::Electrostatics
end

TTM21F(num_waters::Int) = TTM21F(num_waters, TTM21_Constants(), Electrostatics(zeros(Int(num_waters * 4 / 3)), TTM21_Constants())

function compute_M_site_crd(coords::Matrix{Float64}, γ_1::Float64, γ_2::Float64)
    """
    Takes the 3xN coordinates you get coming in and returns the coordinates
    with the M-site insered in OHHM order.
    """
    #SPEED: Consider using Static arrays here.
    @assert isinteger(size(coords, 2) / 3) "Number of coordinates not divisble by three. Is this water?"
    new_coords = zeros(3, Int(4 * size(coords, 2) / 3))
    nw_index::Int = 1
    for i in 1:size(new_coords, 2)
        if i % 4 == 0
            @views new_coords[:, i] = γ_1 * new_coords[:,i-3] + γ_2 * (new_coords[:,i-2] + new_coords[:, i-1])
        else
            @views new_coords[:, i] = coords[:, nw_index]
            nw_index += 1
        end
    end
    return new_coords
end

function get_energy(ttm21f::TTM21F, coords::Matrix{Float64})
    """
    Evaluates the ttm21f energy of the system containing water called coords.
    """
    coords_with_M = compute_M_site_crd(coords, ttm21f.constants.γ_1, ttm21f.constants.γ_2)
    # HERE! #
end