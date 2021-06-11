include("electrostatics.jl")
include("Partridge_Schwenke.jl")

mutable struct TTM21F
    num_waters::Int
    M_site_coords::Matrix{Float64} # @SPEED use StaticArrays

    constants::TTM21_Constants
    elec_data::Electrostatics
end

# You don't have to specify the number of waters ahead of time, but this will be fastest
TTM21F(num_waters::Int) = TTM21F(num_waters, zeros(3, 3 * num_waters), TTM21_Constants(), Electrostatics(zeros(Int(num_waters * 4)), TTM21_Constants()))

function compute_M_site_crd!(ttm21f::TTM21F, coords::AbstractMatrix{Float64})
    """
    Takes the 3xN coordinates you get coming in and stores the coordinates with
    M site in the TTM21F struct.
    """
    #@SPEED: Consider using Static arrays here.
    @assert isinteger(size(coords, 2) / 3) "Number of coordinates not divisble by three. Is this water?"
    nw_index::Int = 1
    for i in 1:size(new_coords, 2)
        if i % 4 == 0
            @views ttm21f.M_site_coords[:, i] = ttm21f.constants.γ_1 * ttm21f.M_site_coords[:,i-3] + ttm21f.constants.γ_2 * (ttm21f.M_site_coords[:,i-2] + ttm21f.M_site_coords[:, i-1])
        else
            @views ttm21f.M_site_coords[:, i] = coords[:, nw_index]
            nw_index += 1
        end
    end
end

#function charges!(ttm21f::TTM21F)
#end
function evaluate!(ttm21f::TTM21F, coords::AbstractMatrix{Float64}, grads::Union{Matrix{Float64}, Nothing}=nothing)
    """
    Evaluates the ttm21f energy of the system containing water called coords.
    """
    compute_M_site_crd!(ttm21f, coords)
    # @SPEED factor all of the gradient intermediates into a different function or block
    # which is only executed if the gradients are actually requested
    grads_q = convert(MArray{Tuple{3, 3, 3}, Float64, 3, 27}, zeros(27))
    grads_E = zero(grads)
    # calculate the INTRA-molecular energy and Dipole Moment
    # Surface (Partridge-Schwenke)

    E_int::Float64 = 0.0
    # @SPEED make threaded
    for n in 1:ttm21f.num_waters
        q3 = @MVector zeros(3)
        dq3 = convert(MArray{Tuple{3, 3, 3}, Float64, 3, 27}, zeros(27))

        if (grads !== nothing)
            E_int += pot_nasa(convert(SMatrix{3, 3, Float64}, @view(coords[:, (3*n-2):3*n])), grads)
            dms_nasa(convert(SMatrix{3, 3, Float64}, coords), q3, dq3)
        else
            E_int += pot_nasa(convert(SMatrix{3, 3, Float64}, @view(coords[:, (3*n-2):3*n])), nothing)
            dms_nasa(convert(SMatrix{3, 3, Float64}, coords), q3)
        end

        # TTM2.1-F assignment 
        tmp::Float64 = 0.5*gammaM / (1.0 - gammaM)
        ttm21f.elec_data.q[4*(n-1) + 1] = 0.0                                                   # O
        ttm21f.elec_data.q[4*(n-1) + 2] = q3[2] + tmp * (q3[2] + q3[3]) # H1
        ttm21f.elec_data.q[4*(n-1) + 3] = q3[3] + tmp * (q3[2] + q3[3]) # H2
        ttm21f.elec_data.q[4*(n-1) + 4] = q3[1] / (1.0 - gammaM)                                # M

        if (grads === nothing)
            continue
        end

        for k in 1:3
            grads_q[1, 1, k] = dq3[1, 1, k] + tmp*(dq3[1, 1, k] + dq3[1, 2, k])
            grads_q[2, 1, k] = dq3[2, 1, k] + tmp*(dq3[2, 1, k] + dq3[2, 2, k])
            grads_q[3, 1, k] = dq3[3, 1, k] + tmp*(dq3[3, 1, k] + dq3[3, 2, k])

            grads_q[1, 2, k] = dq3[1, 2, k] + tmp*(dq3[1, 2, k] + dq3[1, 1, k])
            grads_q[2, 2, k] = dq3[2, 2, k] + tmp*(dq3[2, 2, k] + dq3[2, 1, k])
            grads_q[3, 2, k] = dq3[3, 2, k] + tmp*(dq3[3, 2, k] + dq3[3, 1, k])

            grads_q[1, 3, k] = dq3[1, 3, k] - 2*tmp*(dq3[1, 1, k] + dq3[1, 2, k])
            grads_q[2, 3, k] = dq3[2, 3, k] - 2*tmp*(dq3[2, 1, k] + dq3[2, 2, k])
            grads_q[3, 3, k] = dq3[3, 3, k] - 2*tmp*(dq3[3, 1, k] + dq3[3, 2, k])
        end
    end
    ttm21f.elec_data.q *= CHARGECON
    
    if grads !== nothing
        grads_q *= CHARGECON
    end
    #---------------------------------------------------------------!
    # Calculate the vdW interactions for all atoms                  !
    #---------------------------------------------------------------!

    # PICK UP FROM HERE PORTING THINGS OVER STILL    

    E_vdw::Float64 = 0.0

    for (size_t i = 0; i < nw - 1; ++i) {
        for (size_t j = i + 1; j < nw; ++j) {
            // Calculate Oxygen-Oxygen interactions
            double dRijsq = 0.0, Rij[3];
            for (size_t k = 0; k < 3; ++k) {
                Rij[k] = crd[3*3*i + k] - crd[3*3*j + k];
                dRijsq += Rij[k]*Rij[k];
            }

            const double dRij = std::sqrt(dRijsq);
            const double expon = vdwD*std::exp(-vdwE*dRij);

            // Add Evdw van der Waal energy for O-O interaction
            const double x = 1/dRijsq;
            Evdw += expon + x*x*x*(vdwC + x*x*(vdwB + x*vdwA));

            if (grad) {
                const double tmp = vdwE*expon/dRij
                    + x*x*x*x*(6*vdwC + x*x*(10*vdwB + 12*x*vdwA));

                for (size_t k = 0; k < 3; ++k) {
                    grad[3*3*i + k] -= tmp*Rij[k];
                    grad[3*3*j + k] += tmp*Rij[k];
                }
            } // grad
        }
    }
end