using MolecularPotentials
using StaticArrays
include("constants.jl")

let # scope all the constants we need for this potential
    global pot_nasa

    C = PS_Constants # just an alias for convenience cause C === PS_Constants

    function pot_nasa(coords::SMatrix{3, 3, Float64}, grads::Union{MMatrix{3, 3, Float64}, Nothing}=nothing)
        rOH1::SVector{3, Float64} = coords[:, 1] - coords[:, 2]
        rOH2::SVector{3, Float64} = coords[:, 1] - coords[:, 3]
        rHH ::SVector{3, Float64} = coords[:, 2] - coords[:, 3]

        cosθ::Float64 = rOH1 ⋅ rOH2 / (norm(rOH1) * norm(rOH2))
        
        deoh::Float64   = C.f5z * C.deohA
        phh1::Float64   = C.f5z * C.phh1A * exp(C.phh2)
        cosθ_e::Float64 = -.24780227221366464506
        exp1::Float64   = exp(-C.alphaoh * (norm(rOH1) - C.roh))
        exp2::Float64   = exp(-C.alphaoh * (norm(rOH2) - C.roh))
        Va::Float64     = deoh * (exp1 * (exp1 - 2.0) + exp2 * (exp2 - 2.0))
        Vb::Float64     = phh1 * exp(-C.phh2 * norm(rHH))
        dVa1::Float64   = 2.0 * C.alphaoh * deoh * exp1 * (1.0 - exp1) / norm(rOH1)
        dVa2::Float64   = 2.0 * C.alphaoh * deoh * exp2 * (1.0 - exp2) / norm(rOH2)
        dVb::Float64    = -C.phh2 * Vb / norm(rHH)
        x1::Float64     = (norm(rOH1) - C.reoh) / C.reoh
        x2::Float64     = (norm(rOH2) - C.reoh) / C.reoh
        x3::Float64     = cosθ - cosθ_e

        # SPEED: might be able to speed this up by making it 16,3?
        fmat = @MMatrix zeros(3, 16)
        for i in 1:3
            fmat[i, 1] = 0.0
            fmat[i, 2] = 1.0
        end

        for i in 3:16
            fmat[1, i] = fmat[1, i-1] * x1
            fmat[2, i] = fmat[2, i-1] * x2
            fmat[3, i] = fmat[3, i-1] * x3
        end

        efac::Float64 = exp(-C.b1 * ((norm(rOH1) - C.reoh)^2 + (norm(rOH2) - C.reoh)^2))
        
        sum0::Float64 = 0.0
        sum1::Float64 = 0.0
        sum2::Float64 = 0.0
        sum3::Float64 = 0.0

        for j in 1:244
            inI::Int = idx1[j+1]
            inJ::Int = idx2[j+1]
            inK::Int = idx3[j+1]
    
            sum0 += C.c5z[j+1]*(fmat[1,inI+1] * fmat[2, inJ+1] + fmat[1, inJ+1]*fmat[2, inI+1])*fmat[3, inK+1]
    
            sum1 += C.c5z[j+1]*((inI - 1)*fmat[1, inI]*fmat[2, inJ+1] + (inJ - 1)*fmat[1, inJ]*fmat[2, inI+1])*fmat[3, inK+1]
    
            sum2 += C.c5z[j+1]*((inJ - 1)*fmat[1, inI+1]*fmat[2, inJ] + (inI - 1)*fmat[1, inJ+1]*fmat[2, inI])*fmat[3, inK+1]
    
            sum3 += C.c5z[j+1]*(fmat[1, inI+1]*fmat[2, inJ+1] + fmat[1, inJ+1]*fmat[2, inI+1]) *(inK - 1)*fmat[3, inK]
        end
        
        # Energy
        Vc::Float64 = 2 * C.c5z[1] + efac * sum0;
        energy::Float64 = Va + Vb + Vc;

        energy += 0.44739574026257 # correction
        energy *= conversion(:wavenumbers, :hartree)
        if grads !== nothing
            dVcdr1::Float64 = (-2 * C.b1 * efac * (norm(rOH1) - C.reoh) * sum0 + efac * sum1 / C.reoh) / norm(rOH1)

            dVcdr2::Float64 = (-2 * C.b1 * efac * (norm(rOH2) - C.reoh) * sum0 + efac * sum2 / C.reoh) / norm(rOH2)

            dVcdcth::Float64 = efac * sum3;

            # H1
            @views grads[:, 2] = dVa1 * rOH1 + dVb * rHH + dVcdr1 * rOH1 + dVcdcth * (rOH2 / (norm(rOH1) * norm(rOH2)) - cosθ * rOH1 / (norm(rOH1) * norm(rOH1)))
            # H2
            @views grads[:, 3] = dVa2 * rOH2 - dVb * rHH + dVcdr2 * rOH2 + dVcdcth * (rOH1 / (norm(rOH1) * norm(rOH2)) - cosθ * rOH2 / (norm(rOH2) * norm(rOH2)))
            # O
            @views grads[:, 1] = -(grads[:, 2] + grads[:, 3])
            grads .*= -conversion(:wavenumbers, :hartree) # also converts from forces to gradients
            return energy
        end
    
        return energy
    end

end # end of local scope