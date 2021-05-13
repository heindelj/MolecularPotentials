using StaticArrays
using LinearAlgebra
include("/home/heindelj/dev/julia_development/wally/src/molecule_tools/units.jl")
include("constants.jl")

let # scope all the constants we need for this potential
    global pot_nasa, dms_nasa!

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
        fmat = zeros(3, 16)
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

    function dms_nasa!(coords::SMatrix{3, 3, Float64}, q::MVector{3, Float64}, q_derivative::Union{MArray{Tuple{3, 3, 3}, Float64, 3, 27}, Nothing}=nothing, ttm3::Bool=false)        
        rOH1::SVector{3, Float64} = coords[:, 1] - coords[:, 2]
        rOH2::SVector{3, Float64} = coords[:, 1] - coords[:, 3]
        #rHH::SVector{3, Float64}  = coords[:, 2] - coords[:, 3]

        cosθ::Float64 = rOH1 ⋅ rOH2 / (norm(rOH1) * norm(rOH2))
        

        cosθ_e::Float64 = -0.24780227221366464506
        aθ_0::Float64   =  1.82400520401572996557

        x1::Float64     = (norm(rOH1) - C.reoh) / C.reoh
        x2::Float64     = (norm(rOH2) - C.reoh) / C.reoh
        x3::Float64     = cosθ - cosθ_e

        efac::Float64 = exp(-C.b1D * ((norm(rOH1) - C.reoh)^2 + (norm(rOH2) - C.reoh)^2))
        # SPEED: might be able to speed this up by making it 16,3?
        fmat = zeros(3, 16)
        for i in 1:3
            fmat[i, 1] = 0.0
            fmat[i, 2] = 1.0
        end

        for i in 3:16
            fmat[1, i] = fmat[1, i-1] * x1
            fmat[2, i] = fmat[2, i-1] * x2
            fmat[3, i] = fmat[3, i-1] * x3
        end

        # Calculate the dipole moment

        p1::Float64 = 0.0
        p2::Float64 = 0.0
        pl1 = cosθ
        pl2 = 0.5*(3 * pl1 * pl1 - 1.0)
    
        dp1dr1::Float64 = 0.0
        dp1dr2::Float64 = 0.0
        dp1dcabc::Float64 = 0.0
        dp2dr1::Float64 = 0.0
        dp2dr2::Float64 = 0.0
        dp2dcabc::Float64 = 0.0
    
        for j in 1:83
            inI = idxD0[j+1]
            inJ = idxD1[j+1]
            inK = idxD2[j+1]
    
            p1 += coefD[j+1] * fmat[1, inI+1] * fmat[2, inJ+1] * fmat[3, inK+1]
            p2 += coefD[j+1] * fmat[1, inJ+1] * fmat[2, inI+1] * fmat[3, inK+1]
    
            if (q_derivative === nothing) # skip derivatives
                continue
            end
    
            dp1dr1   += coefD[j+1]*(inI) * fmat[1, inI]         * fmat[2, inJ + 1]     * fmat[3, inK + 1]
            dp1dr2   += coefD[j+1]*(inJ) * fmat[1, inI + 1]     * fmat[2, inJ]         * fmat[3, inK + 1]
            dp1dcabc += coefD[j+1]*(inK) * fmat[1, inI + 1]     * fmat[2, inJ + 1]     * fmat[3, inK]
            dp2dr1   += coefD[j+1]*(inJ) * fmat[1, inJ]         * fmat[2, inI + 1]     * fmat[3, inK + 1]
            dp2dr2   += coefD[j+1]*(inI) * fmat[1, inJ + 1]     * fmat[2, inI]         * fmat[3, inK + 1]
            dp2dcabc += coefD[j+1]*(inK) * fmat[1, inJ + 1]     * fmat[2, inI + 1]     * fmat[3, inK]
        end

        xx::Float64  = 5.291772109200e-01
        xx2::Float64 = xx * xx
    
        dp1dr1 /= C.reoh / xx
        dp1dr2 /= C.reoh / xx
        dp2dr1 /= C.reoh / xx
        dp2dr2 /= C.reoh / xx
    
        pc0::Float64 = C.a * (norm(rOH1)^C.b + norm(rOH2)^C.b) * (C.c0 + pl1 * C.c1 + pl2 * C.c2)
    
        dpc0dr1::Float64 = C.a * C.b * norm(rOH1)^(C.b - 1) * (C.c0 + pl1 * C.c1 + pl2 * C.c2) * xx2
        dpc0dr2::Float64 = C.a * C.b * norm(rOH2)^(C.b - 1) * (C.c0 + pl1 * C.c1 + pl2 * C.c2) * xx2
        dpc0dcabc::Float64 = C.a * (norm(rOH1)^C.b + norm(rOH2)^C.b) * (C.c1 + 0.5 * (6.0 * pl1) * C.c2) * xx
    
        defacdr1::Float64 = -2.0 * C.b1D * (norm(rOH1) - C.reoh) * efac * xx
        defacdr2::Float64 = -2.0 * C.b1D * (norm(rOH2) - C.reoh) * efac * xx

        dp1dr1 = dp1dr1*efac + p1*defacdr1 + dpc0dr1;
        dp1dr2 = dp1dr2*efac + p1*defacdr2 + dpc0dr2;
        dp1dcabc = dp1dcabc*efac + dpc0dcabc;
        dp2dr1 = dp2dr1*efac + p2*defacdr1 + dpc0dr1;
        dp2dr2 = dp2dr2*efac + p2*defacdr2 + dpc0dr2;
        dp2dcabc = dp2dcabc*efac + dpc0dcabc;
    
        p1 = coefD[1] + p1 * efac + pc0 * xx # q^H1 in TTM2-F
        p2 = coefD[1] + p2 * efac + pc0 * xx # q^H2 paper
    
        q[1] = -(p1 + p2)  # Oxygen
        q[2] = p1          # Hydrogen-1
        q[3] = p2          # Hydrogen-2
    
        dp1dr1 /= xx
        dp1dr2 /= xx
        dp2dr1 /= xx
        dp2dr2 /= xx
    
        if (ttm3)

            #---------------------------------------------------------------------
            #........ Modification of the gas-phase dipole moment surface.........
            #---------------------------------------------------------------------
    
            AxB = @MVector zeros(3)
    
            AxB[1] = rOH1[2] * rOH2[3] - rOH1[3] * rOH2[2]
            AxB[2] =-rOH1[1] * rOH2[3] + rOH1[3] * rOH2[1]
            AxB[3] = rOH1[1] * rOH2[2] - rOH1[2] * rOH2[1]
    
            sum0::Float64 = sum(AxB.^2)
        
            sinθ::Float64 = sqrt(sum0) / (norm(rOH1) * norm(rOH2))
            angle::Float64   = atan(sinθ, cosθ)
    
            p1 = C.dms_param1 * (norm(rOH1) - C.dms_param2) + C.dms_param3 * (angle - aθ_0)
            p2 = C.dms_param1 * (norm(rOH2) - C.dms_param2) + C.dms_param3 * (angle - aθ_0)
    
            q[1] = q[1] - (p1 + p2)
            q[2] = q[2] + p1
            q[3] = q[3] + p2
    
            dp1dr1 += C.dms_param1
            dp2dr2 += C.dms_param1
            dp1dcabc -= C.dms_param3 / sinθ
            dp2dcabc -= C.dms_param3 / sinθ
    
        end # ttm3

        if (q_derivative === nothing)
            return
        end
    
        f1q1r13::Float64 = (dp1dr1 - (dp1dcabc * cosθ / norm(rOH1))) / norm(rOH1)
        f1q1r23::Float64 = dp1dcabc/(norm(rOH1) * norm(rOH2))
        f2q1r23::Float64 = (dp1dr2 - (dp1dcabc * cosθ / norm(rOH2))) / norm(rOH2)
        f2q1r13::Float64 = dp1dcabc/(norm(rOH1) * norm(rOH2))
        f1q2r13::Float64 = (dp2dr1 - (dp2dcabc * cosθ / norm(rOH1))) / norm(rOH1)
        f1q2r23::Float64 = dp2dcabc/(norm(rOH1) * norm(rOH2))
        f2q2r23::Float64 = (dp2dr2 - (dp2dcabc * cosθ / norm(rOH2))) / norm(rOH2)
        f2q2r13::Float64 = dp2dcabc / (norm(rOH1) * norm(rOH2))
    
        # first index is atom w.r.t. to which the derivative is taken
        # second index is the charge being differentiated
        
        # gradient of charge h1(second index) wrt displacement of h1(third index)
        q_derivative[1, 1, 1] = f1q1r13*rOH1[1] + f1q1r23*rOH2[1]
        q_derivative[2, 1, 1] = f1q1r13*rOH1[2] + f1q1r23*rOH2[2]
        q_derivative[3, 1, 1] = f1q1r13*rOH1[3] + f1q1r23*rOH2[3]
    
        # gradient of charge h1 wrt displacement of h2
        q_derivative[1, 1, 2] = f2q1r13*rOH1[1] + f2q1r23*rOH2[1]
        q_derivative[2, 1, 2] = f2q1r13*rOH1[2] + f2q1r23*rOH2[2]
        q_derivative[3, 1, 2] = f2q1r13*rOH1[3] + f2q1r23*rOH2[3]
    
        # gradient of charge h1 wrt displacement of O
        q_derivative[1, 1, 3] = -(q_derivative[1, 1, 1] + q_derivative[1, 1, 2])
        q_derivative[2, 1, 3] = -(q_derivative[2, 1, 1] + q_derivative[2, 1, 2])
        q_derivative[3, 1, 3] = -(q_derivative[3, 1, 1] + q_derivative[3, 1, 2])

        # gradient of charge h2 wrt displacement of h1
        q_derivative[1, 2, 1] = f1q2r13*rOH1[1] + f1q2r23*rOH2[1]
        q_derivative[2, 2, 1] = f1q2r13*rOH1[2] + f1q2r23*rOH2[2]
        q_derivative[3, 2, 1] = f1q2r13*rOH1[3] + f1q2r23*rOH2[3]

        # gradient of charge h2 wrt displacement of h2
        q_derivative[1, 2, 2] = f2q2r13*rOH1[1] + f2q2r23*rOH2[1]
        q_derivative[2, 2, 2] = f2q2r13*rOH1[2] + f2q2r23*rOH2[2]
        q_derivative[3, 2, 2] = f2q2r13*rOH1[3] + f2q2r23*rOH2[3]

        # gradient of charge h2 wrt displacement of O
        q_derivative[1, 2, 3] = -(q_derivative[1, 2, 1] + q_derivative[1, 2, 2])
        q_derivative[2, 2, 3] = -(q_derivative[2, 2, 1] + q_derivative[2, 2, 2])
        q_derivative[3, 2, 3] = -(q_derivative[3, 2, 1] + q_derivative[3, 2, 2])

        # gradient of charge O wrt displacement of h1
        q_derivative[1, 3, 1] = -(q_derivative[1, 1, 1] + q_derivative[1, 2, 1])
        q_derivative[2, 3, 1] = -(q_derivative[2, 1, 1] + q_derivative[2, 2, 1])
        q_derivative[3, 3, 1] = -(q_derivative[3, 1, 1] + q_derivative[3, 2, 1])

        # gradient of charge O wrt displacement of h2
        q_derivative[1, 3, 2] = -(q_derivative[1, 1, 2] + q_derivative[1, 2, 2])
        q_derivative[2, 3, 2] = -(q_derivative[2, 1, 2] + q_derivative[2, 2, 2])
        q_derivative[3, 3, 2] = -(q_derivative[3, 1, 2] + q_derivative[3, 2, 2])

        # gradient of charge O wrt displacement of O
        q_derivative[1, 3, 3] = -(q_derivative[1, 1, 3] + q_derivative[1, 2, 3])
        q_derivative[2, 3, 3] = -(q_derivative[2, 1, 3] + q_derivative[2, 2, 3])
        q_derivative[3, 3, 3] = -(q_derivative[3, 1, 3] + q_derivative[3, 2, 3])
        return
    end
end # end of local scope