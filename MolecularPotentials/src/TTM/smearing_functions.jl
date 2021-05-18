include("gammq.jl")

function smear1_ttm3(drsq::Float64, α_12::Float64, a::Float64)
    dd::Float64    = sqrt(drsq)
    dri::Float64   = 1.0/dd
    drsqi::Float64 = dri^2

    if (α_12 > eps(Float64))
        g23::Float64      = exp(gammln(2.0/3.0))
        AA::Float64       = α_12^(1.0/6.0)
        rA::Float64       = dd/AA
        rA3::Float64      = rA^3
        exp1::Float64     = exp(-a*rA3)
        a_cubert::Float64 = a^(1.0/3.0)

        ts0::Float64 = (1.0 - exp1 + a_cubert*rA*g23*gammq(2.0/3.0, a*rA3))*dri
        ts1::Float64 = (1.0 - exp1)*dri*drsqi
        return ts0, ts1
    end

    ts0 = dri
    ts1 = dri*drsqi
    return ts0, ts1
end

function smear2_ttm3(drsq::Float64, α_12::Float64, a::Float64)
    dd::Float64    = sqrt(drsq)
    dri::Float64   = 1.0/dd
    drsqi::Float64 = dri^2

    if (α_12 > eps(Float64))
        AA::Float64   = α_12^(1.0/6.0)
        rA::Float64   = dd/AA
        rA3::Float64  = rA^3
        exp1::Float64 = exp(-a*rA3)

        ts1::Float64 = (1.0 - exp1)*dri*drsqi
        ts2::Float64 = (ts1 - exp1*a/std::pow(AA, 3))*drsqi
        return ts1, ts2
    end

    ts1 = dri*drsqi
    ts2 = ts1*drsqi
    return ts1, ts2
end

function smear3_ttm3(drsq::Float64, α_12::Float64, a::Float64)
    dd::Float64    = sqrt(drsq)
    dri::Float64   = 1.0/dd
    drsqi::Float64 = dri^2

    if (α_12 > eps(Float64))
        AA::Float64   = α_12^(1.0/6.0)
        rA::Float64   = dd/AA
        rA3::Float64  = rA^3
        exp1::Float64 = exp(-a*rA3)

        ts1::Float64 = (1.0 - exp1)*dri*drsqi
        ts2::Float64 = (ts1 - exp1*a/std::pow(AA, 3))*drsqi
        ts3::Float64 = (ts2 - 0.6*exp1*dd*a*a/pol12)*drsqi
        return ts1, ts2, ts3
    end 
    ts1 = dri * drsqi
    ts2 = ts1 * drsqi
    ts3 = ts2 * drsqi
    return ts1, ts2, ts3
end

function smear1_ttm4(drsq::Float64, α_12::Float64, a::Float64)
    dd::Float64    = sqrt(drsq)
    dri::Float64   = 1.0/dd
    drsqi::Float64 = dri^2

    if (α_12 > eps(Float64))
        g34::Float64   = exp(gammln(3.0/4.0))
        AA::Float64    = α_12^(1.0/6.0)
        rA::Float64    = dd/AA
        rA4::Float64   = rA^4
        exp1::Float64  = exp(-a*rA4)
        a_mrt::Float64 = a^(1.0/4.0)

        ts0 = (1.0 - exp1 + a_mrt*rA*g34*ttm::gammq(3.0/4.0, a*rA4))*dri
        ts1 = (1.0 - exp1)*dri*drsqi
        return ts0, ts1
    end
    ts0 = dri
    ts1 = dri*drsqi
    return ts0, ts1
end

function smear2_ttm4(drsq::Float64, α_12::Float64, a::Float64)
    dd::Float64    = sqrt(drsq)
    dri::Float64   = 1.0/dd
    drsqi::Float64 = dri^2

    if (α_12 > eps(Float64)) 
        AA::Float64   = α_12^(1.0/6.0)
        rA::Float64   = dd/AA
        rA4::Float64  = rA^4
        exp1::Float64 = exp(-a*rA4)

        ts1 = (1.0 - exp1)*dri*drsqi
        ts2 = (ts1 - (4.0/3.0)*a*exp1*rA4*dri*drsqi)*drsqi
        return ts1, ts2
    end
    ts1 = dri*drsqi
    ts2 = ts1*drsqi
    return ts1, ts2
end

function smear3_ttm4(drsq::Float64, α_12::Float64, a::Float64)
    dd::Float64    = sqrt(drsq)
    dri::Float64   = 1.0/dd
    drsqi::Float64 = dri^2

    if (α_12 > eps(Float64))
        AA::Float64   = α_12^(1.0/6.0)
        AA4::Float64  = AA^4
        rA::Float64   = dd/AA
        rA4::Float64  = rA^4
        exp1::Float64 = exp(-a*rA4)

        ts1 = (1.0 - exp1)*dri*drsqi
        ts2 = (ts1 - (4.0/3.0)*a*exp1*rA4*dri*drsqi)*drsqi
        ts3 = (ts2 - (4.0/15.0)*dri*a*(4*a*rA4 - 1.0)*exp1/AA4)*drsqi
        return ts1, ts2, ts3
    end
    ts1 = dri*drsqi
    ts2 = ts1*drsqi
    ts3 = ts2*drsqi
    return ts1, ts2, ts3
end