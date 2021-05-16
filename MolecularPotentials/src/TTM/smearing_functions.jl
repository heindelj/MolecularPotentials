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

# smear for ttm4