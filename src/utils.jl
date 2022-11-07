whiten!(v, μ=mean(v), σ=std(v)) = @. v = (v - μ) / σ
whiten(v, μ=mean(v), σ=std(v)) =  @. (v - μ) / σ


"""
    vec_ao(pomdp, a, o)

Turns action and observation into a single vector
"""
function vec_oa end

maybe_convert_f32(x) = x
maybe_convert_f32(x::Number) = Float32(x)

struct LinearAnneal
    c0::Float32
    cf::Float32
end

process_coeff(x::Number, t, T) = x

function process_coeff(l::LinearAnneal,t,T)
    m = (l.cf - l.c0) / T
    return l.c0 + m*t
end
