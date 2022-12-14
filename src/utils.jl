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

lerp(a,b,α) = (1-α)*a + α*b

process_coeff(l::LinearAnneal,t,T) = lerp(l.c0, l.cf, t/T)

function full_loss(net, mem, ϵ)
    L_CLIP = 0.0f0
    L_VF = 0.0f0
    L_ENT = 0.0f0

    for i ∈ 1:length(mem)
        mem.first[i] == i && nograd_reset!(net)

        oa = mem.oa[i]
        p = mem.probs[i]
        a = mem.actions[i]
        adv = mem.advantages[i]
        v = mem.values[i]

        a_dist, v̂ = net(oa)
        r_t = a_dist[a]/p

        L_CLIP += min(r_t*adv, clamp(r_t, 1-ϵ, 1+ϵ)*adv)
        L_VF += abs2(v - only(v̂))
        L_ENT += entropy(a_dist)
    end
    return -L_CLIP/length(mem), L_VF/length(mem), -L_ENT/length(mem)
end

##

abstract type AbstractSmoother end

struct GaussSmooth <: AbstractSmoother
    σ::Float64
    GaussSmooth(σ = 10.) = new(convert(Float64, σ))
end

function _gauss_smooth(v::AbstractVector{T}, σ) where T
    dist = Normal(zero(T),T(σ))
    r2 = zero(v)
    for i ∈ eachindex(v)
        val = 0.0
        ws = 0.0
        for j ∈ eachindex(v)
            w = pdf(dist, i-j)
            ws += w
            val += w*v[j]
        end
        r2[i] = val / ws
    end
    return r2
end

(smoother::GaussSmooth)(v) = _gauss_smooth(v, smoother.σ)

struct AvgSmooth <: AbstractSmoother
    n::Int
    AvgSmooth(n = 10) = new(n)
end

moving_average(v, n) = [sum(@view v[i:(i+n-1)])/n for i in 1:(length(v)-(n-1))]

(smoother::AvgSmooth)(v) = moving_average(v, smoother.n)
