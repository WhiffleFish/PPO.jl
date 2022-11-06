function gen_data!(sol, pomdp)
    γ = Float32(discount(pomdp))
    for act ∈ 1:sol.n_actors
        oa, r, a, v̂, p = rollout(sol, pomdp)
        adv,v = generalized_advantage_estimate(r, v̂, γ, sol.λ_GAE)
        append!(sol.mem, oa, r, a, v, p, adv)
    end
    return sol.mem
end

function surrogate_loss(net, data, ϵ, c_value, c_entropy)
    l = 0.0f0
    for (h,a,p,adv,v) ∈ data
        ChainRulesCore.ignore_derivatives() do
            Flux.reset!(net)
        end
        a_dist, v̂ = process_last(net,h)
        r_t = a_dist[a] / p
        l += min(r_t*adv, clamp(r_t, 1-ϵ, 1+ϵ)*adv)
        l -= c_value*abs2(v - only(v̂))
        l += c_entropy*entropy(a_dist)
    end
    return -l
end

@inline function xlogx(x::Number)
    result = x * log(x)
    return iszero(x) ? zero(result) : result
end

@inline function entropy(v::AbstractVector)
    s = zero(eltype(v))
    @inbounds @simd for i ∈ eachindex(v)
        v_i = v[i]
        s -= xlogx(v_i)
    end
    return s
end

function train!(sol)
    net = sol.actor_critic
    p = Flux.params(net)
    opt = sol.opt
    for i ∈ 1:sol.n_epochs
        data = sample_data(sol.mem, sol.batch_size)
        ∇ = Flux.gradient(p) do
            surrogate_loss(net, data, sol.ϵ, sol.c_value, sol.c_entropy)
        end
        Flux.Optimise.update!(opt, p, ∇)
    end
end
