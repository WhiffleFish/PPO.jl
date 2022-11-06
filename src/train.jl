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
        Flux.reset!(net)
        a_dist, v̂ = process_last(net,h)
        r_t = a_dist[a] / p
        l += min(r_t*adv, clamp(r_t, 1-ϵ, 1+ϵ)*adv)
        l -= c_value*abs2(v - only(v̂))
        l += c_entropy*entropy(a_dist)
    end
    return -l
end

function entropy(v::AbstractVector)
    s = zero(eltype(v))
    @inbounds @simd for i ∈ eachindex(v)
        v_i = v[i]
        s -= v_i*log(v_i)
    end
    return s
end

function train!(sol)

end
