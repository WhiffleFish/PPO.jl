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
    L_CLIP = 0.0f0
    L_VF = 0.0f0
    L_ENT = 0.0f0

    for (h,a,p,adv,v) ∈ data
        ChainRulesCore.ignore_derivatives() do
            Flux.reset!(net)
        end
        a_dist, v̂ = process_last(net,h)
        r_t = a_dist[a] / p
        L_CLIP += min(r_t*adv, clamp(r_t, 1-ϵ, 1+ϵ)*adv)
        L_VF += abs2(v - only(v̂))
        L_ENT += entropy(a_dist)
    end
    return L_CLIP, L_VF/length(data), L_ENT
    # return -(L_CLIP - c_value*sqrt(L_VF) + c_entropy*L_ENT)
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

function train!(sol, c_value, c_entropy)
    net = sol.actor_critic
    whiten!(sol.mem.advantages)
    p = Flux.params(net)
    opt = sol.opt
    for i ∈ 1:sol.n_epochs
        data = sample_data(sol.mem, sol.batch_size)
        ∇ = Flux.gradient(p) do
            L_CLIP, L_VF, L_ENT = surrogate_loss(net, data, sol.ϵ, c_value, c_entropy)
            ChainRulesCore.ignore_derivatives() do
                # println("\nL_clp: ", -L_CLIP)
                # println("L_val: ", c_value*sqrt(L_VF))
                # println("L_ent: ", -c_entropy*L_ENT)
                # @show L_CLIP
                # @show c_value*sqrt(L_VF)
                # @show c_entropy*L_ENT
            end
            -(L_CLIP - c_value*L_VF + c_entropy*L_ENT)
        end
        Flux.Optimise.update!(opt, p, ∇)
    end
end
