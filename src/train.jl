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
    R_CLIP = 0.0f0
    L_VF = 0.0f0
    R_ENT = 0.0f0

    for (h,a,p,adv,v) ∈ data
        ChainRulesCore.ignore_derivatives() do
            Flux.reset!(net)
        end
        a_dist, v̂ = process_last(net,h)
        r_t = a_dist[a]/p
        R_CLIP += min(r_t*adv, clamp(r_t, 1-ϵ, 1+ϵ)*adv)
        L_VF += abs2(v - only(v̂))
        R_ENT += entropy(a_dist)
    end
    return R_CLIP/length(data), L_VF/length(data), R_ENT/length(data)
end

@inline function xlogx(x::Number)
    result = x * log(x)
    return iszero(x) ? zero(result) : result
end

@inline function entropy(v::AbstractVector)
    s = zero(eltype(v))
    @inbounds for i ∈ eachindex(v)
        s -= xlogx(v[i])
    end
    return s
end

# @inline entropy(v::AbstractVector) = mapreduce(xlogx, -, v, init=zero(eltype(v)))

function train!(sol, n_batches, c_value, c_entropy)
    net = sol.actor_critic
    sol.normalize_advantage && whiten!(sol.mem.advantages)
    p = Flux.params(net)
    opt = sol.optimizer
    l_hist = NTuple{3, Float32}[]
    for i ∈ 1:n_batches
        data = sample_data(sol.mem, sol.batch_size)
        ∇ = Flux.gradient(p) do
            R_CLIP, L_VF, R_ENT = surrogate_loss(net, data, sol.ϵ, c_value, c_entropy)
            # ChainRulesCore.ignore_derivatives() do
            #     println("\nL_clp: ", -R_CLIP)
            #     println("L_val: ", c_value*L_VF)
            #     println("L_ent: ", -c_entropy*R_ENT)
            #     # @show L_CLIP
            #     # @show c_value*sqrt(L_VF)
            #     # @show c_entropy*L_ENT
            # end
            -(R_CLIP - c_value*L_VF + c_entropy*R_ENT)
        end
        Flux.Optimise.update!(opt, p, ∇)
        push!(l_hist, full_loss(sol.actor_critic, sol.mem, sol.ϵ))
    end
    push!(sol.logger.total_loss, l_hist)
end
