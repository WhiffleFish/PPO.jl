function gen_data!(sol, m)
    γ = Float32(discount(m))
    for act ∈ 1:sol.n_actors
        s, a, r, v̂, p = rollout(sol, m)
        adv,v = generalized_advantage_estimate(r, v̂, γ, sol.λ_GAE)
        append!(sol.mem, s, a, r, v, p, adv)
    end
    return sol.mem
end

function surrogate_loss(net, data, ϵ, c_value, c_entropy)
    R_CLIP = 0.0f0
    L_VF = 0.0f0
    R_ENT = 0.0f0

    for (h,a,p,adv,v) ∈ data
        nograd_reset!(net)
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

entropy(arr::AbstractArray) = -sum(xlogx, arr)

function train_pomdp!(sol, n_batches, c_value, c_entropy)
    net = sol.actor_critic
    sol.normalize_advantage && whiten!(sol.mem.advantages)
    p = Flux.params(net)
    opt = deepcopy(sol.optimizer)
    l_hist = NTuple{3, Float32}[]
    for i ∈ 1:n_batches
        data = sample_data(sol.mem, sol.batch_size)
        ∇ = Flux.gradient(p) do
            R_CLIP, L_VF, R_ENT = surrogate_loss(net, data, sol.ϵ, c_value, c_entropy)
            -(R_CLIP - c_value*L_VF + c_entropy*R_ENT)
        end
        Flux.Optimise.update!(opt, p, ∇)
        push!(l_hist, full_loss(sol.actor_critic, sol.mem, sol.ϵ))
    end
    push!(sol.logger.total_loss, l_hist)
end

function train_full!(sol, n_epochs, c_value, c_entropy)
    net = sol.actor_critic
    sol.normalize_advantage && whiten!(sol.mem.advantages)
    θ = Flux.params(net)
    opt = deepcopy(sol.optimizer)
    l_hist = NTuple{3, Float32}[]
    for i ∈ 1:n_epochs
        ∇ = Flux.gradient(p) do
            L_CLIP, L_VF, L_ENT = full_loss(net, sol.mem, sol.ϵ)
            @ignore_derivatives push!(l_hist, (L_CLIP, L_VF, L_ENT))
            l = L_CLIP + c_value*L_VF + c_entropy*L_ENT
        end
        Flux.Optimise.update!(opt, θ, ∇)
    end
    push!(sol.logger.total_loss, l_hist)
end
