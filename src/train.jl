function train!(sol, n_batches, c_value, c_entropy)
    (;n,T) = sol
    net = sol.actor_critic
    mem = sol.mem

    idxs = collect(1:n*T)


    θ = Flux.params(net)
    opt = sol.optimizer
    l_hist = LossHist()
    for i ∈ 1:sol.n_epochs
        shuffle!(idxs)
        mb_idxs = Iterators.partition(idxs, sol.batch_size)

        # could save on alloc with @view, but probably want everything contiguous in memory
        s_data = (mem.S[:, _idxs] for _idxs ∈ mb_idxs)
        a_data = (mem.A[_idxs] for _idxs ∈ mb_idxs)
        adv_data = (mem.ADV[_idxs] for _idxs ∈ mb_idxs)
        sol.normalize_advantage && foreach(whiten!, adv_data)
        v_data = (mem.V[_idxs] for _idxs ∈ mb_idxs)
        p_data = (mem.P[_idxs] for _idxs ∈ mb_idxs)

        for (S,A,ADV,V,P) ∈ zip(s_data, a_data, adv_data, v_data, p_data)
            ∇ = Flux.gradient(θ) do
                R_CLIP, L_VF, R_ENT = surrogate_loss(net, S, A, ADV, V, P, sol.ϵ)
                -(R_CLIP - c_value*L_VF + c_entropy*R_ENT)
            end
            ∇̂ = norm(∇)
            if isnan(∇̂)
                @warn("NaN gradients")
                replace_nan_grads!(∇)
            end
            Flux.Optimise.update!(opt, θ, ∇)
        end
        rc, lvf, rent = surrogate_loss(net, mem.S, mem.A, mem.ADV, mem.V, mem.P, sol.ϵ)
        push!(l_hist, -rc, lvf, -rent)
    end
    push!(sol.logger.loss, l_hist)
end

function split_train!(sol, n_batches, c_value, c_entropy)
    (;n,T) = sol
    (;actor,critic) = sol.actor_critic
    mem = sol.mem

    idxs = collect(1:n*T)

    θa = Flux.params(actor)
    θc = Flux.params(critic)
    actor_opt = sol.optimizer
    critic_opt = sol.optimizer
    l_hist = LossHist()
    for i ∈ 1:sol.n_epochs
        shuffle!(idxs)
        mb_idxs = Iterators.partition(idxs, sol.batch_size)

        # could save on alloc with @view, but probably want everything contiguous in memory
        s_data = (mem.S[:, _idxs] for _idxs ∈ mb_idxs)
        a_data = (mem.A[_idxs] for _idxs ∈ mb_idxs)
        adv_data = (mem.ADV[_idxs] for _idxs ∈ mb_idxs)
        sol.normalize_advantage && foreach(whiten!, adv_data)
        v_data = (mem.V[_idxs] for _idxs ∈ mb_idxs)
        p_data = (mem.P[_idxs] for _idxs ∈ mb_idxs)

        for (S,A,ADV,V,P) ∈ zip(s_data, a_data, adv_data, v_data, p_data)
            ∇a = Flux.gradient(θa) do
                R_CLIP, R_ENT = actor_loss(actor, S, A, ADV, P, sol.ϵ)
                -(R_CLIP + c_entropy*R_ENT)
            end
            ∇c = Flux.gradient(θc) do
                critic_loss(critic, S, V)
            end
            ∇̂a = norm(∇a)
            ∇̂c = norm(∇c)
            if isnan(∇̂a)
                @warn("actor NaN gradients")
                replace_nan_grads!(∇a)
            end
            if isnan(∇̂c)
                @warn("critic NaN gradients")
                replace_nan_grads!(∇c)
            end
            Flux.Optimise.update!(actor_opt, θa, ∇a)
            Flux.Optimise.update!(critic_opt, θc, ∇c)
        end
        rc,rent = actor_loss(actor, mem.S, mem.A, mem.ADV, mem.P, sol.ϵ)
        lvf = critic_loss(critic, mem.S, mem.V)
        push!(l_hist, -rc, lvf, -rent)
    end
    push!(sol.logger.loss, l_hist)
end

function actor_loss(actor, S, A, ADV, P, ϵ)
    Π = actor(S)
    L = length(P)
    r_t = map(eachcol(Π), P, A) do col, vi, i
        col[i] / vi
    end
    surr1 = r_t .* ADV
    surr2 = clamp.(r_t, 1f0 - ϵ, 1f0 + ϵ) .* ADV
    R_CLIP = sum(min.(surr1, surr2))
    R_ENT = entropy(Π)
    return R_CLIP/L, R_ENT/L
end

critic_loss(critic, S, V) = sum(abs2, vec(critic(S)) .- V)

function surrogate_loss(net, S, A, ADV, V, P, ϵ)
    Π, V̂ = net(S)
    V̂ = vec(V̂)
    L = length(V)
    r_t = map(eachcol(Π), P, A) do col, vi, i
        col[i] / vi
    end
    surr1 = r_t .* ADV
    surr2 = clamp.(r_t, 1f0 - ϵ, 1f0 + ϵ) .* ADV
    R_CLIP = sum(min.(surr1, surr2))
    L_VF = sum(abs2, V̂ .- V)
    R_ENT = entropy(Π)
    return R_CLIP/L, L_VF/L, R_ENT/L
end

@inline function xlogx(x::Number)
    result = x * log(x)
    return iszero(x) ? zero(result) : result
end

# much faster gradient than -sum(xlogx, arr)
entropy(arr::AbstractArray) = -sum(xlogx.(arr))

function replace_nan_grads!(∇)
    for v ∈ values(∇.grads)
        v isa Array && replace!(v) do x
            isnan(x) ? zero(eltype(v)) : x
        end
    end
    ∇
end
