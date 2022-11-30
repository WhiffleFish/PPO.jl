function train!(sol, n_batches, c_value, c_entropy)
    net = sol.actor_critic
    mem = sol.mem

    θ = Flux.params(net)
    opt = sol.optimizer
    l_hist = LossHist()
    for i ∈ 1:sol.n_epochs
        shuffle!(mem)
        s_data = Iterators.partition(mem.s, sol.batch_size)
        a_data = Iterators.partition(mem.a, sol.batch_size)
        adv_data = Iterators.partition(mem.adv, sol.batch_size)
        v_data = Iterators.partition(mem.v, sol.batch_size)
        p_data = Iterators.partition(mem.p, sol.batch_size)

        for (S,A,ADV,V,P) ∈ zip(s_data, a_data, adv_data, v_data, p_data)
            S = reduce(hcat, S)
            ∇ = Flux.gradient(θ) do
                R_CLIP, L_VF, R_ENT = surrogate_loss(net, S, A, ADV, V, P, sol.ϵ)
                # @ignore_derivatives push!(l_hist, -R_CLIP, L_VF, -R_ENT)
                -(R_CLIP - c_value*L_VF + c_entropy*R_ENT)
            end
            ∇̂ = norm(∇)
            if isnan(∇̂)
                @warn("NaN gradients")
                replace_nan_grads!(∇)
            end
            Flux.Optimise.update!(opt, θ, ∇)
        end
        rc, lvf, rent = surrogate_loss(net, reduce(hcat,mem.s), mem.a, mem.adv, mem.v, mem.p, sol.ϵ)
        push!(l_hist, -rc, lvf, -rent)
    end
    push!(sol.logger.loss, l_hist)
end

function split_train!(sol, n_batches, c_value, c_entropy)
    (;actor,critic) = sol.actor_critic
    mem = sol.mem

    θa = Flux.params(actor)
    θc = Flux.params(critic)
    actor_opt = deepcopy(sol.optimizer)
    critic_opt = deepcopy(sol.optimizer)
    l_hist = LossHist()
    for i ∈ 1:sol.n_epochs
        shuffle!(mem)
        s_data = Iterators.partition(mem.s, sol.batch_size)
        a_data = Iterators.partition(mem.a, sol.batch_size)
        adv_data = Iterators.partition(mem.adv, sol.batch_size)
        v_data = Iterators.partition(mem.v, sol.batch_size)
        p_data = Iterators.partition(mem.p, sol.batch_size)

        for (S,A,ADV,V,P) ∈ zip(s_data, a_data, adv_data, v_data, p_data)
            S = reduce(hcat, S)
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
        _S = reduce(hcat,mem.s)
        rc,rent = actor_loss(actor, _S, mem.a, mem.adv, mem.p, sol.ϵ)
        lvf = critic_loss(critic, _S, mem.v)
        push!(l_hist, -rc, lvf, -rent)
    end
    push!(sol.logger.loss, l_hist)
end

function actor_loss(actor, S, A, ADV, P, ϵ)
    Π = actor(S)
    L = length(P)
    r_t = [Π[A[i],i] / P[i] for i ∈ eachindex(P,A)]
    surr1 = r_t .* ADV
    surr2 = clamp.(r_t, 1f0 - ϵ, 1f0 + ϵ) .* ADV
    R_CLIP = sum(min.(surr1, surr2))
    R_ENT = entropy(Π)
    return R_CLIP/L, R_ENT/L
end

critic_loss(critic, S, V) = sum(abs2, critic(S) .- V)

function surrogate_loss(net, S, A, ADV, V, P, ϵ)
    Π, V̂ = net(S)
    # V̂ = vec(V̂)
    L = length(V)
    r_t = [Π[A[i],i] / P[i] for i ∈ eachindex(P,A)]
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

entropy(arr::AbstractArray) = -sum(xlogx, arr)

function replace_nan_grads!(∇)
    for v ∈ values(∇.grads)
        v isa Array && replace!(v) do x
            isnan(x) ? zero(eltype(v)) : x
        end
    end
    ∇
end

function train_full!(sol, n_epochs, c_value, c_entropy)
    net = sol.actor_critic
    sol.normalize_advantage && whiten!(sol.mem.advantages)
    θ = Flux.params(net)
    opt = deepcopy(sol.optimizer)
    l_hist = LossHist()
    for i ∈ 1:n_epochs
        ∇ = Flux.gradient(p) do
            L_CLIP, L_VF, L_ENT = full_loss(net, sol.mem, sol.ϵ)
            @ignore_derivatives push!(l_hist, L_CLIP, L_VF, L_ENT)
            l = L_CLIP + c_value*L_VF + c_entropy*L_ENT
        end
        Flux.Optimise.update!(opt, θ, ∇)
    end
    push!(sol.logger.loss, l_hist)
end
