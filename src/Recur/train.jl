env2memidxs(env::Int, T::Int) = T*(env - 1) + 1 : T*env

function train_recur!(sol, n_batches, c_value, c_entropy)
    (;n,T) = sol
    net = sol.actor_critic
    mem = sol.mem

    env_idxs = collect(1:n)
    envs_per_mb = max(sol.batch_size ÷ T, 1)

    θ = Flux.params(net)
    opt = sol.optimizer
    for i ∈ 1:sol.n_epochs
        shuffle!(env_idxs)
        mb_env_idx_iter = Iterators.partition(env_idxs, envs_per_mb)

        for mb_env_idxs ∈ mb_env_idx_iter
            mem_idxs = (
                env2memidxs(env_idx, T) for env_idx ∈ mb_env_idxs
            )
            S = Tuple(mem.S[:,idxs] for idxs ∈ mem_idxs)
            A = Tuple(mem.A[idxs] for idxs ∈ mem_idxs)
            ADV = Tuple(mem.ADV[idxs] for idxs ∈ mem_idxs)
            V = Tuple(mem.V[idxs] for idxs ∈ mem_idxs)
            P = Tuple(mem.P[idxs] for idxs ∈ mem_idxs)
            DONE = Tuple(mem.next_done[idxs] for idxs ∈ mem_idxs)
            L = length(mb_env_idxs)*T

            ∇ = Flux.gradient(θ) do
                R_CLIP, L_VF, R_ENT = recur_surrogate_loss(net, S, A, ADV, V, P, DONE, sol.ϵ)
                -(R_CLIP - c_value*L_VF + c_entropy*R_ENT)/L
            end
            # @show ∇.grads
            # ∇̂ = norm(∇.grads)
            # if isnan(∇̂)
                # @warn("NaN gradients")
                # replace_nan_grads!(∇)
            # end
            Flux.Optimise.update!(opt, θ, ∇)
        end
    end
end

function split_train_recur!(sol, n_batches, c_value, c_entropy)
    (;n,T) = sol
    (;actor,critic) = sol.actor_critic
    mem = sol.mem

    idxs = collect(1:n*T)

    θa = Flux.params(actor)
    θc = Flux.params(critic)
    actor_opt = deepcopy(sol.optimizer)
    critic_opt = deepcopy(sol.optimizer)
    for i ∈ 1:sol.n_epochs
        shuffle!(idxs)
        mb_idxs = Iterators.partition(idxs, sol.batch_size)

        # could save on alloc with @view, but probably want everything contiguous in memory
        s_data = (mem.S[:, _idxs] for _idxs ∈ mb_idxs)
        a_data = (mem.A[_idxs] for _idxs ∈ mb_idxs)
        adv_data = (mem.ADV[_idxs] for _idxs ∈ mb_idxs)
        v_data = (mem.V[_idxs] for _idxs ∈ mb_idxs)
        p_data = (mem.P[_idxs] for _idxs ∈ mb_idxs)
        done_data = (mem.next_done[_idxs] for _idxs ∈ mb_idxs)

        for (S,A,ADV,V,P,DONE) ∈ zip(s_data, a_data, adv_data, v_data, p_data, done_data)
            ∇a = Flux.gradient(θa) do
                R_CLIP, R_ENT = recur_actor_loss(actor, S, A, ADV, P, DONE, sol.ϵ)
                -(R_CLIP + c_entropy*R_ENT)
            end
            ∇c = Flux.gradient(θc) do
                recur_critic_loss(critic, S, V)
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
    end
end

function recur_actor_loss(actor, S, A, ADV, P, DONE, ϵ)
    R_CLIP = 0f0
    R_ENT = 0f0
    for env ∈ eachindex(S,A,ADV,P,DONE)
        _S = S[env]
        _A = A[env]
        _ADV = ADV[env]
        _P = P[env]
        _DONE = DONE[env]
        nograd_reset!(actor)
        for i ∈ eachindex(_S,_A, _ADV,_P,_DONE)
            π_i = actor(S[:,i])
            r_t = π_i[_A[i]] / _P[i]
            surr1 = r_t * _ADV[i]
            surr2 = clamp(r_t, 1f0 - ϵ, 1f0 + ϵ) * _ADV[i]
            R_CLIP += min(surr1, surr2)
            R_ENT += entropy(π_i)
            _DONE[i] && nograd_reset!(actor)
        end
    end
    return R_CLIP/L, R_ENT/L
end

function recur_critic_loss(critic, S, V, DONE)
    l = 0f0
    for env ∈ eachindex(S,V,DONE)
        _S = S[env]
        _V = V[env]
        _DONE = DONE[env]
        nograd_reset!(actor)
        for i ∈ eachindex(_S,_V, _DONE)
            _DONE[i] && nograd_reset!(critic)
            l += abs2(only(critic(S[:,i])) - _V[i])
        end
    end
    return l
end

function recur_surrogate_loss(net, S, A, ADV, V, P, DONE, ϵ)
    R_CLIP = 0f0
    R_ENT = 0f0
    L_V = 0f0
    for env ∈ eachindex(S,A,ADV,V,P,DONE)
        _S = S[env]
        _A = A[env]
        _ADV = ADV[env]
        _V = V[env]
        _P = P[env]
        _DONE = DONE[env]
        nograd_reset!(net)
        for i ∈ eachindex(_A,_ADV,_V,_P,_DONE)
            π_i,v = net(_S[:,i])
            r_t = π_i[_A[i]] / _P[i]
            surr1 = r_t * _ADV[i]
            surr2 = clamp(r_t, 1f0 - ϵ, 1f0 + ϵ) * _ADV[i]
            R_CLIP += min(surr1, surr2)
            R_ENT += entropy(π_i)
            L_V += abs2(only(v) - _V[i])
            _DONE[i] && nograd_reset!(net)
        end
    end
    return R_CLIP, L_V, R_ENT
end
