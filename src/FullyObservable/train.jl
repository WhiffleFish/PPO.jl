function train_mdp!(sol, n_batches, c_value, c_entropy)
    net = sol.actor_critic
    mem = sol.mem

    θ = Flux.params(net)
    opt = sol.optimizer
    l_hist = NTuple{3,Float32}[]
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
                @ignore_derivatives push!(l_hist, (-R_CLIP, L_VF, -R_ENT))
                -(R_CLIP - c_value*L_VF + c_entropy*R_ENT)
            end
            Flux.Optimise.update!(opt, θ, ∇)
        end
    end
    push!(sol.logger.total_loss, l_hist)
end

function surrogate_loss(net, S, A, ADV, V, P, ϵ)
    Π, V̂ = net(S)
    L = length(V)
    r_t = [Π[A[i],i] / P[i] for i ∈ eachindex(P)]
    surr1 = r_t .* ADV
    surr2 = clamp.(r_t, 1f0 - ϵ, 1f0 + ϵ) .* ADV
    R_CLIP = sum(min.(surr1, surr2))
    L_VF = sum(abs2, V̂ .- V)
    R_ENT = entropy(Π)
    return R_CLIP/L, L_VF/L, R_ENT/L
end
