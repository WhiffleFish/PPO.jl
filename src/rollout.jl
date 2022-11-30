function rollout(policy, mdp::MDP, max_steps::Int=policy.max_steps, s=rand(initialstate(mdp)))
    A = actions(mdp)
    γ = discount(mdp)
    step = 0
    v = 0f0
    s_hist = Vec32[]
    a_hist = Int[]
    r_hist = Float32[]
    v_hist = Float32[]
    p_hist = Float32[]
    while step < max_steps && !isterminal(mdp, s)
        s_v = s_vec(mdp, s)
        a_dist, v̂ = policy(s_v)
        v = only(v̂)
        a = weighted_sample(a_dist)
        s, r = @gen(:sp,:r)(mdp, s, A[a])
        push!(s_hist, s_v)
        push!(a_hist, a)
        push!(r_hist, r)
        push!(v_hist, v)
        push!(p_hist, a_dist[a])
        step += 1
    end
    vp = isterminal(mdp,s) ? 0f0 : v
    push!(v_hist, vp)
    return s_hist, a_hist, r_hist, v_hist, p_hist
end

function s_vec end

function gen_data!(sol, m)
    γ = Float32(discount(m))
    for act ∈ 1:sol.n_actors
        s, a, r, v̂, p = rollout(sol, m)
        adv,v = generalized_advantage_estimate(r, v̂, γ, sol.λ_GAE)
        append!(sol.mem, s, a, r, v, p, adv)
    end
    return sol.mem
end
