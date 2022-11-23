function rollout(policy, pomdp::POMDP, max_steps::Int=policy.max_steps, s=rand(initialstate(pomdp)))
    Flux.reset!(policy)
    A = actions(pomdp)
    γ = discount(pomdp)
    step = 0
    v̂ = 0.0f0
    oa = init_hist(pomdp)
    oa_hist = typeof(oa)[]
    a_hist = Int[]
    r_hist = Float32[]
    v_hist = Float32[]
    p_hist = Float32[]
    while step < max_steps && !isterminal(pomdp, s)
        a_dist, v = policy(oa)
        v̂ = only(v)
        a = weighted_sample(a_dist)
        s, o, r = @gen(:sp,:o,:r)(pomdp, s, A[a])
        push!(oa_hist, oa)
        push!(r_hist, r)
        push!(a_hist, a)
        push!(v_hist, v̂)
        push!(p_hist, a_dist[a])
        oa = vec_oa(pomdp, o, a)
        step += 1
    end
    vp = isterminal(mdp,s) ? 0.0f0 : v̂
    push!(v_hist, vp)
    return oa_hist, a_hist, r_hist, v_hist, p_hist
end

function init_hist end
