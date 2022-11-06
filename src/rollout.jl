function rollout(policy, pomdp::POMDP, max_steps::Int=policy.max_steps, s=rand(initialstate(pomdp)))
    Flux.reset!(policy)
    γ = discount(pomdp)
    step = 0
    oa = init_hist(pomdp)
    oa_hist = typeof(oa)[]
    r_hist = Float32[]
    a_hist = Int[]
    v_hist = Float32[]
    p_hist = Float32[]
    while step < max_steps && !isterminal(pomdp, s)
        a_dist, v = policy(oa)
        a = weighted_sample(a_dist)
        s, o, r = @gen(:sp,:o,:r)(pomdp, s, a)
        push!(oa_hist, oa)
        push!(r_hist, r)
        push!(a_hist, a)
        push!(v_hist, only(v))
        push!(p_hist, a_dist[a])
        oa = vec_oa(pomdp, o, a)
        step += 1
    end
    return oa_hist, r_hist, a_hist, v_hist, p_hist
end

function init_hist end
