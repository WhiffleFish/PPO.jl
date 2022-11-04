function rollout(policy::PPOSolver, pomdp::POMDP, max_steps::Int=policy.max_steps, s=rand(initialstate(pomdp)))
    Flux.reset!(policy)
    γ = discount(pomdp)
    step = 0
    oa = init_hist(pomdp)
    oa_hist = typeof(oa)[]
    r_hist = Float32[]
    v_hist = Float32[]
    while step < max_steps && !isterminal(pomdp, s)
        a_dist, v = policy(oa)
        a = weighted_sample(a_dist)
        s, o, r = @gen(:sp,:o,:r)(pomdp, s, a)
        push!(oa_hist, oa)
        push!(r_hist, r)
        push!(v_hist, only(v))
        oa = vec_oa(pomdp, o, a)
        step += 1
    end
    return oa_hist, r_hist, v_hist
end

function init_hist end
