function gen_data(sol, pomdp)
    γ = Float32(discount(pomdp))
    advantages = Vector{Float32}[]
    for act ∈ 1:sol.n_actors
        oa_hist, r_hist, v_hist = rollout(sol, pomdp)
        Â = generalized_advantage_estimate(r_hist, v_hist, γ, sol.λ_GAE)
        push!(advantages,Â)
    end
    return advantages
end

function surrogate_loss(sol, adv)

end

function train!(sol::PPOSolver, pomdp::POMDP)
    advantages = gen_data(sol, pomdp)
end
