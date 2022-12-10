function POMDPs.solve(sol::PPOSolver, m::POMDP)
    T = sol.n_iters
    @showprogress for t ∈ 1:T
        gen_data!(sol, m)
        v̂ = cumulative_rewards(sol.mem, sol.n_actors)
        println(" ",round(v̂;sigdigits=3))
        push!(sol.logger.rewards, v̂)
        if sol.actor_critic isa SplitActorCritic
            split_train!(
                sol,
                round(Int, process_coeff(sol.n_epochs, t, T)),
                process_coeff(sol.c_value, t, T),
                process_coeff(sol.c_entropy, t, T)
            )
        else
            train!(
                sol,
                round(Int, process_coeff(sol.n_epochs, t, T)),
                process_coeff(sol.c_value, t, T),
                process_coeff(sol.c_entropy, t, T)
            )
        end
    end
end
