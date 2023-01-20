function POMDPs.solve(sol::PPOSolver, m::POMDP)
    γ = discount(m)
    statedim = length(init_hist(m))
    sol.mem = Buffer(sol.n, sol.T, statedim)
    T = sol.n_iters
    pomdp = VectorizedPOMDP(m, sol.n, sol.T)
    @showprogress for t ∈ 1:T
        gen_data!(sol, pomdp)
        v̂ = cumulative_rewards(sol.mem, sol.n)
        println(" ",round(v̂;sigdigits=3))
        push!(sol.logger.rewards, v̂)
        if sol.actor_critic isa SplitActorCritic
            split_train_recur!(
                sol,
                round(Int, process_coeff(sol.n_epochs, t, T)),
                process_coeff(sol.c_value, t, T),
                process_coeff(sol.c_entropy, t, T)
            )
        else
            train_recur!(
                sol,
                round(Int, process_coeff(sol.n_epochs, t, T)),
                process_coeff(sol.c_value, t, T),
                process_coeff(sol.c_entropy, t, T)
            )
        end
    end
end
