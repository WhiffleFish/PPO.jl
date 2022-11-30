mutable struct PPOSolver{AC, OPT}
    actor_critic::AC
    mem::FOBuffer
    optimizer::OPT
    λ_GAE::Float32
    n_iters::Int
    n_actors::Int
    n_epochs
    n_steps
    max_steps::Int
    batch_size::Int
    c_value
    c_entropy
    ϵ::Float32
    normalize_advantage::Bool
    logger::Logger
end

function PPOSolver(
    actor_critic;
    optimizer = Adam(3f-4),
    λ_GAE::Real = 0.95f0,
    n_iters::Integer = 10,
    n_actors::Integer = 10,
    n_epochs = 50,
    n_steps = 20,
    max_steps::Integer = 20,
    batch_size::Integer = 16,
    c_value = 0.1f0,
    c_entropy = 0.1f0,
    ϵ = 0.20f0,
    normalize_advantage::Bool = true
    )

    return PPOSolver(
        actor_critic,
        FOBuffer(),
        optimizer,
        maybe_convert_f32(λ_GAE),
        n_iters,
        n_actors,
        n_epochs,
        n_steps,
        max_steps,
        batch_size,
        maybe_convert_f32(c_value),
        maybe_convert_f32(c_entropy),
        maybe_convert_f32(ϵ),
        normalize_advantage,
        Logger()
    )
end

(ppo::PPOSolver)(x) = ppo.actor_critic(x)

Flux.@functor PPOSolver

function POMDPs.solve(sol::PPOSolver, m::MDP)
    T = sol.n_iters
    @showprogress for t ∈ 1:T
        empty!(sol.mem)
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
