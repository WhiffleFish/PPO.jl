struct PPOSolver{AC, MEM, OPT, CV, CE}
    actor_critic::AC
    mem::MEM
    opt::OPT
    λ_GAE::Float32
    n_iters::Int
    n_actors::Int
    n_epochs::Int
    max_steps::Int
    batch_size::Int
    c_value::CV
    c_entropy::CE
    ϵ::Float32
end

function PPOSolver(
    actor_critic;
    optimizer = Adam(Float32(1e-3)),
    λ_GAE::Real = 0.95f0,
    n_iters::Integer = 10,
    n_actors::Integer = 10,
    n_epochs::Integer = 10,
    max_steps::Integer = 20,
    batch_size::Integer = 16,
    c_value = 0.1f0,
    c_entropy = 0.1f0,
    ϵ = 0.20f0
    )

    return PPOSolver(
        actor_critic,
        HistoryMemory(),
        optimizer,
        maybe_convert_f32(λ_GAE),
        n_iters,
        n_actors,
        n_epochs,
        max_steps,
        batch_size,
        maybe_convert_f32(c_value),
        maybe_convert_f32(c_entropy),
        maybe_convert_f32(ϵ)
    )
end

(ppo::PPOSolver)(x) = ppo.actor_critic(x)

Flux.@functor PPOSolver

function POMDPs.solve(sol::PPOSolver, pomdp::POMDP)
    T = sol.n_iters
    @showprogress for t ∈ 1:T
        empty!(sol.mem)
        gen_data!(sol, pomdp)
        v̂ = cumulative_rewards(sol.mem)
        println(" ",v̂)
        train!(sol, process_coeff(sol.c_value, t, T), process_coeff(sol.c_entropy, t, T))
    end
end
