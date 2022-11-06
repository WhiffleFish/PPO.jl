struct PPOSolver{AC, MEM, OPT}
    actor_critic::AC
    mem::MEM
    opt::OPT
    λ_GAE::Float32
    n_iters::Int
    n_actors::Int
    n_epochs::Int
    max_steps::Int
    batch_size::Int
    c_value::Float32
    c_entropy::Float32
    ϵ::Float32
end

function PPOSolver(
    actor_critic;
    optimizer = Adam(Float32(1e-3)),
    λ_GAE = 0.5f0,
    n_iters::Int = 10,
    n_actors::Int = 10,
    n_epochs::Int = 10,
    max_steps::Int = 20,
    batch_size::Int = 16,
    c_value = 1.0f0,
    c_entropy = 0.1f0,
    ϵ = 0.20f0
    )

    return PPOSolver(
        actor_critic,
        HistoryMemory(),
        optimizer,
        λ_GAE,
        n_iters,
        n_actors,
        n_epochs,
        max_steps,
        batch_size,
        c_value,
        c_entropy,
        ϵ
    )
end

(ppo::PPOSolver)(x) = ppo.actor_critic(x)

Flux.@functor PPOSolver

function POMDPs.solve(sol::PPOSolver, pomdp::POMDP)
    @showprogress for i ∈ 1:sol.n_iters
        empty!(sol.mem)
        gen_data!(sol, pomdp)
        train!(sol)
    end
end
