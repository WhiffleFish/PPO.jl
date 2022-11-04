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
        c_entropy
    )
end

(ppo::PPOSolver)(x) = ppo.actor_critic(x)

Flux.@functor PPOSolver

function POMDPs.solve(sol::PPOSolver, pomdp::POMDP)
    γ = discount(pomdp)
    advantages = Vector{Float32}[]
    for i ∈ 1:n_iters
        for act ∈ 1:n_actors
            oa_hist, r_hist, v_hist = rollout(sol, pomdp)
            Â = generalized_advantage_estimate(r_hist, v_hist, γ, sol.λ_GAE)
            push!(advantages,Â)
        end
    end
end
