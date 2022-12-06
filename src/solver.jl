mutable struct PPOSolver{AC, OPT}
    actor_critic::AC
    mem::Buffer
    optimizer::OPT
    λ_GAE::Float32
    n_iters::Int
    n::Int
    n_epochs
    n_steps
    T::Int
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
    n::Integer = 10,
    n_epochs = 50,
    T = 20,
    max_steps::Integer = 20,
    batch_size::Integer = 16,
    c_value = 0.1f0,
    c_entropy = 0.1f0,
    ϵ = 0.20f0,
    normalize_advantage::Bool = true
    )

    return PPOSolver(
        actor_critic,
        Buffer(0,0,0),
        optimizer,
        maybe_convert_f32(λ_GAE),
        n_iters,
        n,
        n_epochs,
        T,
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

function POMDPs.solve(sol::PPOSolver, mdp::MDP)
    γ = discount(mdp)
    statedim = length(s_vec(mdp,rand(initialstate(mdp))))
    sol.mem = Buffer(sol.n, sol.T, statedim)
    T = sol.n_iters
    prog = Progress(T)
    m = VectorizedMDP(mdp, sol.n, sol.T)
    for t ∈ 1:T
        gen_data!(sol, m, γ)
        v̂ = cumulative_rewards(sol.mem, γ)
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
        next!(prog, showvalues = [(:v̂,round(v̂;sigdigits=3))])
    end
    ProgressMeter.finish!(prog)
end
