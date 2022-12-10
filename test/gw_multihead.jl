begin
    using Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
    using PPO
    Pkg.activate(@__DIR__)
    using POMDPModels
    using POMDPs
    using Flux
    using Plots
end

mdp = SimpleGridWorld()

PPO.s_vec(m::SimpleGridWorld, s) = (s .- (m.size ./2)) ./ m.size

_init = (args...) -> Flux.orthogonal(args...;gain=√2)

# orthogonal init with gain √2
d = PPO.MultiHead(
    Chain(Dense(2,64,tanh; init=_init), Dense(64,64,tanh;init=_init)),
    Chain(Dense(64,4;init=_init), softmax),
    Dense(64, 1;init=_init)
)

sol = PPOSolver(
    d;
    n = 30,
    T = 30,
    n_iters = 500,
    n_epochs = 25,
    batch_size = 128,
    optimizer = Flux.Optimiser(
        ClipNorm(0.5f0),
        Adam(1f-4, (0.9f0, 0.999f0), 1.0f-5)
    ),
    ϵ = 0.2f0,
    c_entropy = 0.1f0
)

solve(sol, mdp)

# NOTE: SHOULD NOT RUN TRAINING UNTIL CONVERGENCE.
# THIS OVERFITS TO CURRENTLY COLLECTED DATASET
# NEW POLICY/VALUE FUNCTION SHOULD BE A BLEND OF PREVIOUS AND CURRENT FITTED DATA

using Plots
plot(sol.logger.loss[1].clip)
plot(sol.logger.loss[4].value)
plot(sol.logger.loss[10].entropy)
plot(sol.logger.rewards)

plot(sol.logger, GaussSmooth(20), lw=2)
plot(moving_average(sol.logger.rewards,10), lw=2, label="", title="PPO GW returns")
plot!(sol.logger.rewards, alpha=0.5, label="")
# savefig("ppo_returns.svg")
begin
    v = zeros(mdp.size...)
    r = zeros(mdp.size...)
    for s ∈ states(mdp)
        v̂ = last(d(PPO.s_vec(mdp,s)))
        if all(s .> GWPos(0,0))
            v[s...] = only(v̂)
            r[s...] = reward(mdp, s)
        end
    end
    heatmap(v;c=:PiYG, xticks=1:10, yticks=1:10, clims=(-10,10))
end
savefig("ppo_learned_value_function_low_lr.svg")


begin
    v = zeros(mdp.size...)
    n = zeros(mdp.size...)
    for s ∈ states(mdp)
        vs = sol.mem.V[map(≈(PPO.s_vec(mdp,s)),eachcol(sol.mem.S))]
        if all(s .> GWPos(0,0))
            v̂ = sum(vs) / length(vs)
            v[s...] = v̂
        end
    end
    heatmap(v;c=:PiYG, xticks=1:10, yticks=1:10, clims=(-10,10))
end

heatmap(r; c=:PiYG, xticks=1:10, yticks=1:10)
savefig("reward_distribution.svg")


plot(sol.logger, GaussSmooth(10), lw=2)
plot(sol.logger, AvgSmooth(10), lw=2)
plot(sol.logger)
