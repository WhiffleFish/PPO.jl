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

PPO.s_vec(m::SimpleGridWorld, s) = s ./ m.size

actor = Chain(Dense(2,64, tanh), Dense(64,4), softmax)
critic = Chain(Dense(2,64, tanh), Dense(64,64, tanh), Dense(64,1))
ac = PPO.SplitActorCritic(actor, critic)

sol = PPOSolver(
    ac;
    n = 30,
    n_iters = 500,
    n_epochs = 50,
    batch_size = 128,
    optimizer = Flux.Optimiser(ClipNorm(0.5f0), Adam(1f-4, (0.9f0, 0.999f0), 1.0f-5)),
    ϵ = 0.2f0,
    c_entropy = 0.01f0
)

solve(sol, mdp)

# NOTE: SHOULD NOT RUN TRAINING UNTIL CONVERGENCE.
# THIS OVERFITS TO CURRENTLY COLLECTED DATASET
# NEW POLICY/VALUE FUNCTION SHOULD BE A BLEND OF PREVIOUS AND CURRENT FITTED DATA

plot(sol.logger.loss[5].clip)
plot(sol.logger.loss[4].value)
plot(sol.logger.loss[100].entropy)
plot(sol.logger.rewards)

plot(sol.logger, GaussSmooth(20), lw=2)

plot(moving_average(sol.logger.rewards,10), lw=2, label="", title="PPO GW returns")
plot!(sol.logger.rewards, alpha=0.5, label="")
# savefig("ppo_returns.svg")
begin
    v = zeros(mdp.size...)
    r = zeros(mdp.size...)
    for s ∈ states(mdp)
        v̂ = critic(PPO.s_vec(mdp,s))
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

##
(;n,T) = sol
(;actor,critic) = sol.actor_critic
sol.mem

findall(map(!=(0),sol.mem.R))


any(isnan,sol.mem.P)

sol.mem.P
actor_loss(ac.actor)
