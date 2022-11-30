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

PPO.s_vec(m::SimpleGridWorld, s) = s# ./ m.size

actor = Chain(Dense(2,64, relu), Dense(64,4), softmax)
critic = Chain(Dense(2,64, relu), Dense(64,64, relu), Dense(64,1))
ac = PPO.SplitActorCritic(actor, critic)
sol = PPOSolver(ac; n_actors=30, n_iters=100, n_epochs=500, batch_size=128, optimizer=Adam(3f-4))

solve(sol, mdp)

using Plots
plot(sol.logger.loss[1].value)
plot(sol.logger.loss[1].entropy)
plot(sol.logger.rewards)

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
    heatmap(v;c=:magma, xticks=1:10, yticks=1:10)
end

begin
    v = zeros(mdp.size...)
    n = zeros(mdp.size...)
    for s ∈ states(mdp)
        vs = sol.mem.v[map(==(s),sol.mem.s)]
        if all(s .> GWPos(0,0))
            v̂ = sum(vs) / length(vs)
            v[s...] = v̂
        end
    end
    heatmap(v;c=:PiYG, xticks=1:10, yticks=1:10, clims=(-10,10))
end

heatmap(r; alpha=0.5, c=:PiYG, xticks=1:10, yticks=1:10, clims=(-10,10))

## is the network expressive enough to capture this value function???

net = Chain(Dense(2,64, relu), Dense(64,64, relu), Dense(64,1))
opt = Adam(1f-3)
X = states(mdp)[1:end-1]
Y = zeros(Float32, length(X))
for (i,s) ∈ enumerate(states(mdp)[1:end-1])
    vs = sol.mem.v[map(==(s),sol.mem.s)]
    if all(s .> GWPos(0,0))
        v̂ = sum(vs) / length(vs)
        Y[i] = v̂
    end
end

PPO.surrogate_loss(ac,)


_X = reduce(hcat, X)
net(_X) - Y'
sum(net(_X) .- Y)
