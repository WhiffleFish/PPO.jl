begin
    using Pkg
    Pkg.activate(joinpath(@__DIR__, ".."))
    using Flux
    using POMDPs
    using PPO
    Pkg.activate(@__DIR__)
    using POMDPModels
    using StaticArrays
end

pomdp = BabyPOMDP()
actions(pomdp)
observations(pomdp)
γ = discount(pomdp)
PPO.vec_oa(::BabyPOMDP, o, a) = SA[Float32(o),Float32(a)]
PPO.init_hist(::BabyPOMDP) = @SVector zeros(Float32, 2)
h0 = PPO.init_hist(pomdp)

ac = PPO.RecurMultiHead(
    Chain(LSTM(2=>16), LSTM(16=>16)),
    Dense(16,16, tanh),
    Chain(Dense(16=>2), softmax),
    Dense(16=>1)
)

sol = PPOSolver(
    ac,
    optimizer = Adam(Float32(1e-4)),
    n_actors = 50,
    n_iters = 100,
    n_epochs = 100,
    batch_size = 64,
    c_entropy = 0.05f0,
    c_value = 0.01f0,
    ϵ = 0.2
)
solve(sol, pomdp)


sarsop_policy = solve(SARSOPSolver(), pomdp)
value(sarsop_policy, initialstate(pomdp))
