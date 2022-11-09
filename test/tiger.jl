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

pomdp = TigerPOMDP()
γ = discount(pomdp)
PPO.vec_oa(::TigerPOMDP, o, a) = SA[Float32(o),Float32(a)]
PPO.init_hist(::TigerPOMDP) = @SVector zeros(Float32, 2)

ac = PPO.RecurMultiHead(
    Chain(LSTM(2=>16), LSTM(16=>16)),
    Dense(16,64, tanh),
    Chain(Dense(64=>64, tanh), Dense(64,3), softmax),
    Chain(Dense(64=>64, tanh), Dense(64, 1))
)

sol = PPOSolver(
    ac,
    optimizer = Flux.Optimiser(ClipNorm(0.5),Adam(Float32(1e-4))),
    n_actors = 10,
    n_iters = 2,
    n_epochs = 10,
    max_steps = 40,
    batch_size = 32,
    c_entropy = 0.1f0,
    c_value = 1.0f0,
    ϵ = 0.1
)


solve(sol, pomdp)
@profiler solve(sol, pomdp)
