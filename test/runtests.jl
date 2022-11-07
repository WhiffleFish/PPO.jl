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
h0 = PPO.init_hist(pomdp)

ac = PPO.RecurMultiHead(
    Chain(LSTM(2=>16), LSTM(16=>16)),
    Dense(16,16, tanh),
    Chain(Dense(16=>3), softmax),
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

sol(h0)
sol(SA[0.0f0, 1.0f0])

##


Flux.reset!(sol.actor_critic.recur)
vv1 = [ac.recur(h0)]
for i ∈ 1:10
    push!(vv1, ac.recur(SA[0.0f0, 1.0f0]))
end

Flux.reset!(sol.actor_critic.recur)
vv2 = [ac.recur(h0)]
for i ∈ 1:10
    push!(vv2, ac.recur(SA[1.0f0, 1.0f0]))
end

using Plots
i = 1
j = 2
plot(getindex.(vv1, i), getindex.(vv1, j), lw=2, label="")
plot!(getindex.(vv2, i), getindex.(vv2, j), lw=2, label="")

p = Flux.params(sol.actor_critic.recur)
p |> propertynames

pp = collect(p.params)
histogram(vec(pp[7]))


i = 1
j = 2
plot(getindex.(vv1, i), getindex.(vv1, j), lw=2)
plot!(getindex.(vv2, i), getindex.(vv2, j), lw=2, ls=:dash)

PPO.sample_data(sol.mem, 10)[1]


PPO.cumulative_rewards(sol.mem)

ac.recur[1].cell
ac.recur[2].cell.Wh

Flux.reset!(sol)
sol(h0)

@profiler solve(sol, pomdp)



##
using SARSOP
sarsop_policy = solve(SARSOPSolver(), pomdp)
