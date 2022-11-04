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
PPO.vec_oa(::TigerPOMDP, o, a) = SA[Float32(o),Float32(a)]
PPO.init_hist(::TigerPOMDP) = @SVector zeros(Float32, 2)

h0 = PPO.init_hist(pomdp)
ac = PPO.MultiHead(Chain(Dense(2,16, relu),Dense(16,16, relu)), Chain(Dense(16, 3), softmax), Dense(16,1))

policy = PPOSolver(ac)
h0 = PPO.init_hist(pomdp)

policy.actor_critic(h0)

oa_hist, r_hist, v_hist = PPO.rollout(policy, pomdp)
PPO.generalized_advantage_estimate(r_hist, v_hist, discount(pomdp), policy.λ_GAE)

gen_data()
