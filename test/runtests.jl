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

ac = PPO.RecurMultiHead(LSTM(2 => 16), Dense(16=>16, relu), Chain(Dense(16=>3), softmax), Dense(16=>1))
sol = PPOSolver(ac)
PPO.gen_data(sol, pomdp)

ac([0.0f0, 3.0f0])

data = PPO.sample_data(sol.mem, 10)
PPO.surrogate_loss(ac, data, sol.ϵ, sol.c_value, sol.c_entropy)
oa_hist, r_hist, a_hist, v_hist, p_hist = PPO.rollout(sol, pomdp)

adv, v = PPO.generalized_advantage_estimate(r_hist, v_hist, γ, sol.λ_GAE)
r_hist

last(v_hist)
