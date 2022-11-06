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

@profiler solve(sol, pomdp)

##
net = sol.actor_critic
opt = sol.opt
data = PPO.sample_data(sol.mem, sol.batch_size)
PPO.surrogate_loss(net, data, sol.ϵ, sol.c_value, sol.c_entropy)

net([SA[0.0f0, 0.10f0]] for i ∈ 1:10])
PPO.process_last(net, [SA[0.0f0, 0.10f0] for i ∈ 1:10])

@code_warntype net(SA[0.0f0, 0.10f0])
@code_warntype PPO.process_last(net, [SA[0.0f0, 0.10f0] for i ∈ 1:10])

using ChainRulesCore
p = Flux.params(net)
∇ = Flux.gradient(p) do
    ChainRulesCore.ignore_derivatives() do
        Flux.reset!(net)
    end
    a_dist, v = net(SA[0.0f0, 0.1f0])
    only(v) - 1.
end




## type stable tuple mapping?!?
t = (1,2,3)
ac.heads
@code_warntype map(x->2x, t)
