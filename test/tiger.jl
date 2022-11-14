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
    Chain(LSTM(2=>8), LSTM(8,8)),
    Dense(8=>16, tanh),
    Chain(Dense(16=>16, tanh), Dense(16=>3), softmax),
    Chain(Dense(16=>16, tanh), Dense(16=>1), only)
)

sol = PPOSolver(
    ac,
    optimizer = Flux.Optimiser(Adam(0.1f0), ExpDecay(1.0, 0.1)),
    n_actors = 20,
    n_iters = 20,
    n_epochs = 100,
    max_steps = 100,
    batch_size = 128,
    c_entropy = 0.3f0,
    c_value = 0.5f0,
    normalize_advantage=true,
    ϵ = 0.2
)

solve(sol, pomdp)

using Plots
plot(getindex.(sol.logger.total_loss[4],1))

getindex.(sol.logger.total_loss[2],1)

Flux.reset!(ac)
ac(SA[0.0f0, 0.0f0])
ac(SA[1.0f0, 1.0f0])

ac.recur


sol.mem
sol.mem.oa
sol.mem.advantages
sol.mem.values

function full_test_loss(net, data)
    nograd_reset!(net)
    l = 0.0f0
    for (oa,v) ∈ data
        v̂ = net(oa)
        l += (v̂ - v)^2
    end
    return l
end

function train_value!(nn, data, n_batches, opt)
    p = Flux.params(nn)
    l_hist = Float32[]
    @showprogress for i ∈ 1:n_batches
        ∇ = Flux.gradient(p) do
            l = full_test_loss(nn, data)
            ChainRulesCore.ignore_derivatives() do
                push!(l_hist, l)
            end
            l
        end
        Flux.Optimise.update!(opt, p, ∇)
    end
    return l_hist
end

v_net = Chain(LSTM(2,2), Dense(2,1), only)
data = zip(sol.mem.oa, sol.mem.values)
l_hist = train_value!(v_net, data, 1_000, Flux.Optimiser(Adam(1.0), ExpDecay(1.0, 0.5)))
plot(l_hist)

full_test_loss(v_net, data)

function plot_v_diff(net, data)
    Flux.reset!(net)
    v̂_hist = Float32[]
    v_hist = Float32[]
    for (oa, v) ∈ data
        v̂ = net(oa)
        push!(v̂_hist, v̂)
        push!(v_hist, v)
    end
    plot(v̂_hist)
    plot!(v_hist)
end

plot_v_diff(v_net, data)

histogram(sol.mem.advantages[sol.mem.actions .== 1])
