using ChainRulesCore
using ProgressMeter
using Plots
pomdp = TigerPOMDP()
PPO.vec_oa(::TigerPOMDP, o, a) = SA[Float32(o),Float32(a)]
PPO.init_hist(::TigerPOMDP) = @SVector zeros(Float32, 2)
h0 = PPO.init_hist(pomdp)

net = PPO.RecurMultiHead(
    LSTM(2=>16),
    Dense(16=>32, tanh),
    Chain(Dense(32=>32, tanh), Dense(32=>3), softmax),
    Chain(Dense(32=>32, tanh), Dense(32=>1), only)
)

oa_hist, r_hist, a_hist, v_hist, p_hist = PPO.rollout(net, pomdp, 20)
adv, v = PPO.generalized_advantage_estimate(r_hist, v_hist, 0.95f0, 0.95f0)

data = [(oa_hist[1:i], v[i]) for i ∈ eachindex(oa_hist, v)]

opt = Adam(1e-3)
loss_hist = train_pls(net, data, 10_000, 32, opt)

plot(loss_hist)


function loss(net, data)
    l = 0.0f0
    for (h, v) ∈ data
        ChainRulesCore.ignore_derivatives() do
            Flux.reset!(net)
        end
        a_dist, v̂ = PPO.process_last(net,h)
        l += abs2(v̂ - v)
    end
    return l
end

function train_pls(net, data, epochs, batch_size, opt)
    l_hist = Float32[]
    p = Flux.params(net)
    @showprogress for i ∈ 1:epochs
        ∇ = Flux.gradient(p) do
            l = loss(net, data)
            ChainRulesCore.ignore_derivatives() do
                push!(l_hist, l)
            end
            l
        end
        Flux.Optimise.update!(opt, p, ∇)
    end
    return l_hist
end



## is the Flux.reset! fucking things up?
function accumulate_loss_grads(net, p, data)
    ∇ = Flux.gradient(p) do
        a_dist, v̂ = PPO.process_last(net,data[1][1])
        abs2(v̂ - data[1][2])
    end
    for (h,v) ∈ data[2:end]
        Flux.reset!(net)
        ∇ .+= Flux.gradient(p) do
            a_dist, v̂ = PPO.process_last(net,h)
            abs2(v̂ - v)
        end
    end
    return ∇
end

function full_loss(net, data)
    l = 0.0f0
    for (h,v) ∈ data
        Flux.reset!(net)
        a_dist, v̂ = PPO.process_last(net,h)
        l += abs2(v̂ - v)
    end
    l
end

function train_pls2(net, data, epochs, batch_size, opt)
    l_hist = Float32[]
    p = Flux.params(net)
    @showprogress for i ∈ 1:epochs
        ∇ = accumulate_loss_grads(net, p, data)
        push!(l_hist, full_loss(net, data))
        Flux.Optimise.update!(opt, p, ∇)
    end
    return l_hist
end

net = PPO.RecurMultiHead(
    Chain(LSTM(2=>32),LSTM(32=>32),LSTM(32=>32)),
    Dense(32=>32, tanh),
    Chain(Dense(32=>32, tanh), Dense(32=>3), softmax),
    Chain(Dense(32=>32, tanh), Dense(32=>1), only)
)

data2 = [(oa_hist[1:i], rand(Float32)) for i ∈ eachindex(oa_hist, v)]

l_hist2 = train_pls2(net, data2, 1_000, 32, Adam(1e-4))
plot(l_hist2)
l_hist3 = train_pls2(net, data, 10_000, 32, Adam(1e-3))
plot(l_hist3)
l_hist4 = train_pls2(net, data, 10_000, 32, Adam(1e-4))
plot(l_hist4)

plot(vcat(l_hist2, l_hist3, l_hist4))
