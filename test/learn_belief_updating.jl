using Flux
using POMDPModels
using POMDPTools
using POMDPs

tiger = TigerPOMDP()
bu = DiscreteUpdater(tiger)
b0 = initialize_belief(bu, initialstate(tiger))

## THIS WORKS
N = 3
b = b0
b_hist = [Float32.(b0.b)]
for i ∈ 1:N
    b = update(bu, b, 0, true)
    push!(b_hist, Float32.(b.b))
end
b_hist

X = [PPO.init_hist(pomdp), (SA[1.0f0, 1.0f0] for _ ∈ 1:N)...]
Y = b_hist

net = Chain(LSTM(2,8), LSTM(8,2))
lhist = train!(net, X, Y, 1000, Adam(1e-3))
plot(lhist)
last(lhist)


Flux.reset!(net)
bs = [net(x) for x ∈ X]
plot(reduce(hcat,bs)')
plot!(reduce(hcat,Y)', ls=:dash)


@inline function nograd_reset!(x)
    ChainRulesCore.ignore_derivatives() do
        Flux.reset!(x)
    end
end

function loss(net, X, Y)
    l = 0.0f0
    for i ∈ eachindex(X,Y)
        l += sum(abs2,net(X[i]) .- Y[i])
    end
    return l
end

function train!(net, X, Y, epochs, opt)
    l_hist = zeros(epochs)
    p = Flux.params(net)
    for i ∈ 1:epochs
        Flux.reset!(net)
        ∇ = Flux.gradient(p) do
            l = loss(net, X, Y)
            ChainRulesCore.ignore_derivatives() do
                l_hist[i] = l
            end
            l
        end
        Flux.Optimise.update!(opt, p, ∇)
    end
    return l_hist
end

## This DOES work - fuckin idiot
function process_last(net, X)
    u = net(first(X))
    for i ∈ eachindex(X)[2:end]
        u = net(X[i])
    end
    return u
end

function loss2(net, X, Y)
    l = 0.0f0
    for i ∈ eachindex(X,Y)
        nograd_reset!(net)
        ŷ = process_last(net,X[1:i])
        l += sum(abs2, ŷ .- Y[i])
    end
    return l
end

function train2!(net, X, Y, epochs, opt)
    l_hist = zeros(epochs)
    p = Flux.params(net)
    for i ∈ 1:epochs
        ∇ = Flux.gradient(p) do
            l = loss2(net, X, Y)
            ChainRulesCore.ignore_derivatives() do
                l_hist[i] = l
            end
            l
        end
        Flux.Optimise.update!(opt, p, ∇)
    end
    return l_hist
end

net = Chain(LSTM(2,8), LSTM(8,2))
lhist = train2!(net, X, Y, 1000, Adam(1e-3))
plot(lhist)
last(lhist)

Flux.reset!(net)
bs = [net(x) for x ∈ X]
plot(reduce(hcat,bs)')
plot!(reduce(hcat,Y)', ls=:dash)
##
function TLTRDiff(net, _j)
    Flux.reset!(net)
    vv1 = [net(h0)]
    for i ∈ 1:10
        push!(vv1, net(SA[0.0f0, 1.0f0]))
    end
    Flux.reset!(net)
    vv2 = [net(h0)]
    for i ∈ 1:10
        push!(vv2, net(SA[1.0f0, 1.0f0]))
    end
    plot(getindex.(vv1, _j), lw=2, label="All tiger left")
    plot!(getindex.(vv2, _j), lw=2, ls=:dash, label="All tiger right")
end

## generalize to both tiger left and tiger right obs

b = b0
b_hist_left = [Float32.(b0.b)]
for i ∈ 1:3
    b = update(bu, b, 0, true)
    push!(b_hist_left, Float32.(b.b))
end

b = b0
b_hist_right = [Float32.(b0.b)]
for i ∈ 1:3
    b = update(bu, b, 0, false)
    push!(b_hist_right, Float32.(b.b))
end

X_l = [(SA[1.0f0, 1.0f0] for _ ∈ 1:3)...]
X_r = [(SA[0.0f0, 1.0f0] for _ ∈ 1:3)...]
X = (X_l, X_r)
Y = (b_hist_left[2:end], b_hist_right[2:end])

function loss_lr(net, X, Y)
    l = 0.0f0
    nograd_reset!(net)
    for i ∈ eachindex(X[1],Y[1])
        l += sum(abs2, net(X[1][i]) .- Y[1][i])
    end
    nograd_reset!(net)
    for i ∈ eachindex(X[2],Y[2])
        l += sum(abs2, net(X[2][i]) .- Y[2][i])
    end
    return l
end

function train_lr!(net, X, Y, epochs, opt)
    l_hist = zeros(epochs)
    p = Flux.params(net)
    @showprogress for i ∈ 1:epochs
        ∇ = Flux.gradient(p) do
            l = loss_lr(net, X, Y)
            ChainRulesCore.ignore_derivatives() do
                l_hist[i] = l
            end
            l
        end
        Flux.Optimise.update!(opt, p, ∇)
    end
    return l_hist
end

net = Chain(LSTM(2,2), softmax)
lhist = train_lr!(net, X, Y, 50_000, Adam(1e-4))
plot(lhist)

Flux.reset!(net)
ŷ = net(X[1][1])
ŷ .- Y[2][1]

loss_lr(net, X, Y)

j = 1
TLTRDiff(net,1)
plot!(getindex.(Y[1], j))
plot!(getindex.(Y[2], j))



Flux.reset!(net)
vv1 = []
for i ∈ 1:10
    push!(vv1, net(SA[0.0f0, 1.0f0]))
end

Flux.reset!(net)
vv2 = []
for i ∈ 1:10
    push!(vv2, net(SA[1.0f0, 1.0f0]))
end

plot(reduce(hcat,vv1)')
plot(reduce(hcat,vv2)')


Y = (b_hist_left, b_hist_right)
Y[1] == b_hist_left
Y[2] == b_hist_right
plot(getindex.(Y[1], 1))
plot(getindex.(b_hist_left, 1))
plot(getindex.(b_hist_right, 1))


## It works!

b = b0
b_hist_left = [Float32.(b0.b)]
for i ∈ 1:3
    b = update(bu, b, 0, true)
    push!(b_hist_left, Float32.(b.b))
end

b = b0
b_hist_right = [Float32.(b0.b)]
for i ∈ 1:3
    b = update(bu, b, 0, false)
    push!(b_hist_right, Float32.(b.b))
end

X_l = [SA[0.0f0,0.0f0], (SA[1.0f0, 1.0f0] for _ ∈ 1:3)...]
X_r = [SA[0.0f0,0.0f0], (SA[0.0f0, 1.0f0] for _ ∈ 1:3)...]
X = (X_l, X_r)
Y = (b_hist_left, b_hist_right)

function loss_lr(net, X, Y)
    l = 0.0f0
    nograd_reset!(net)
    for i ∈ eachindex(X[1],Y[1])
        l += sum(abs2, net(X[1][i]) .- Y[1][i])
    end
    nograd_reset!(net)
    for i ∈ eachindex(X[2],Y[2])
        l += sum(abs2, net(X[2][i]) .- Y[2][i])
    end
    return l
end

function train_lr!(net, X, Y, epochs, opt)
    l_hist = zeros(epochs)
    p = Flux.params(net)
    @showprogress for i ∈ 1:epochs
        ∇ = Flux.gradient(p) do
            l = loss_lr(net, X, Y)
            ChainRulesCore.ignore_derivatives() do
                l_hist[i] = l
            end
            l
        end
        Flux.Optimise.update!(opt, p, ∇)
    end
    return l_hist
end

net = LSTM(2,2)
lhist = train_lr!(net, X, Y, 50_000, Adam(1e-4))
plot(lhist)

Flux.reset!(net)
vv1 = [net(h0)]
for i ∈ 1:10
    push!(vv1, net(SA[0.0f0, 1.0f0]))
end

Flux.reset!(net)
vv2 = [net(h0)]
for i ∈ 1:10
    push!(vv2, net(SA[1.0f0, 1.0f0]))
end

plot(reduce(hcat,vv1)')
plot(reduce(hcat,vv2)')


## Now what if we use `process_last`?

b = b0
b_hist_left = [Float32.(b0.b)]
for i ∈ 1:3
    b = update(bu, b, 0, true)
    push!(b_hist_left, Float32.(b.b))
end

b = b0
b_hist_right = [Float32.(b0.b)]
for i ∈ 1:3
    b = update(bu, b, 0, false)
    push!(b_hist_right, Float32.(b.b))
end

X_l = [SA[0.0f0,0.0f0], (SA[1.0f0, 1.0f0] for _ ∈ 1:3)...]
X_r = [SA[0.0f0,0.0f0], (SA[0.0f0, 1.0f0] for _ ∈ 1:3)...]
X = (X_l, X_r)
Y = (b_hist_left, b_hist_right)

function loss_lr(net, X, Y)
    l = 0.0f0
    nograd_reset!(net)
    for i ∈ eachindex(X[1],Y[1])
        l += sum(abs2, process_last(net, X[1][1:i]) .- Y[1][i])
    end
    nograd_reset!(net)
    for i ∈ eachindex(X[2],Y[2])
        l += sum(abs2, process_last(net, X[2][1:i]) .- Y[2][i])
    end
    return l
end

function train_lr!(net, X, Y, epochs, opt)
    l_hist = zeros(epochs)
    p = Flux.params(net)
    @showprogress for i ∈ 1:epochs
        ∇ = Flux.gradient(p) do
            l = loss_lr(net, X, Y)
            ChainRulesCore.ignore_derivatives() do
                l_hist[i] = l
            end
            l
        end
        Flux.Optimise.update!(opt, p, ∇)
    end
    return l_hist
end

net = LSTM(2,2)
lhist = train_lr!(net, X, Y, 10_000, Adam(1e-3))
plot(lhist)

Flux.reset!(net)
vv1 = [net(h0)]
for i ∈ 1:10
    push!(vv1, net(SA[0.0f0, 1.0f0]))
end

Flux.reset!(net)
vv2 = [net(h0)]
for i ∈ 1:10
    push!(vv2, net(SA[1.0f0, 1.0f0]))
end

plot(reduce(hcat,vv1)')
plot(reduce(hcat,vv2)')

## Now what if we generalize to any ao sequence?

function gen_xy(b0, bu, n)
    
end
