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

mutable struct TigerUpdater
    p::Float32
    n::Int
    TigerUpdater() = new(0.5f0, 0)
end

Flux.@functor TigerUpdater

function Flux.reset!(a::TigerUpdater)
    a.p = 0.5f0
    a.n = 1
end

function (a::TigerUpdater)(x)
    if isone(x[2])
        isone(x[1]) && (a.p += 1.0f0)
        a.n += 1
    else
        a.p = 0.5f0
        a.n = 1
    end
    [a.p ./ a.n]
end


ac = PPO.RecurMultiHead(
    TigerUpdater(),#Chain(LSTM(2=>16), LSTM(16=>16)),
    Dense(1,64, tanh),
    Chain(Dense(64=>64, tanh), Dense(64,3), softmax),
    Chain(Dense(64=>64, tanh), Dense(64, 1))
)

sol = PPOSolver(
    ac,
    optimizer = Flux.Optimiser(ClipNorm(0.5),Adam(Float32(1e-2))),
    n_actors = 20,
    n_iters = 200,
    n_epochs = 50,
    batch_size = 64,
    c_entropy = 0.01f0,
    c_value = 1.0f0,
    ϵ = 0.1
)
solve(sol, pomdp)

m1 = hcat(
    (getindex.(sol.logger.total_loss[i], 1)
    for i ∈ eachindex(sol.logger.total_loss))...
); m2 = hcat(
    (getindex.(sol.logger.total_loss[i], 2)
    for i ∈ eachindex(sol.logger.total_loss))...
); m3 = hcat(
    (getindex.(sol.logger.total_loss[i], 3)
    for i ∈ eachindex(sol.logger.total_loss))...
)

plot(getindex.(sol.logger.total_loss[5], 2))

m2_max = maximum(m2, dims=2)
m2_min = minimum(m2, dims=2)
m2_normed = copy(m2)
for col ∈ 1:size(m2, 2)
    _min, _max = extrema(@view(m2[:,col]))
    @. m2_normed[:,col] = (m2[:,col] .- _min) / (_max - _min)
end


plot(m2_normed, label="", color=:blue, linealpha=0.25, lw=2)
plot(norm_m2[5,:], label="")
plot(m2[:,3], label="")
plot(m2, label="", yscale=:log10)

plot(m1, label="", color=:blue, linealpha=0.25, lw=4)
plot!(mean(m1, dims=2), color=:red, lw=4, label="")

vs = getindex.(sol.logger.total_loss[i], j)
plot(vs)


sol(h0)
sol(SA[0.0f0, 1.0f0])

##
using Plots

Flux.reset!(ac)
vv1 = [ac.recur(h0)]
for i ∈ 1:10
    push!(vv1, ac.recur(SA[0.0f0, 1.0f0]))
end

Flux.reset!(ac)
vv2 = [ac.recur(h0)]
for i ∈ 1:10
    push!(vv2, ac.recur(SA[1.0f0, 1.0f0]))
end

begin
    _j = 7
    plot(getindex.(vv1, _j), lw=2, label="All tiger left")
    plot!(getindex.(vv2, _j), lw=2, ls=:dash, label="All tiger right")
end


Flux.reset!(ac)
vv3 = [ac.recur(h0)]
push!(vv3, ac.recur(SA[1.0f0, 1.0f0]))
push!(vv3, ac.recur(SA[1.0f0, 1.0f0]))
push!(vv3, ac.recur(SA[1.0f0, 1.0f0]))
push!(vv3, ac.recur(SA[0.0f0, 2.0f0]))

begin
    _j = 1
    plot(getindex.(vv3, _j), lw=2)
    # plot!(getindex.(vv2, _j), lw=2, ls=:dash)
end

i = 1
j = 4
plot(getindex.(vv1, i), getindex.(vv1, j), lw=2, label="")
plot!(getindex.(vv2, i), getindex.(vv2, j), lw=2, label="")





p = Flux.params(sol.actor_critic.recur[2])
p |> propertynames

pp = collect(p.params)
histogram(vec(pp[5]))


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
using ProgressMeter


sol.mem

opt = Adam(Float32(1e-5))
net = deepcopy(ac)
l_hist1 = test_training(net, sol.mem, Adam(Float32(1e-3)), 64, 1_000, 1.0f0, 0.0f0)
l_hist2 = test_training(net, sol.mem, Adam(Float32(1e-5)), 64, 1_000, 1.0f0, 0.0f0)


function test_training(net, mem, opt, batch_size, n_batches, c_value, c_entropy)
    l_hist = NTuple{3, Float32}[]
    p = Flux.params(net)
    @showprogress for i ∈ 1:n_batches
        data = PPO.sample_data(mem, batch_size)
        ∇ = Flux.gradient(p) do
            R_CLIP, L_VF, R_ENT = PPO.surrogate_loss(net, data, 0.2, c_value, c_entropy)
            -(R_CLIP - c_value*L_VF + c_entropy*R_ENT)
        end
        Flux.Optimise.update!(opt, p, ∇)
        push!(l_hist, PPO.full_loss(net, sol.mem, sol.ϵ))
    end
    return l_hist
end


net2 = PPO.RecurMultiHead(
    Chain(RNN(2=>8),RNN(8=>8)),
    Dense(8,32, tanh),
    Chain(Dense(32=>16), Dense(16,3), softmax),
    Chain(Dense(32=>16, relu), Dense(16, 1))
)

l_hist2 = test_training(net2, sol.mem, Adam(Float32(1e-3)), 64, 1_000, 1.0f0, 0.0f0)
plot(getindex.(l_hist1, 2))
plot!(getindex.(l_hist2, 2))

sol.mem[10]


no_listen_idxs = findall(sol.mem.actions .≠ 1)
idx = no_listen_idxs[1]
sol.mem.rewards[idx]
sol.mem[idx]

idx = no_listen_idxs[2]
sol.mem.rewards[idx]
sol.mem[idx]

##
data = PPO.sample_data(sol.mem, sol.batch_size)

@code_warntype PPO.surrogate_loss(ac, data, sol.ϵ, sol.c_value, sol.c_entropy)

@code_warntype PPO.process_last(ac,[SA[1.0f0, 1.0f0] for _ ∈ 1:10])

sol.mem
ob = PPO.ordered_batch(sol.mem, 100)
plot(length.(getfield.(ob, :r)))

div.(rand(1:400, 300), 20) |> histogram
