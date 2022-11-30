mdp = SimpleGridWorld()
A = actions(mdp)

PPO.s_vec(m::SimpleGridWorld, s::GWPos) = convert(SVector{2,Float32}, s) ./ m.size

actor = Chain(Dense(2,32, relu), Dense(32,4), softmax)
critic = Chain(Dense(2,32, relu), Dense(32,1))
ac = PPO.SplitActorCritic(actor, critic)
sol = PPOSolver(
    ac;
    n_iters=1000,
    batch_size=32,
    n_epochs=100,
    n_actors=20,
    optimizer=Adam(1f-3),
    c_entropy=1f-1)
solve(sol, mdp)


mem = sol.mem

net = ac
θ = Flux.params(net)
(;c_value, c_entropy) = sol

# nan params

for i ∈ eachindex(mem.s)
    S = mem.s[i:i]
    A = mem.a[i:i]
    ADV = mem.adv[i:i]
    V = mem.v[i:i]
    P = mem.p[i:i]

    S = reduce(hcat, S)
    ∇ = Flux.gradient(θ) do
        R_CLIP, L_VF, R_ENT = PPO.surrogate_loss(net, S, A, ADV, V, P, sol.ϵ)
        # @ignore_derivatives push!(l_hist, (-R_CLIP, L_VF, -R_ENT))
        -(R_CLIP - c_value*L_VF + c_entropy*R_ENT)
    end
    ∇̂ = norm(∇)
    isnan(∇̂) && error("NaN gradients ($i)")
end

i_nan = 18
S = reduce(hcat,mem.s[i_nan:i_nan])
A = mem.a[i_nan:i_nan]
ADV = mem.adv[i_nan:i_nan]
V = mem.v[i_nan:i_nan]
P = mem.p[i_nan:i_nan]

R_CLIP, L_VF, R_ENT = PPO.surrogate_loss(net, S, A, ADV, V, P, sol.ϵ)
-(R_CLIP - c_value*L_VF + c_entropy*R_ENT)

# S = reduce(hcat, S)
∇ = Flux.gradient(θ) do
    R_CLIP, L_VF, R_ENT = PPO.surrogate_loss(net, S, A, ADV, V, P, sol.ϵ)
    # @ignore_derivatives push!(l_hist, (-R_CLIP, L_VF, -R_ENT))
    -(R_CLIP - c_value*L_VF + c_entropy*R_ENT)
end
∇̂ = norm(∇)
for v ∈ values(∇.grads)
    v isa Array && replace!(v) do x
        isnan(x) ? 0.0 : x
    end
end



θ = Flux.params(ac)
∇ = Flux.gradient(θ) do
    sum(actor(SA[1.0f0, 1.0f0]))
end

using LinearAlgebra



moving_average(vs,n) = [sum(@view vs[i:(i+n-1)])/n for i in 1:(length(vs)-(n-1))]

using Plots
plot(getindex.(sol.logger.total_loss[8],1))

plot(moving_average(getindex.(sol.logger.total_loss[1],2), 100))
