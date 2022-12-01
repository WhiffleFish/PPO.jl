using Statistics, Distributions

function gauss_smooth(v, σ)
    dist = Normal(0,σ)
    r2 = zero(v)
    for i ∈ eachindex(r)
        val = 0.0
        ws = 0.0
        for j ∈ eachindex(r)
            w = pdf(dist, i-j)
            ws += w
            val += w*v[j]
        end
        r2[i] = val / ws
    end
    return r2
end

##
r = sol.logger.rewards
steps = collect(0:length(r)-1) * 20*20
r2 = gauss_smooth(r, 25)
plot(steps, r2, lw=3, title="PPO Training History", xlabel="Steps", ylabel="Returns", label="Smoothed")
plot!(steps, r, alpha=0.5, label="Raw")
savefig("gauss_smoothed_gw_returns.svg")


using DiscreteValueIteration
p = solve(ValueIterationSolver(), mdp)
mean([value(p, s) for s ∈ states(mdp)])

sss = [rand(initialstate(mdp)) for _ in 1:1000]

histogram(first.(sss))
histogram(last.(sss))
