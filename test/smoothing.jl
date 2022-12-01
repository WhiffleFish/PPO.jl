r = sol.logger.rewards
using Statistics, Distributions


dist = Normal(0,25)
r2 = zero(r)
for i ∈ eachindex(r)
    v = 0.0
    ws = 0.0
    for j ∈ eachindex(r)
        w = pdf(dist, i-j)
        ws += w
        v += w*r[j]
    end
    r2[i] = v / ws
end

plot(r)
plot!(r2, lw=3)
