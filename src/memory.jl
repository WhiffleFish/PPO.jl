struct HistoryMemory{H}
    first::Vector{Int}
    lengths::Vector{Int}
    oa::Vector{H}
    rewards::Vector{Float64}
    value::Vector{Float64}
    HistoryMemory() = new{Vector{Float64}}(Int[], Int[], Vector{Float64}[], Float64[], Float64[])
end

Base.length(h::HistoryMemory) = length(h.oa)

function Base.getindex(h::HistoryMemory{H}, i) where H
    @boundscheck checkbounds(h.rewards, i)
    l = h.lengths[i]
    idxs = (i-(l-1)):i
    return (h.oa[idxs], h.rewards[i], h.values[i])
end

function Base.append!(hist::HistoryMemory, h, r, v)
    l_buffer = length(hist)
    l_data = length(r)
    append!(hist.first, Iterators.repeated(l_buffer+1, l_data))
    append!(hist.lengths, Iterators.repeated(l_data, l_data))
    append!(hist.rewards, r)
    append!(hist.values, v)
end

Base.rand(hist::HistoryMemory, n) = rand(Random.GLOBAL_RNG, hist, n)

function Base.rand(rng, hist::HistoryMemory, n)
    return [get_hist(hist, i) for i ∈ rand(rng, 1:length(hist), n)]
end

get_hist(hist::HistoryMemory, i) = hist.oa[hist.first[i]:i]

function generalized_advantage_estimate(rewards, values, γ, λ)
    gae = 0.0f0
    v = last(values)
    Â = zero(rewards)
    v_vec = copy(values)
    for i ∈ (length(rewards)-1):-1:1
        δ = rewards[i] + γ * values[i+1] - values[i]
        gae = δ + γ * λ * gae
        Â[i] = gae
        v_vec[i] = v += rewards[i] + γ*v
    end
    return Â
end
