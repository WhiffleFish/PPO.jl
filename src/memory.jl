const Vec32 = Vector{Float32}

struct HistoryMemory{H}
    first::Vector{Int}
    lengths::Vector{Int}
    oa::Vector{H}
    rewards::Vec32
    actions::Vector{Int}
    values::Vec32
    probs::Vec32
    advantages::Vec32
    HistoryMemory() = new{Vec32}(
        Int[],
        Int[],
        Vec32[],
        Float32[],
        Float32[],
        Float32[],
        Float32[],
        Float32[],
    )
end

function Base.empty!(mem::HistoryMemory)
    for field in propertynames(mem)
        empty!(getfield(mem, field))
    end
end

Base.length(h::HistoryMemory) = length(h.oa)

function Base.append!(mem::HistoryMemory, oa, r, a, v, p, adv)
    l_buffer = length(mem)
    l_data = length(r)
    append!(mem.first, Iterators.repeated(l_buffer+1, l_data))
    append!(mem.lengths, Iterators.repeated(l_data, l_data))
    append!(mem.oa, oa)
    append!(mem.rewards, r)
    append!(mem.actions, a)
    append!(mem.values, v)
    append!(mem.probs, p)
    append!(mem.advantages, adv)
end

Base.rand(hist::HistoryMemory, n) = rand(Random.GLOBAL_RNG, hist, n)

function Base.rand(rng, hist::HistoryMemory, n)
    return [get_hist(hist, i) for i ∈ rand(rng, 1:length(hist), n)]
end


function Base.getindex(mem::HistoryMemory, i)
    @boundscheck checkbounds(mem.rewards, i)
    return (
        get_hist(mem, i),
        mem.actions[i],
        mem.probs[i],
        mem.advantages[i],
        mem.values[i]
    )
end

"""
generate vector of (h,a,p,adv,v)
"""
sample_data(rng, mem::HistoryMemory, n) = [mem[i] for i ∈ rand(rng, 1:length(mem), n)]

sample_data(mem::HistoryMemory, n) = sample_data(Random.GLOBAL_RNG, mem, n)

get_hist(hist::HistoryMemory, i) = hist.oa[hist.first[i]:i]

function cumulative_rewards(mem::HistoryMemory)
    L = length(mem)
    n = 0
    v = 0.0f0
    first_idx = 1
    while first_idx < L
        l = mem.lengths[first_idx]
        last_idx = first_idx + l - 1
        v += sum(@view(mem.rewards[first_idx:last_idx]))
        n += 1
        first_idx = last_idx + 1
    end
    return v / n
end

# function generalized_advantage_estimate(rewards, values, γ, λ)
#     gae = 0.0f0
#     v = last(values)
#     Â = zero(rewards)
#     for i ∈ (length(rewards)-1):-1:1
#         δ = rewards[i] + γ * values[i+1] - values[i]
#         gae = δ + γ * λ * gae
#         Â[i] = gae
#         v = rewards[i] + γ*v
#     end
#     return Â, Â .+ values
# end

function generalized_advantage_estimate(rewards, values, γ, λ)
    gae = 0.0f0
    v = last(values)
    Â = zero(rewards)
    for i ∈ length(rewards):-1:1
        done = i == length(rewards)
        # δ = rewards[i] + γ * values[i+1] - values[i]
        gae = λ*γ*gae + rewards[i] - values[i]
        !done && (gae += γ*values[i+1])
        Â[i] = gae
        v = rewards[i] + γ*v
    end
    return Â, Â .+ values
end
