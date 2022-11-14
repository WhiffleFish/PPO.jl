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

"""
append obs-act vector, reward,
"""
function Base.append!(mem::HistoryMemory, oa, r, a, v, p, adv)
    @assert length(oa) == length(r) == length(a) == length(v) == length(p) == length(adv) """
    Loa = $(length(oa))
    Lr = $(length(r))
    La = $(length(a))
    Lv = $(length(v))
    Lp = $(length(p))
    Ladv = $(length(adv))
    """
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

struct MicroHist
    hist::Vector{Vec32}
    r::Vec32
    a::Vector{Int}
    v::Vec32
    p::Vec32
    adv::Vec32
    idxs::Vector{Int}
    MicroHist() = new(Vec32[], Float32[], Int[], Float32[], Float32[], Float32[], Int[])
end

function Base.push!(h::MicroHist, r, a, v, p, adv, idx)
    push!(h.r, r)
    push!(h.a, a)
    push!(h.v, v)
    push!(h.p, p)
    push!(h.adv, adv)
    push!(h.idxs, idx)
end

function ordered_batch(mem, n) # get random batch of size n
    idxs = rand(1:length(mem), n) |> sort! |> unique!

    i = 1
    data = MicroHist[]
    while i ≤ length(idxs)
        last_first = i
        f = last_first
        mch = MicroHist()
        while f == last_first && i ≤ length(idxs)
            idx = idxs[i]
            r = mem.rewards[idx]
            a = mem.actions[idx]
            adv = mem.advantages[idx]
            v = mem.values[idx]
            p = mem.probs[idx]
            rel_idx = idx - last_first + 1
            push!(mch, r, a, adv, v, p, rel_idx)
            i += 1
            i ≥ length(idxs) && break
            f = mem.first[idxs[i]]
        end
        last_first = f
        append!(mch.hist, get_hist(mem, idxs[i-1]))
        push!(data, mch)
    end
    return data
end

function generalized_advantage_estimate(rewards, values, γ, λ)
    gae = 0.0f0
    T = length(rewards)
    vp = last(values)
    Â = zeros(Float32, T)
    # vals = zeros(Float32, T)
    for i ∈ T:-1:1
        δ = rewards[i] + γ * values[i+1] - values[i]
        gae = δ + γ * λ * gae
        Â[i] = gae
        vp = rewards[i] + γ*vp
        # vals[i] = vp
    end
    return Â, Â .+ values[1:end-1]
end

# function generalized_advantage_estimate(rewards, values, γ, λ)
#     gae = 0.0f0
#     v = last(values)
#     Â = zero(rewards)
#     for i ∈ length(rewards):-1:1
#         done = i == length(rewards)
#         # δ = rewards[i] + γ * values[i+1] - values[i]
#         gae = λ*γ*gae + rewards[i] - values[i]
#         !done && (gae += γ*values[i+1])
#         Â[i] = gae
#         v = rewards[i] + γ*v
#     end
#     return Â, Â .+ values
# end
