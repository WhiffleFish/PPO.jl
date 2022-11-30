const Vec32 = Vector{Float32}

struct FOBuffer
    s::Vector{Vec32}
    a::Vector{Int}
    r::Vec32
    adv::Vec32
    v::Vec32
    p::Vec32
    FOBuffer() = new(Vec32[], Int[], Float32[], Float32[], Float32[], Float32[])
end

function Base.empty!(mem::FOBuffer)
    empty!(mem.s)
    empty!(mem.a)
    empty!(mem.r)
    empty!(mem.adv)
    empty!(mem.v)
    empty!(mem.p)
end

function Base.append!(mem::FOBuffer, S, A::Vector{Int}, R, V, P, ADV)
    append!(mem.s, S)
    append!(mem.a, A)
    append!(mem.r, R)
    append!(mem.v, V)
    append!(mem.p, P)
    append!(mem.adv, ADV)
end

function Base.length(mem::FOBuffer)
    (;s,a,r,v,adv,p) = mem
    l = length(s)
    @assert l == length(a) == length(r) == length(adv) == length(p) == length(v)
    return l
end

function Random.shuffle!(mem::FOBuffer)
    (;s,a,r,adv,v,p) = mem
    idxs = shuffle(1:length(mem))
    @views s .= s[idxs]
    @views a .= a[idxs]
    @views r .= r[idxs]
    @views adv .= adv[idxs]
    @views v .= v[idxs]
    @views p .= p[idxs]
end

cumulative_rewards(mem::FOBuffer, n) = sum(mem.r) / n

function generalized_advantage_estimate(rewards, values, γ, λ)
    gae = 0.0f0
    T = length(rewards)
    vp = last(values)
    Â = zeros(Float32, T)
    for i ∈ T:-1:1
        δ = rewards[i] + γ * values[i+1] - values[i]
        gae = δ + γ * λ * gae
        Â[i] = gae
        vp = rewards[i] + γ*vp
    end
    return Â, Â .+ values[1:end-1]
end
