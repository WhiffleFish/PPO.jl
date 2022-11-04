struct MultiHead{B, H<:Tuple}
    base::B
    heads::H
    MultiHead(base, args...) = new{typeof(base), typeof(args)}(base, args)
end

function (m::MultiHead)(x)
    u = m.base(x)
    return Tuple(head(u) for head ∈ m.heads)
end

function (m::MultiHead)(x, i::Int)
    u = m.base(x)
    return m.heads[i](u)
end

Flux.@functor MultiHead

function weighted_sample(rng::Random.AbstractRNG, σ::AbstractVector)
    t = rand(rng)
    i = 1
    cw = σ[1]
    while cw < t && i < length(σ)
        i += 1
        cw += σ[i]
    end
    return i
end

weighted_sample(σ::AbstractVector) = weighted_sample(Random.GLOBAL_RNG, σ)

#=
using Flux
base = Chain(Dense(3,16), Dense(16,16))
head1 = Dense(16,5)
head2 = Dense(16,1)

m = MultiHead(base, head1, head2)


x = rand(3)
@profiler for _ in 1:10_000; m(x); end
=#
