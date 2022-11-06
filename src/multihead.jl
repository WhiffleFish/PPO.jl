#=
TODO:
is there a way to run only the recurrent part of a network to get the desired hidden state
without needing to run everything following the initial recurrent part?
=#
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


struct RecurMultiHead{R, B, H<:Tuple}
    recur::R
    base::B
    heads::H
    RecurMultiHead(recur, base, args...) = new{typeof(recur), typeof(base), typeof(args)}(recur, base, args)
end

function (m::RecurMultiHead)(x)
    u1 = m.recur(x)
    u2 = m.base(u1)
    return map(f -> f(u2), m.heads)
end

function (m::RecurMultiHead)(x, i::Int)
    x = m.recur(x)
    x = m.base(x)
    return m.heads[i](x)
end

function process_full(m::RecurMultiHead, X)
    [m(x_i) for x_i ∈ X]
end

function process_last(m::RecurMultiHead, X)
    u1 = m.recur(first(X))
    for i ∈ eachindex(X)[2:end]
        u1 = m.recur(X[i])
    end
    u2 = m.base(u1)
    return map(f -> f(u2), m.heads)
end

Flux.@functor RecurMultiHead



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
