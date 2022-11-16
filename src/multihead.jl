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

@inline function nograd_reset!(x)
    ChainRulesCore.ignore_derivatives() do
        Flux.reset!(x)
    end
end

function process_last(m::RecurMultiHead, X)
    nograd_reset!(m)
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

struct SplitActorCritic{A,C}
    actor::A
    critic::C
end

Flux.@functor SplitActorCritic

(ac::SplitActorCritic)(x) = ac.actor(x), ac.critic(x)

struct JointActorCritic{AC}
    ac::AC
end
