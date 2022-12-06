const Vec32 = Vector{Float32}

struct Buffer
    S::Matrix{Float32}
    A::Vector{Int}
    R::Vector{Float32}
    V::Vector{Float32}
    _V::Vector{Float32}
    P::Vector{Float32}
    ADV::Vector{Float32}
    next_done::BitVector

    n::Int
    T::Int

    Buffer(n, T, statedim) = new(
        Matrix{Float32}(undef, statedim, n*T),
        Vector{Int}(undef, n*T),
        Vector{Float32}(undef, n*T),
        Vector{Float32}(undef, n*T),
        Vector{Float32}(undef, n*(T+1)),
        Vector{Float32}(undef, n*T),
        Vector{Float32}(undef, n*T),
        BitVector(undef, n*T),
        n, T
    )
end

# function cumulative_rewards(buff::Buffer, γ=1f0)
#     (;n,T) = buff
#     v = 0f0
#     for env ∈ 1:n
#         discount = 1f0
#         for r ∈ @view buff.R[T*(env-1)+1 : T*env]
#             v += discount*r
#             discount *= γ
#         end
#     end
#     return v / n
# end

function cumulative_rewards(buff::Buffer, γ=1f0)
    return sum(buff.R) / buff.n
end
