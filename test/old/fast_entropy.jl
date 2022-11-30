##
using Zygote
using BenchmarkTools
arr = rand(100,100)


@btime PPO.entropy($arr)
@btime new_entropy($arr)
@btime new_entropy2($arr)


function new_entropy(A)
    s = zero(eltype(A))
    @inbounds @simd for i ∈ eachindex(A)
        s -= PPO.xlogx(A[i])
    end
    s
end

new_entropy2(A) = -sum(PPO.xlogx.(A))


@btime Zygote.gradient($PPO.entropy, $arr)
@btime Zygote.gradient($new_entropy, $arr)
@btime Zygote.gradient($new_entropy2, $arr)
