whiten!(v, μ=mean(v), σ=std(v)) = @. v = (v - μ) / σ
whiten(v, μ=mean(v), σ=std(v)) =  @. (v - μ) / σ

"""
    vec_ao(pomdp, a, o)

Turns action and observation into a single vector
"""
function vec_oa end

maybe_convert_f32(x) = x
maybe_convert_f32(x::Number) = Float32(x)

struct LinearAnneal
    c0::Float32
    cf::Float32
end

process_coeff(x::Number, t, T) = x

lerp(a,b,α) = (1-α)*a + α*b

process_coeff(l::LinearAnneal,t,T) = lerp(l.c0, l.cf, t/T)

function full_loss(net, mem, ϵ)
    L_CLIP = 0.0f0
    L_VF = 0.0f0
    L_ENT = 0.0f0

    for i ∈ 1:length(mem)
        mem.first[i] == i && nograd_reset!(net)

        oa = mem.oa[i]
        p = mem.probs[i]
        a = mem.actions[i]
        adv = mem.advantages[i]
        v = mem.values[i]

        a_dist, v̂ = net(oa)
        r_t = a_dist[a]/p

        L_CLIP += min(r_t*adv, clamp(r_t, 1-ϵ, 1+ϵ)*adv)
        L_VF += abs2(v - only(v̂))
        L_ENT += entropy(a_dist)
    end
    return -L_CLIP/length(mem), L_VF/length(mem), -L_ENT/length(mem)
end

# https://github.com/ancorso/Crux.jl/blob/369ca517819015b24068ee91d2019d6868eef5af/src/utils.jl#L49
function LinearAlgebra.norm(grads::Flux.Zygote.Grads; p::Real = 2)
    v = []
    for θ in grads.params
        !isnothing(grads[θ]) && push!(v, norm(grads[θ] |> cpu, p))
    end
    norm(v, p)
end
