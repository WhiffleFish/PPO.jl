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

function min_val_loss(mem)
    last_hash = UInt(0)
    oa_dict = Dict{UInt, Vector{Float32}}()
    err_dict = Dict{UInt, Float32}()
    for i ∈ 1:length(mem)
        mem.first[i] == i && (last_hash = UInt(0))
        last_hash = hash(mem.oa[i], last_hash)
        vals = get!(oa_dict, last_hash) do
            Float32[]
        end
        push!(vals, mem.values[i])
        err_dict[last_hash] = sum(abs2, vals .- mean(vals))
    end
    return sum(values(err_dict)) / length(err_dict)
end
