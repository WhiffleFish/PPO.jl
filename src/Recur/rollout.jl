function oa_vec end
function init_hist end

struct VectorizedPOMDP{M<:POMDP}
    pomdp::M
    n::Int
    T::Int
end

# TODO: actually vectorize
function rollout!(buff::Buffer, policy, m::VectorizedPOMDP)
    (;n,T,pomdp) = m
    ACT = actions(pomdp)

    for env ∈ 1:n
        s = rand(initialstate(pomdp))
        oa = init_hist(pomdp)
        for t ∈ 1:T
            idx = t + T*(env-1)

            A,v = policy(oa)
            a_idx = weighted_sample(A)
            a = ACT[a_idx]
            sp, o, r = @gen(:sp,:o,:r)(pomdp, s, a)
            term = isterminal(pomdp, sp)

            buff.S[:, idx] .= oa
            buff.A[idx] = a_idx
            buff.R[idx] = r
            buff._V[idx] = v
            buff.P[idx] = A[a_idx]
            buff.next_done[idx] = term

            term && (sp = rand(initialstate(mdp)))
            oa = oa_vec(pomdp, o, a)
            s = sp
        end
        _,v = policy(oa)
        buff._V[T*env + 1] = v
    end
    buff
end
