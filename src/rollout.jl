function s_vec end

struct VectorizedMDP{M<:MDP}
    mdp::M
    n::Int
    T::Int
end

POMDPs.initialstate(mdp::VectorizedMDP) = [rand(initialstate(mdp.mdp)) for i ∈ 1:mdp.n]

function gen!(m::VectorizedMDP, S::AbstractVector, S_v::AbstractMatrix, A, V, buff::Buffer, i)
    (;mdp, n, T) = m
    ACT = actions(mdp)
    for env ∈ eachindex(S)
        idx = i + T*(env-1)

        a_idx = weighted_sample(@view A[:, env])
        P = A[a_idx, env]
        sp, r = @gen(:sp, :r)(mdp, S[env], ACT[a_idx])
        term = isterminal(mdp, sp)
        sv = s_vec(mdp, S[env])

        S_v[:,env] .= sv
        buff.S[:,idx] .= sv
        buff.A[idx] = a_idx
        buff.R[idx] = r
        buff._V[idx] = V[env]
        buff.P[idx] = P
        buff.next_done[idx] = term
        term && (sp = rand(initialstate(mdp)))
        S[env] = sp
    end
end

function rollout!(buff::Buffer, policy, m::VectorizedMDP)
    (;n, T, mdp) = m
    S = initialstate(m)
    S_v = Matrix(mapreduce(s -> s_vec(m.mdp, s), hcat, S))

    for i ∈ 1:T
        A,V = policy(S_v)
        gen!(m, S, S_v, A, V, buff, i)
    end

    _, V = policy(S_v)
    buff._V[T+1 : T : n*T+1] .= vec(V)
    buff
end


function fill_gae!(buff, γ, λ)
    (;n,T) = buff
    for env ∈ 1:n
        idxs = T*(env-1)+1 : T*env
        r = @view buff.R[idxs]
        V = @view buff.V[idxs]
        v = @view buff._V[T*(env-1)+1 : T*env + 1]
        Â = @view buff.ADV[idxs]
        done = @view buff.next_done[idxs]
        @assert length(r) == length(V) == length(v)-1 == length(Â) == length(done) == T
        generalized_advantage_estimate!(Â, V, r, v, done, γ, λ)
    end
end

function generalized_advantage_estimate!(Â, V, r, v, done, λ, γ)
    gae = 0f0
    T = length(r)
    for t ∈ T:-1:1
        nextnonterminal = 1f0-done[t]
        δ = r[t] + γ * v[t+1] * nextnonterminal - v[t]
        Â[t] = gae = δ + γ * λ * gae * nextnonterminal
    end

    return Â, copyto!(V,Â .+ @view(v[1:end-1]))
end

function gen_data!(sol, v_mdp, γ=1f0)
    rollout!(sol.mem, sol.actor_critic, v_mdp)
    fill_gae!(sol.mem, γ, sol.λ_GAE)
end
