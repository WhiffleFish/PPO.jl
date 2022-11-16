mdp = SimpleGridWorld()
A = actions(mdp)
actor = Chain(Dense(2,32, relu), Dense(32,4), softmax)
critic = Chain(Dense(2,32, relu), Dense(32,1))
ac = PPO.SplitActorCritic(actor, critic)

PPO.s_vec(m::SimpleGridWorld, s::GWPos) = convert(SVector{2,Float32}, s) ./ m.size

sol = PPOSolver(ac; n_iters=1000, batch_size=64, n_epochs=100, n_actors=20, optimizer=Adam(1f-2), c_entropy=1f-4)
solve(sol, mdp)

using Plots
plot(getindex.(sol.logger.total_loss[3],1))

(;s,a,adv,v,p) = sol.mem
PPO.surrogate_loss(ac, reduce(hcat,s), a, adv, v, p, 0.95f0)

a
ac(reduce(hcat,s))

PPO.rollout(sol, mdp)


GWPos(8, 10)


sp, r = @gen(:sp,:r)(mdp, GWPos(8, 9), A[2])

isterminal(mdp, GWPos(8,8))

@edit reward(mdp, GWPos(9,9), :left)

mdp.rewards
mdp.rewards[GWPos(9, 9)]


mdp.size


sol.mem.r


sol(PPO.s_vec(mdp,SA[6,6]))

mdp
