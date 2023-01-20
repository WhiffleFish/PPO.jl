struct SplitRecurrentActorCritic{A,C}
    actor::A
    critic::C
end

Flux.@functor SplitRecurrentActorCritic

(ac::SplitRecurrentActorCritic)(x) = ac.actor(x), ac.critic(x)
