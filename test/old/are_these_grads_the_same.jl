lstm = LSTM(3,3)

target = fill(1.0f0,3)
target_pos = 5
X = [rand(Float32,3) for _ ∈ 1:10]
Flux.reset!(lstm)
Y = [lstm(x) for x ∈ X]
Y[target_pos] = target


p = Flux.params(lstm)
Flux.reset!(lstm)
∇1 = Flux.gradient(p) do
    l = 0.0f0
    for i ∈ 1:5
        ŷ = lstm(X[i])
        l += sum(abs2, ŷ .- Y[i])
    end
    ChainRulesCore.@ignore_derivatives @show l
    l
end

Flux.reset!(lstm)
∇2 = Flux.gradient(p) do
    nograd_reset!(lstm)
    ŷ = process_last(lstm, X[1:5])
    l = sum(abs2, ŷ .- target)
    ChainRulesCore.@ignore_derivatives @show l
    l
end

i = 5
∇1.grads[p[i]]
∇2.grads[p[i]]

##
X = [rand(Float32,3) for _ ∈ 1:10]
Y = [rand(Float32,3)*i for i ∈ 1:10]

p = Flux.params(lstm)
Flux.reset!(lstm)
∇1 = Flux.gradient(p) do
    l = 0.0f0
    for i ∈ 1:5
        ŷ = lstm(X[i])
        l += sum(abs2, ŷ .- Y[i])
    end
    ChainRulesCore.@ignore_derivatives @show l
    l
end

Flux.reset!(lstm)
∇2 = Flux.gradient(p) do
    l = 0.0f0
    for i ∈ 1:5
        nograd_reset!(lstm)
        ŷ = process_last(lstm, X[1:i])
        l += sum(abs2, ŷ .- Y[i])
    end
    ChainRulesCore.@ignore_derivatives @show l
    l
end

i = 4
∇1.grads[p[i]] .- ∇2.grads[p[i]]

∇1.grads[p[i]] == ∇2.grads[p[i]]
