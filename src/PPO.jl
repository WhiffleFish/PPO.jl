module PPO

using POMDPs
using Flux
using Base.Iterators
using LinearAlgebra
using Statistics
using ProgressMeter
using ChainRulesCore
using Random
using RecipesBase
using Distributions

export PPOSolver

include("buffer.jl")

include("utils.jl")
export GaussSmooth, AvgSmooth

include("multihead.jl")
include("rollout.jl")
include("train.jl")
include("logging.jl")
include("solver.jl")

include(joinpath("Recur", "recur.jl"))

end # module PPO
