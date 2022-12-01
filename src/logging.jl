struct LossHist
    clip::Vec32
    value::Vec32
    entropy::Vec32
    LossHist() = new(Float32[],Float32[],Float32[])
end

function Base.push!(h::LossHist, c, v, e)
    push!(h.clip, c)
    push!(h.value, v)
    push!(h.entropy, e)
end

struct Logger
    rewards::Vec32
    loss::Vector{LossHist}
    Logger() = new(Float32[], LossHist[])
end

@recipe function f(log::Logger, smooth::AbstractSmoother)
    xlabel --> "Training Iteration"
    ylabel --> "Returns"
    @series begin
        label --> "smooth"
        smooth(log.rewards)
    end
    @series begin
        alpha --> 0.5
        label --> "raw"
        log.rewards
    end

end

@recipe function f(log::Logger)
    xlabel --> "Training Iteration"
    ylabel --> "Returns"
    label --> ""
    log.rewards
end
