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
