struct Logger
    total_loss::Vector{Vector{NTuple{3, Float32}}}
end

Logger() = Logger(Vector{NTuple{3, Float32}}[])
