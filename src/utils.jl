whiten!(v, μ=mean(v), σ=std(v)) = @. v = (v - μ) / σ
whiten(v, μ=mean(v), σ=std(v)) =  @. (v - μ) / σ


"""
    vec_ao(pomdp, a, o)

Turns action and observation into a single vector
"""
function vec_oa end
