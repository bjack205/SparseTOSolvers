
abstract type MeritFunction end
using FiniteDiff
const GradCache{T,M} = FiniteDiff.GradientCache{Nothing,Nothing,Nothing,Vector{T},M,T,Val{true}()} where {T,M}
# (merit::MeritFunction)(nlp, Z=TO.get_primals(nlp), λ=TO.get_primals(nlp)) = 
#     merit(nlp, Z, λ)
# grad(merit::MeritFunction, nlp, Z=TO.get_primals(nlp), λ=TO.get_primals(nlp)) = 
#     grad(merit, nlp, Z, λ)

mutable struct NormPenalty{T}  <: MeritFunction
    μ::T
    p_norm::T
    grad::Vector{T}
    c::Vector{T}
    jac::SparseMatrixCSC{T,Int}
    function NormPenalty{T}(ρ, p_norm, N::Integer, M::Integer) where T
        grad = zeros(T,N)
        # cache = FiniteDiff.GradientCache(grad, zeros(T,N))
        c = zeros(T,M)
        jac = spzeros(M,N)
        new{T}(ρ, p_norm, grad, c, jac)
    end
end
NormPenalty(args...) = NormPenalty{Float64}(args...)

function (merit::NormPenalty)(nlp, Z, c=merit.c)
    J = eval_f(nlp, Z)
    eval_c!(nlp, c, Z)
    p = merit.p_norm
    return J + merit.μ * norm(c, merit.p_norm)
end

function gradient(merit::NormPenalty, nlp, Z)
    grad_f!(nlp, merit.grad, Z)
    eval_c!(nlp, merit.c, Z)
    jac_c!(nlp, merit.jac, Z)
    for i = 1:length(merit.c)
        merit.c[i] = sign(merit.c[i])
    end
    mul!(merit.grad, merit.jac', merit.c, merit.μ, 1.0)
end

function dgrad(merit::NormPenalty, nlp, Z, dZ)
    if merit.p_norm == 1
        grad_f!(nlp, merit.grad, Z)
        eval_c!(nlp, merit.c, Z)
        return merit.grad'dZ  - merit.μ * norm(merit.c, 1)
    end
end

function minimum_penalty(merit::NormPenalty, Q, q, c, dZ; ρ=0.5) 
    a = dot(dZ, Q, dZ) 
    σ = a > 0
    num = (q'dZ + σ*0.5*dot(dZ, Q, dZ))
    (q'dZ + σ*0.5*dot(dZ, Q, dZ)) / ((1-ρ) * norm(c,1))
end