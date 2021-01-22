using TrajectoryOptimization
using Altro
using RobotZoo
using RobotDynamics
using LinearAlgebra
using FiniteDiff
using StaticArrays
const TO = TrajectoryOptimization
const RD = RobotDynamics
include("problems.jl")

##  Solve with ALTRO
prob = DoubleIntegrator()
solver = ALTROSolver(prob)
solve!(solver)

## Solve Direct
prob = DoubleIntegrator()
TO.add_dynamics_constraints!(prob)
nlp = TO.TrajOptNLP(prob, remove_bounds=false) 
ϕ = NormPenalty(10, 2, TO.num_vars(nlp))
step(nlp, ϕ)
cost(nlp) - cost(solver)

##
# Form the KKT system
function KKT_solve(nlp)
    # Evaluate at the current point
    Z = nlp.Z.Z
    g = TO.grad_f!(nlp)
    H = TO.hess_L!(nlp, Z)
    d = TO.eval_c!(nlp) 
    D = TO.jac_c!(nlp)
    λ = nlp.data.λ
    P = length(d)
    NN = length(Z)

    # Find the search direction 
    A = [H D'; D zeros(P,P)]
    b = -[g; d]
    dv = A\b
    dz = dv[1:NN]
    dλ = dv[NN+1:end]

    return dz, dλ
end

function residuals(nlp, Z=TO.get_primals(nlp), λ=TO.get_duals(nlp); 
        recalc::Bool=true, p_norm=Inf
    )
    if recalc || Z !== TO.get_primals(nlp)
        TO.grad_f!(nlp,Z)
        TO.eval_c!(nlp,Z) 
        TO.jac_c!(nlp,Z)
    end
    D,d = nlp.data.D, nlp.data.d
    g   = nlp.data.g

    # Get the residuals
    res_p = norm(g + D'λ, p_norm)
    res_d = norm(d, p_norm)
    return SA[res_p, res_d]
end

function step(nlp, ϕ; alg=:newton, ls_cond=:res, ls_iters=20, p_norm=Inf, c1=1e-4, c2=0.9)
    Z = TO.get_primals(nlp)
    Zbar = copy(Z)
    λ = TO.get_duals(nlp)
    λbar = copy(λ)

    # Calculate step
    if alg == :newton
        dZ, dλ = KKT_solve(nlp)
    end
    J0 = TO.eval_f(nlp)
    res0 = residuals(nlp, Z, λ, recalc=false, p_norm=p_norm)
    res = copy(res0)
    println("Initial residuals: $(res0[1]), $(res0[2])")

    # Get step size 
    println("Linesearch")
    α = 1.0
    dJ0 = Inf
    for i = 1:ls_iters
        @. Zbar = Z + α*dZ
        @. λbar = α * dλ
        res = residuals(nlp, Zbar, λbar, p_norm=p_norm)
        println("  α = $α, res = $(res[1]), $(res[2])")

        if ls_cond == :res
            norm(res, p_norm) < norm(res0, p_norm) && break
        else
            if i == 1
                J0  = ϕ(nlp)
                dJ0 = grad(merit, nlp)
            end
            J = ϕ(nlp, Zbar, λbar)
            if ls_cond == :goldstein
                gold = J0 + (1-c1)*α*dJ0 <= J <= J0 + c1*α*dJ0
                gold && break
            else
                dJ = grad(merit, nlp, Zbar, λbar)'dZ
                armijo = J <= J + c1*α*dJ
                curv = dJ0'dZ >= c2*dJ
                curv2 = abs(dJ) <= c2 * abs(dJ0)
                (ls_cond == :armijo && armijo) && break
                (ls == :wolfe && armijo && curv) && break
                (ls == :strongwolfe && armijo && curv2) && break
            end
        end
        α *= 0.5  # backtrack
    end
    Z .= Zbar
    λ .= λbar
    return res 
end

abstract type MeritFunction end
using FiniteDiff
const GradCache{T,M} = FiniteDiff.GradientCache{Nothing,Nothing,Nothing,Vector{T},M,T,Val{true}()} where {T,M}
(merit::MeritFunction)(nlp, Z=TO.get_primals(nlp), λ=TO.get_primals(nlp)) = 
    merit(nlp, Z, λ)
grad(merit::MeritFunction, nlp, Z=TO.get_primals(nlp), λ=TO.get_primals(nlp)) = 
    grad(merit, nlp, Z, λ)

mutable struct NormPenalty{T}  <: MeritFunction
    ρ::T
    p_norm::T
    grad::Vector{T}
    cache::GradCache{T,Val{:central}()}
    function NormPenalty{T}(ρ,p_norm,N::Integer) where T
        grad = zeros(T,N)
        cache = FiniteDiff.GradientCache(grad, zeros(T,N))
        new{T}(ρ, p_norm, grad, cache)
    end
end
NormPenalty(args...) = NormPenalty{Float64}(args...)

function (merit::NormPenalty)(nlp, Z, λ)
    J = TO.eval_f(nlp, Z)
    d = TO.eval_c!(nlp, Z)
    p = merit.p_norm
    return J + merit.ρ * norm(d, merit.p_norm)
end

function grad(merit::NormPenalty, nlp, Z, λ)
    ϕ(z) = merit(nlp, z, λ)
    FiniteDiff.finite_difference_gradient!(merit.grad, ϕ, Z, merit.cache)
end
