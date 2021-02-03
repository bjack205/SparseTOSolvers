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

prob = Cartpole() 
solver = ALTROSolver(prob)
solve!(solver)

## Solve Cartpole
prob = Cartpole() 
TO.add_dynamics_constraints!(prob)
nlp = TO.TrajOptNLP(prob, remove_bounds=false) 
ϕ = NormPenalty(10, 2, TO.num_vars(nlp))
Z0,λ0 = copy(TO.get_primals(nlp)), copy(TO.get_duals(nlp))
ϕ(nlp, Z0, λ0)
Z,λ = step(nlp, ϕ, ls_cond=:res)
ϕ(nlp, Z, λ)
TO.get_primals(nlp) == Z

##
# Form the KKT system
function KKT_solve(nlp)
    # Evaluate at the current point
    Z = nlp.Z.Z
    g = TO.grad_f!(nlp, Z)
    H = TO.hess_L!(nlp, Z)
    d = TO.eval_c!(nlp, Z) 
    D = TO.jac_c!(nlp, Z)
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
    Z0 = TO.get_primals(nlp)
    λ0 = TO.get_duals(nlp)
    Z = copy(Z0)
    Zbar = copy(Z)
    λ = copy(λ0)
    λbar = copy(λ)

    # Calculate step
    if alg == :newton
        dZ, dλ = KKT_solve(nlp)
    end
    J0 = TO.eval_f(nlp, Z)
    res0 = residuals(nlp, Z, λ, recalc=true, p_norm=p_norm)
    res = copy(res0)
    println("Initial cost: ", J0)
    println("Initial residuals: $(res0[1]), $(res0[2])")

    # Get step size 
    println("Linesearch")
    α = 1.0
    dJ0 = Inf
    for i = 1:ls_iters
        @. Zbar = Z + α*dZ
        @. λbar = α * dλ
        res = residuals(nlp, Zbar, λbar, p_norm=p_norm)

        if ls_cond == :res
            norm(res, p_norm) < norm(res0, p_norm) && break
            println("  α = $α, res = $(res[1]), $(res[2])")
        else
            if i == 1
                J0  = ϕ(nlp, Z, λ)
                dJ0 = grad(ϕ, nlp, Z, λ)'dZ
                println("Initial merit vals: $J0, $dJ0")
            end
            J = ϕ(nlp, Zbar, λbar)
            if ls_cond == :goldstein
                gold = J0 + (1-c1)*α*dJ0 <= J <= J0 + c1*α*dJ0
                gold && break
            else
                dJ = grad(ϕ, nlp, Zbar, λbar)'dZ
                armijo = J <= J0 + c1*α*dJ
                curv = dJ >= c2*dJ0
                curv2 = abs(dJ) <= c2 * abs(dJ0)
                println("  α = $α, ϕ = $J, ϕ′=$dJ, ($armijo,$curv,$curv2)")
                (ls_cond == :armijo && armijo) && break
                (ls_cond == :wolfe && armijo && curv) && break
                (ls_cond == :strongwolfe && armijo && curv2) && break
            end
        end
        α *= 0.5  # backtrack
    end
    Z0 .= Zbar
    λ0 .= λbar
    return Zbar, λbar
end
