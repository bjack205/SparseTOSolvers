using ForwardDiff
using FiniteDiff
using SparseArrays
using SparseDiffTools
using OSQP
using BenchmarkTools
using Test
using SuiteSparse
using Printf

const SRC = joinpath(@__DIR__, "..", "src")
include(joinpath(SRC, "nlp.jl"))
include(joinpath(SRC, "qp_solvers.jl"))
include(joinpath(SRC, "meritfuns.jl"))
include("problems.jl")

## Generate Problem
LinearAlgebra.isdiag(fact::QDLDL.QDLDLFactorisation) = nnz(fact.L) == 0
function QDLDL.solve(fact::QDLDL.QDLDLFactorisation, B::AbstractMatrix)
    if isdiag(fact)
        X = fact.Dinv * B
    else
        X = copy(B) 
        QDLDL.solve!(fact, X)
    end
    return X
end

function QDLDL.solve!(fact::QDLDL.QDLDLFactorisation, X::AbstractMatrix)
    if isdiag(fact)
        X .= fact.Dinv * X
    else
        for x in eachcol(X)
            QDLDL.solve!(fact, x)
        end
    end
    return X
end


function solve_sqp!(nlp, Z, λ;
        iters=100,
        qp_solver=:osqp,
        gauss_newton::Bool=true,
        adaptive_reg::Bool=!gauss_newton,
        verbose=0
    )

    # Initialize solution
    qp = TOQP(size(nlp)..., num_eq(nlp), 0)
    ϕ = NormPenalty(10.0, 1, num_primals(nlp), num_eq(nlp))

    if qp_solver == :osqp
        qp_solver = OSQP.Model(qp, verbose=false)
    elseif qp_solver == :shur
        qp_solver = ShurSolver(qp)
    elseif qp_solver == :kkt
        qp_solver = KKTSolver(qp)
    elseif qp_solver == :qdldl_kkt
        qp_solver = QDLDLSolver(qp, method=:kkt)
    elseif qp_solver == :qdldl_shur
        qp_solver = QDLDLSolver(qp, method=:shur)
    end

    reg = 0.0
    dZ = zero(Z)

    for iter = 1:iters
        ## Check the residuals and cost
        res_p = primal_residual(nlp, Z, λ)
        res_d = dual_residual(nlp, Z, λ)
        J = eval_f(nlp, Z)
        verbose > 0 && @printf("Iteration %d: cost = %0.2f, res_p = %0.2e, res_d = %0.2e,", iter, J, res_p, res_d)

        if res_p < 1e-6 && res_d < 1e-6
            verbose > 0 && println()
            break
        end

        # Build QP
        build_qp!(qp, nlp, Z, λ, gauss_newton) 

        # Solve the QP
        if adaptive_reg
            # reg = find_reg(qp.Q, step=5, iters=20)
            qp.Q .+= I(num_primals(nlp))*reg
        end
        dZ, dλ = solve_qp!(qp_solver, qp, 
            verbose=false, polish=true, eps_rel=1e-10, eps_abs=1e-10) 
        qp_res_p = norm(qp.Q*dZ + qp.q + qp.A'dλ)
        qp_res_d = norm(qp.A*dZ - qp.b)
        verbose > 1 && @printf(" qp_res_p = %0.2e, qp_res_d = %0.2e, δλ = %0.2e", qp_res_p, qp_res_d, norm(dλ,Inf))

        ## Line Search

        # Update penalty paramter
        μ_min = minimum_penalty(ϕ, qp.Q, qp.q, qp.b, dZ)
        dot(dZ,qp.Q,dZ)
        if ϕ.μ < μ_min
            ϕ.μ = μ_min*1.1
        end

        # Actual line search
        α = 1.0
        phi0 = ϕ(nlp,Z)
        dphi0 = dgrad(ϕ, nlp, Z, dZ)
        phi = Inf

        τ = 0.5
        η = 1e-2
        Z̄ = copy(Z)
        dphi1 = gradient(ϕ, nlp, Z)'dZ
        verbose > 2 && @printf("\n   ϕ0: %0.2f, ϕ′: %0.2e, %0.2e\n", phi0, dphi0, dphi1)
        for i = 1:10
            Z̄ .= Z .+ α .* dZ
            λbar = λ - α*dλ
            phi = ϕ(nlp, Z̄)
            res_d = dual_residual(nlp, Z̄, λbar)
            res_p = primal_residual(nlp, Z̄, λbar)
            verbose > 2 && @printf("   ls iter: %d, Δϕ: %0.2e, ϕ′: %0.2e, res_p: %0.2e, res_d: %0.2e\n", 
                i, phi-phi0, η*α*dphi0, res_p, res_d)
            if phi < phi0 + η*α*dphi0
                reg = 0
                break
            else
                α *= τ
            end
            if i == 10
                # reg += 10
                α = 0
                Z̄ .= Z
            end
        end
        Z .= Z̄
        # λ .= -λhat
        λ .= λ - α*dλ
        A = qp.A
        # λ .= (A*A')\(A*qp.q)
        verbose > 0 && @printf("   α = %0.2f, ΔJ: %0.2e, Δϕ: %0.2e, ϕ′: %0.2e, reg: %0.2f, pen: %d\n", 
            α, J - eval_f(nlp, Z), phi0 - phi, dgrad(ϕ, nlp, Z, dZ), reg, ϕ.μ)

    end 
    return Z, λ, dZ
end

##
prob = CartpoleProb()
nlp = NLP(prob)
Z = Vector(prob.Z)
λ = zeros(num_duals(nlp))

# Gauss-Newton
@time Zsqp, λsqp, dZ = solve_sqp!(nlp, copy(Z), copy(λ), iters=30, qp_solver=:shur, verbose=1)
@time Zsqp, λsqp, dZ = solve_sqp!(nlp, copy(Z), copy(λ), iters=30, qp_solver=:osqp, verbose=1)
@time Zsqp, λsqp, dZ = solve_sqp!(nlp, copy(Z), copy(λ), iters=30, qp_solver=:kkt, verbose=1)
@time Zsqp, λsqp, dZ = solve_sqp!(nlp, copy(Z), copy(λ), iters=30, qp_solver=:qdldl_kkt, verbose=1)
@time Zsqp, λsqp, dZ = solve_sqp!(nlp, copy(Z), copy(λ), iters=30, qp_solver=:qdldl_shur, verbose=1)

# Full Hessian
@time Zsqp, λsqp, dZ = solve_sqp!(nlp, copy(Z), copy(λ), iters=30, qp_solver=:kkt, verbose=1, gauss_newton=false, adaptive_reg=false)
@time Zsqp, λsqp, dZ = solve_sqp!(nlp, copy(Z), copy(λ), iters=30, qp_solver=:qdldl_kkt, verbose=1, gauss_newton=false, adaptive_reg=false)
Zsqp, λsqp, dZ = solve_sqp!(nlp, copy(Z), copy(λ), iters=30, qp_solver=:qdldl_shur, verbose=1, gauss_newton=false, adaptive_reg=false)

qp = TOQP(nlp)
solver = KKTSolver(qp)
build_qp!(qp, nlp, Z, λ)
solve_qp!(solver, qp)
Q = qp.Q
A = qp.A
K = [Q A'; A zeros(M,M)]

M = num_duals(nlp)
norm.(eachcol(solver.K))
Matrix(solver.K)
solver.b
solver.t
num_duals(qp)
num_ineq(qp)
num_eq(qp)


Zsqp, λsqp, dZ = solve_sqp!(nlp, copy(Z), copy(λ), iters=2)
Z = copy(Zsqp)
λ = copy(λsqp)
primal_residual(nlp, Z, λ)
dual_residual(nlp, Z, λ)

##
ϕ = NormPenalty(10.0, 1, num_primals(nlp), num_eq(nlp))
phi0 = ϕ(nlp, Z)
dphi0 = dgrad(ϕ, nlp, Z, dZ)
dphi1 = gradient(ϕ, nlp, Z)'dZ

meritfun(α) = ϕ(nlp, Z + α*dZ)
meritfun(1) - meritfun(0)
using Plots
alphas = -range(0,10, length=101)
phi = meritfun.(alphas)
plot(alphas, phi)
g = gradient(ϕ, nlp, Z)
g'dZ
##
Zsqp - Zipopt
λsqp - λipopt
primal_residual(nlp, Zipopt, λipopt)
dual_residual(nlp, Zipopt, λipopt)
eval_f(nlp, Zipopt)

Zsqp, λsqp = solve_sqp!(nlp, copy(Z), copy(λ), iters=10)