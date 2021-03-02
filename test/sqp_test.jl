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
include(joinpath(SRC, "sqp.jl"))
include("problems.jl")

## Generate Problem

## Unconstrained
prob = CartpoleProb(101, 1000) 
nlp = NLP(prob, false)
Z = Vector(prob.Z)
λ = zeros(num_duals(nlp))

@time Zsqp, λsqp, dZ = solve_sqp!(nlp, copy(Z), copy(λ), 
    eps_primal=1e-4, eps_dual=1e-6, eps_fn=1e-1,
    iters=2000, qp_solver=:kkt, verbose=1)
norm(Zsqp[nlp.xinds[end]] - prob.xf)

## Constrained
prob = CartpoleProb(101, 1) 
nlp = NLP(prob, true)
Z = Vector(prob.Z)
λ = zeros(num_duals(nlp))

@time Zsqp, λsqp, dZ = solve_sqp!(nlp, copy(Z), copy(λ), 
    eps_primal=1e-3, eps_dual=1e-6, eps_fn=1e0,
    iters=1000, qp_solver=:kkt, verbose=1)
norm(Zsqp[nlp.xinds[end]] - prob.xf)

##
prob = CartpoleProb(101)
nlp = NLP(prob)
Z = Vector(prob.Z)
λ = zeros(num_duals(nlp))
# @time Zsqp, λsqp, dZ = solve_sqp!(nlp, copy(Z), copy(λ), iters=2000, qp_solver=:shur, verbose=1)
@time Zsqp, λsqp, dZ = solve_sqp!(nlp, copy(Z), copy(λ), iters=30, qp_solver=:kkt, hess=:newton, verbose=1)

## Gauss-Newton
@time Zsqp, λsqp, dZ = solve_sqp!(nlp, copy(Z), copy(λ), iters=30, qp_solver=:shur, verbose=1)
@time Zsqp, λsqp, dZ = solve_sqp!(nlp, copy(Z), copy(λ), iters=30, qp_solver=:osqp, verbose=1)
@time Zsqp, λsqp, dZ = solve_sqp!(nlp, copy(Z), copy(λ), iters=30, qp_solver=:kkt, verbose=1)
@time Zsqp, λsqp, dZ = solve_sqp!(nlp, copy(Z), copy(λ), iters=30, qp_solver=:qdldl_kkt, verbose=1)
@time Zsqp, λsqp, dZ = solve_sqp!(nlp, copy(Z), copy(λ), iters=30, qp_solver=:qdldl_shur, verbose=1)

# Full Hessian
@time Zsqp, λsqp, dZ = solve_sqp!(nlp, copy(Z), copy(λ), iters=30, qp_solver=:kkt, hess=:newton, verbose=1)
@time Zsqp, λsqp, dZ = solve_sqp!(nlp, copy(Z), copy(λ), iters=30, qp_solver=:qdldl_kkt, hess=:newton, verbose=1) 
@time Zsqp, λsqp, dZ = solve_sqp!(nlp, copy(Z), copy(λ), iters=30, qp_solver=:qdldl_shur, hess=:newton, verbose=1)

# BFGS
solve_sqp!(nlp, copy(Z), copy(λ), iters=30, 
    qp_solver=:shur, hess=:bfgs, verbose=1)


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