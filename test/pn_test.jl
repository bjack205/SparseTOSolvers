using ForwardDiff
using FiniteDiff
using SparseArrays
using SparseDiffTools
using OSQP
using BenchmarkTools
using Test
using SuiteSparse
using Printf
using Altro

const SRC = joinpath(@__DIR__, "..", "src")
include(joinpath(SRC, "nlp.jl"))
include(joinpath(SRC, "qp_solvers.jl"))
include(joinpath(SRC, "meritfuns.jl"))
include("problems.jl")

##
prob = CartpoleProb()
nlp = NLP(prob)
Z = Vector(prob.Z)
λ = zeros(num_duals(nlp))

Z = copy(Zsqp)
λ = copy(λsqp)

qp = TOQP(size(nlp)..., num_eq(nlp), 0)
ϕ = NormPenalty(10.0, 1, num_primals(nlp), num_eq(nlp))

res_p = primal_residual(nlp, Z, λ)
res_d = dual_residual(nlp, Z, λ)
J = eval_f(nlp, Z)

# Get QP
build_qp!(qp, nlp, Z, λ, true)
H,q = qp.Q, qp.q
D,d = qp.A, qp.b

# Solve KKT
N = num_primals(nlp)
M = num_duals(nlp)
K = [H D'; D spzeros(M,M)]
k = [-q; d]
dY = K\k
dZ = dY[1:N]
dλ = dY[N+1:end]

res_p = primal_residual(nlp, Z + dZ, λ - dλ)
res_d = dual_residual(nlp, Z + dZ, λ - dλ)
ϕ(nlp, Z + dZ) < ϕ(nlp, Z)
dgrad(ϕ, nlp, Z, dZ)

# Solve OSQP
model = OSQP.Model()
dZ2, dλ2 =  solve_qp!(model, qp, polish=true, eps_abs=1e-10, eps_rel=1e-10)
res_p = primal_residual(nlp, Z + dZ2, λ - dλ2)
res_d = dual_residual(nlp, Z + dZ2, λ - dλ2)
ϕ(nlp, Z + dZ2) < ϕ(nlp, Z)
dgrad(ϕ, nlp, Z, dZ2)

# Solve Schur compliment
S = Symmetric(D*(H\D'))
s = D*(H\q) + d
dλ3 = -S\s
dZ3 = -H\(D'dλ3 + q)
res_p = primal_residual(nlp, Z, λ)
res_d = dual_residual(nlp, Z, λ)
res_p = primal_residual(nlp, Z + dZ3, λ - dλ3)
res_d = dual_residual(nlp, Z + dZ3, λ - dλ3)
ϕ(nlp, Z + dZ3) < ϕ(nlp, Z)
ϕ(nlp, Z + dZ3) - ϕ(nlp, Z)
dgrad(ϕ, nlp, Z, dZ3)
norm(dZ3 - dZ)
norm(dλ3 - dλ)

ssolver = ShurSolver()
dZ4, dλ4 =  solve_qp!(ssolver, qp)
res_p = primal_residual(nlp, Z + dZ4, λ - dλ4)
res_d = dual_residual(nlp, Z + dZ4, λ - dλ4)
norm(dZ4 - dZ)
norm(dλ4 - dλ)

kktsolver = KKTSolver(qp)
dZ5, dλ5 = solve_qp!(kktsolver, qp)
res_p = primal_residual(nlp, Z + dZ5, λ - dλ5)
res_d = dual_residual(nlp, Z + dZ5, λ - dλ5)
norm(dZ5 - dZ)
norm(dλ5 - dλ)


α = 1.0
Zbar = Z + α*dZ 
λbar = λ + dλ

primal_residual(nlp, Zbar, λbar)
dual_residual(nlp, Zbar, λbar)