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
build_qp!(qp, nlp, Z, λ, false)
H0 = copy(qp.Q)  # full Hessian 

qp.Q .= 0
dropzeros!(qp.Q)
build_qp!(qp, nlp, Z, λ, true)
H,q = qp.Q, qp.q
D,d = qp.A, qp.b

# Solve KKT
N = num_primals(nlp)
M = num_duals(nlp)
K = [H D'; D spzeros(M,M)]
K = [H D'; D -I*1e-12]
t = [-q; d]
dY = K\t
dZ = dY[1:N]
dλ = dY[N+1:end]

res_p = primal_residual(nlp, Z + dZ, λ - dλ)
res_d = dual_residual(nlp, Z + dZ, λ - dλ)
ϕ(nlp, Z + dZ) < ϕ(nlp, Z)
dgrad(ϕ, nlp, Z, dZ)

# Solve OSQP
model = OSQP.Model()
OSQP.setup!(model, P=qp.Q, q=qp.q, A=qp.A, l=qp.b, u=qp.b; 
    polish=true, eps_abs=1e-10, eps_rel=1e-10, verbose=false)
dZ2, dλ2 =  solve_qp!(model, qp, setup=false) 
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

## KKT Solver
kktsolver = KKTSolver(qp)
dZ5, dλ5 = solve_qp!(kktsolver, qp)
res_p = primal_residual(nlp, Z + dZ5, λ - dλ5)
res_d = dual_residual(nlp, Z + dZ5, λ - dλ5)
norm(dZ5 - dZ)
norm(dλ5 - dλ)

## QDLDL Solver
qd_kkt = QDLDLSolver(qp)
dZ6, dλ6 = solve_qp!(qd_kkt, qp)
res_p = primal_residual(nlp, Z + dZ6, λ - dλ6)
res_d = dual_residual(nlp, Z + dZ6, λ - dλ5)
norm(dZ6 - dZ)
norm(dλ6 - dλ)

qd_shur = QDLDLSolver(qp, method=:shur)
dZ7, dλ7 = solve_qp!(qd_shur, qp)
res_p = primal_residual(nlp, Z + dZ7, λ - dλ7)
res_d = dual_residual(nlp, Z + dZ7, λ - dλ7)
norm(dZ7 - dZ)
norm(dλ7 - dλ)

## timing comparison
OSQP.update_settings!(model, eps_abs=1e-3, eps_rel=1e-3, polish=false)
@btime solve_qp!($model, $qp, setup=false) 
@btime solve_qp!($ssolver, $qp, gauss_newton=true)
@btime solve_qp!($qd_shur, $qp)   # 4x faster  (fastest)
@btime solve_qp!($kktsolver, $qp)
@btime solve_qp!($qd_kkt, $qp)    # 3x faster 


## Try with Full Hessian
qp.Q .= 0
dropzeros!(qp.Q)
build_qp!(qp, nlp, Z, λ, false)
dZ0, dλ0 = solve_qp!(kktsolver, qp)
dZ1, dλ1 =  solve_qp!(qd_kkt, qp)
dZ2, dλ2 =  solve_qp!(ssolver, qp)
dZ3, dλ3 =  solve_qp!(qd_shur, qp)
norm(dZ0 - dZ1)
norm(dZ0 - dZ2)
norm(dZ0 - dZ3)
norm(dλ0 - dλ1)
norm(dλ0 - dλ2)
norm(dλ0 - dλ3)

@btime solve_qp!($kktsolver, $qp)   # moderate
@btime solve_qp!($qd_kkt, $qp)      # fastest (2x)
@btime solve_qp!($ssolver, $qp)     # moderate
@btime solve_qp!($qd_shur, $qp)     # slowest
