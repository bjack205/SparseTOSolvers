import Pkg; Pkg.activate(joinpath(@__DIR__,".."))
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
include(joinpath(SRC, "auglag.jl"))

include("problems.jl")

##
prob = CartpoleProb(101, 1000)
# prob = DoubleIntegrator()
nlp = NLP(prob,false)
Z = Vector(prob.Z)
λ = zeros(num_duals(nlp))

Zsol, λsol, res = al_solve(nlp, Z, eps_inner=1e-1, eps_constraint=1e-6, 
    newton_reg=1e-6, eps_fn=1e1, ρ=1e4, ϕ=10.0, newton_iters=100, refine=0)
# Zsol, λsol, res = al_solve(nlp, Z, eps_inner=1e-3, newton_reg=1e-4, eps_fn=1e-3, ρ=1e4, pd=true)
norm(Zsol[nlp.xinds[end]] - prob.xf)

##
ρ = 100.0

Znew = al_newton(nlp, Z, λ, ρ, max_iters=100)
primal_residual(nlp, Znew, λ)
dual_residual(nlp, Znew, λ)

##
λnew = dual_update!(nlp, Z, λ, ρ)
primal_residual(nlp, Z, λnew)
dual_residual(nlp, Z, λnew)
λ .= λnew
ρ *= 10

