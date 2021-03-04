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
using Plots
using Preconditioners

const SRC = joinpath(@__DIR__, "..", "src")
include(joinpath(SRC, "nlp.jl"))
include(joinpath(SRC, "qp_solvers.jl"))
include(joinpath(SRC, "meritfuns.jl"))
include(joinpath(SRC, "auglag.jl"))

include("problems.jl")


## Unconstrained
prob = CartpoleProb(101, 1000)
nlp = NLP(prob,false)
Z = Vector(prob.Z)
λ = zeros(num_duals(nlp))

Zsol, λsol, res = al_solve(nlp, Z, eps_inner=1e-1, eps_dual=1e-6, eps_primal=1e-4,
    newton_reg=1e-6, eps_fn=1e1, ρ=1e4, ϕ=10.0, newton_iters=100, refine=0, linear_solve=:qdldl)
norm(Zsol[nlp.xinds[end]] - prob.xf)
plot(res, label="qdldl", 
    title="Unconstrained", yscale=:log10, xlabel="iterations", ylabel="Constraint Violation")

## Gauss Newton
prob = CartpoleProb(101, 1000)
nlp = NLP(prob,false)
Z = Vector(prob.Z)
λ = zeros(num_duals(nlp))

Zsol, λsol, res = al_solve(nlp, Z, eps_inner=1e-1, eps_dual=1e-6, eps_primal=1e-4,
    newton_reg=1e-6, eps_fn=1e-11, ρ=1e4, ϕ=10.0, newton_iters=100, refine=0, linear_solve = :qdldl)
norm(Zsol[nlp.xinds[end]] - prob.xf)
plot!(res, label="qdldl-gn")

## PCG - Full Newton
prob = CartpoleProb(101, 1000)
nlp = NLP(prob,false)
Z = Vector(prob.Z)
λ = zeros(num_duals(nlp))

Zsol, λsol, res = al_solve(nlp, Z, eps_inner=1e-1, eps_dual=1e-6, eps_primal=1e-4,
    newton_reg=1e-6, eps_fn=1e1, ρ=1e4, ϕ=10.0, newton_iters=100, refine=0, linear_solve = :pcg)
norm(Zsol[nlp.xinds[end]] - prob.xf)
plot!(res, label="pcg")

## PCG - Gauss Newton
prob = CartpoleProb(101, 1000)
nlp = NLP(prob,false)
Z = Vector(prob.Z)
λ = zeros(num_duals(nlp))

Zsol, λsol, res = al_solve(nlp, Z, eps_inner=1e-1, eps_dual=1e-6, eps_primal=1e-4,
    newton_reg=1e-6, eps_fn=1e-11, ρ=1e4, ϕ=10.0, newton_iters=100, refine=0, linear_solve = :pcg)
norm(Zsol[nlp.xinds[end]] - prob.xf)
plot!(res, label="pcg-gn")


## Unconstrained Primal-Dual
prob = CartpoleProb(101, 1000)
nlp = NLP(prob,false)
Z = Vector(prob.Z)
λ = zeros(num_duals(nlp))

Zsol, λsol, res = al_solve(nlp, Z, eps_inner=1e-1, eps_dual=1e-6, eps_primal=1e-4,
    newton_reg=1e-6, eps_fn=1e1, ρ=1e4, ϕ=10.0, newton_iters=100, refine=0, pd=true)
norm(Zsol[nlp.xinds[end]] - prob.xf)
plot!(res, label="qdldl-pd")

############################################################################################
##                                      Constrained
############################################################################################

## Full Newton
prob = CartpoleProb(101, 1)
nlp = NLP(prob,true)
Z = Vector(prob.Z)
λ = zeros(num_duals(nlp))

Zsol, λsol, res = al_solve(nlp, Z, eps_inner=1e-2, 
    newton_reg=1e-4, eps_fn=1e2, pd=false, 
    max_iters=20, newton_iters=100,
    ρ = 1e4, ρterm=1e1, 
    linear_solve = :qdldl,
    )
norm(Zsol[nlp.xinds[end]] - prob.xf)
plot(res, label="qdldl", 
    title="Constrained", yscale=:log10, xlabel="iterations", ylabel="Constraint Violation")

## Gauss Newton
prob = CartpoleProb(101, 1)
nlp = NLP(prob,true)
Z = Vector(prob.Z)
λ = zeros(num_duals(nlp))

Zsol, λsol, res = al_solve(nlp, Z, eps_inner=1e-2, 
    newton_reg=1e-4, eps_fn=1e-22, pd=false, 
    max_iters=20, newton_iters=100,
    ρ = 1e4, ρterm=1e1, 
    linear_solve = :qdldl,
    )
norm(Zsol[nlp.xinds[end]] - prob.xf)
plot!(res, label="qdldl-gn")

## PCG - Full Newton 
prob = CartpoleProb(101, 1)
nlp = NLP(prob,true)
Z = Vector(prob.Z)
λ = zeros(num_duals(nlp))

Zsol, λsol, res = al_solve(nlp, Z, eps_inner=1e-2, 
    newton_reg=1e-4, eps_fn=1e2, pd=false, 
    max_iters=5, newton_iters=100,
    ρ = 1e4, ρterm=1e1, 
    linear_solve = :pcg,
)
norm(Zsol[nlp.xinds[end]] - prob.xf)
plot!(res, label="pcg")

## PCG - Gauss Newton 
prob = CartpoleProb(101, 1)
nlp = NLP(prob,true)
Z = Vector(prob.Z)
λ = zeros(num_duals(nlp))

Zsol, λsol, res = al_solve(nlp, Z, eps_inner=1e-2, 
    newton_reg=1e-4, eps_fn=1e-22, pd=false, 
    max_iters=5, newton_iters=100,
    ρ = 1e4, ρterm=1e1, 
    linear_solve = :pcg,
)
norm(Zsol[nlp.xinds[end]] - prob.xf)
plot!(res, label="pcg-gn")


############################################################################################
##                              Preconditioners
############################################################################################
prob = CartpoleProb(101, 1000)
nlp = NLP(prob,false)
Z = Vector(prob.Z)
λ = zeros(num_duals(nlp))

Zsol, λsol, res = al_solve(nlp, Z, eps_inner=1e-1, eps_dual=1e-6, eps_primal=1e-4,
    newton_reg=1e-6, eps_fn=1e1, ρ=1e4, ϕ=10.0, newton_iters=100, refine=0, 
    linear_solve = :pcg, Pl=nothing
)
norm(Zsol[nlp.xinds[end]] - prob.xf)
plot(res, label="identity",
    title="Preconditioners", yscale=:log10, xlabel="iterations", ylabel="Constraint Violation"
)

## Diagonal
prob = CartpoleProb(101, 1000)
nlp = NLP(prob,false)
Z = Vector(prob.Z)
λ = zeros(num_duals(nlp))

Zsol, λsol, res = al_solve(nlp, Z, eps_inner=1e-1, eps_dual=1e-6, eps_primal=1e-4,
    newton_reg=1e-6, eps_fn=1e1, ρ=1e4, ϕ=10.0, newton_iters=100, refine=0, 
    linear_solve = :pcg, Pl=DiagonalPreconditioner
)
norm(Zsol[nlp.xinds[end]] - prob.xf)
plot!(res, label="diagonal")

## Cholesky 
prob = CartpoleProb(101, 1000)
nlp = NLP(prob,false)
Z = Vector(prob.Z)
λ = zeros(num_duals(nlp))

Zsol, λsol, res = al_solve(nlp, Z, eps_inner=1e-1, eps_dual=1e-6, eps_primal=1e-4,
    newton_reg=1e-6, eps_fn=1e1, ρ=1e4, ϕ=10.0, newton_iters=100, refine=0, 
    linear_solve = :pcg, Pl=CholeskyPreconditioner
)
norm(Zsol[nlp.xinds[end]] - prob.xf)
plot!(res, label="cholesky")

## AMG
prob = CartpoleProb(101, 1000)
nlp = NLP(prob,false)
Z = Vector(prob.Z)
λ = zeros(num_duals(nlp))

Zsol, λsol, res = al_solve(nlp, Z, eps_inner=1e-1, eps_dual=1e-6, eps_primal=1e-4,
    newton_reg=1e-6, eps_fn=1e1, ρ=1e4, ϕ=10.0, newton_iters=100, refine=0, 
    linear_solve = :pcg, Pl=AMGPreconditioner{RugeStuben}
)
norm(Zsol[nlp.xinds[end]] - prob.xf)
plot!(res, label="AMG-RugeStuben")

## AMG-Smoothed
prob = CartpoleProb(101, 1000)
nlp = NLP(prob,false)
Z = Vector(prob.Z)
λ = zeros(num_duals(nlp))

Zsol, λsol, res = al_solve(nlp, Z, eps_inner=1e-1, eps_dual=1e-6, eps_primal=1e-4,
    newton_reg=1e-6, eps_fn=1e1, ρ=1e4, ϕ=10.0, newton_iters=100, refine=0, 
    linear_solve = :pcg, Pl=AMGPreconditioner{SmoothedAggregation}
)
norm(Zsol[nlp.xinds[end]] - prob.xf)
plot!(res, label="AMG-SmoothedAggregation")