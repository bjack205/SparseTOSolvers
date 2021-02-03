using ForwardDiff
using FiniteDiff
using SparseArrays
using SparseDiffTools
using OSQP
using BenchmarkTools
using Test
using SuiteSparse
using Printf
using Ipopt

const SRC = joinpath(@__DIR__, "..", "src")
include(joinpath(SRC, "nlp.jl"))
include(joinpath(SRC, "qp_solvers.jl"))
include(joinpath(SRC, "meritfuns.jl"))
include(joinpath(SRC, "moi.jl"))
include("problems.jl")

##
prob = CartpoleProb()
nlp = NLP(prob)
Z = Vector(prob.Z)

optimizer = Ipopt.Optimizer()
build_MOI!(optimizer, nlp, Z)
MOI.optimize!(optimizer)
λipopt = MOI.get(optimizer, MOI.NLPBlockDual())
Zipopt = MOI.get(optimizer, MOI.VariablePrimal(), MOI.VariableIndex.(1:504))

using Altro
prob = CartpoleProb()
solver = ALTROSolver(prob)
solve!(solver)
cost(solver)

prob2 = Cartpole() 
TO.add_dynamics_constraints!(prob)

nlp = TO.TrajOptNLP(prob, remove_bounds=true, jac_type=:vector)
optimizer = Ipopt.Optimizer()
TO.build_MOI!(nlp, optimizer)
MOI.optimize!(optimizer)

