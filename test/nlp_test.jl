include(joinpath(@__DIR__, "..", "src", "nlp.jl"))
include("problems.jl")

using ForwardDiff
using FiniteDiff
using SparseArrays
using SparseDiffTools
using BenchmarkTools

prob = Cartpole()
Z = Vector(prob.Z)

## Build NLP
nlp = NLP(prob)
n,m,T = size(nlp)
N = num_vars(nlp)
M = num_cons(nlp)

############################################################################################
#                              COST EXPANSION
############################################################################################
# compare costs
J = eval_f(nlp, Z)
abs(cost(prob) - J)

# Forward Diff
grad = zeros(N)
hess = spzeros(N,N)
ForwardDiff.gradient!(grad, x->eval_f(nlp,x), Z)
ForwardDiff.hessian!(hess, x->eval_f(nlp,x), Z)
grad0 = copy(grad)
hess0 = copy(hess)

# FiniteDiff
grad_cache = FiniteDiff.GradientCache(grad, Z)
hess_cache = FiniteDiff.HessianCache(Z)
FiniteDiff.finite_difference_gradient!(grad, x->eval_f(nlp,x), Z, grad_cache)
FiniteDiff.finite_difference_hessian!(hess, x->eval_f(nlp,x), Z, hess_cache)
grad ≈ grad0
norm(hess - hess0)

# Analytical
hess = spzeros(N,N)
grad_f!(nlp, grad, Z)
hess_f!(nlp, hess, Z)
grad ≈ grad0
hess ≈ hess0

# Timing results
@btime ForwardDiff.gradient!($grad, x->eval_f($nlp,x), $Z)
@btime FiniteDiff.finite_difference_gradient!($grad, x->eval_f($nlp,x), $Z, $grad_cache)
@btime grad_f!($nlp, $grad, $Z)

############################################################################################
#                          CONSTRAINTS
############################################################################################
# constraints
Z = rand(N)
jac = spzeros(M,N)
c = zeros(M)
con(c,x) = eval_dynamics_constraints!(nlp,c,x)

# FiniteDiff coloring
jac_cache = FiniteDiff.JacobianCache(Z,c)
FiniteDiff.finite_difference_jacobian!(jac,con,Z,jac_cache)
spar = jac .!= 0
colvec = matrix_colors(spar)

# ForwardDiff + coloring
ForwardDiff.jacobian!(jac, con, c, Z)
forwarddiff_color_jacobian!(jac, con, Z, dx=c, colorvec=colvec)
jac0 = copy(jac)

# FiniteDiff
FiniteDiff.finite_difference_jacobian!(jac,con,Z,jac_cache)
FiniteDiff.finite_difference_jacobian!(jac,con,Z, jac_cache, colorvec=colvec)
jac ≈ jac0

# Analytical
jac_dynamics!(nlp, jac, Z)
jac ≈ jac0

# Timing results
@btime ForwardDiff.jacobian!($jac, $con, $c, $Z)
@btime forwarddiff_color_jacobian!($jac, $con, $Z, dx=$c, colorvec=$colvec)
@btime FiniteDiff.finite_difference_jacobian!($jac,$con,$Z, $jac_cache, colorvec=$colvec)
@btime jac_dynamics!($nlp, $jac, $Z)

############################################################################################
#                                 LAGRANGIAN
############################################################################################
λ = rand(M)
L = lagrangian(nlp, Z, λ, c)
grad_L = zeros(N)

# Check gradient of constraint term
jacvec_dynamics!(nlp, grad_L, Z, λ)
grad_L ≈ jac'λ

## Gradient of the Lagrangian ##
# ForwardDiff 
lag(x) = lagrangian(nlp, x, λ)
ForwardDiff.gradient!(grad_L, lag, Z)
grad_L0 = copy(grad_L)

# FiniteDiff
FiniteDiff.finite_difference_gradient!(grad_L, x->lagrangian(nlp,x,λ,c), Z, grad_cache)
grad_L ≈ grad_L0

# Analytical
grad_lagrangian!(nlp, grad_L, Z, λ)
grad_L ≈ grad_L0

# Timing
@btime ForwardDiff.gradient!($grad, x->lagrangian($nlp,x,$λ), $Z)
@btime FiniteDiff.finite_difference_gradient!($grad, x->lagrangian($nlp,x,$λ,$c), $Z, $grad_cache)
@btime grad_lagrangian!($nlp, $grad_L, $Z, $λ)

## Hessian of the Lagrangian ##
hess_L = spzeros(N,N)
gradlag!(g,x) = grad_lagrangian!(nlp, g, x, λ)

# ForwardDiff
ForwardDiff.jacobian!(hess_L, gradlag!, grad_L, Z)
hess_L0 = copy(hess_L)

# FiniteDiff
pat = hess_L .!= 0
colorvec = matrix_colors(pat)
hessL_cache = FiniteDiff.JacobianCache(Z,grad_L)
FiniteDiff.finite_difference_jacobian!(hess_L, gradlag!, Z, hessL_cache)
FiniteDiff.finite_difference_jacobian!(hess_L, gradlag!, Z, hessL_cache, colorvec=colorvec)
hess_L ≈ hess_L0

# Analytical
hess_lagrangian!(nlp, hess_L, Z, λ)
hess_L ≈ hess_L0

# Timing results
@btime ForwardDiff.jacobian!($hess_L, $gradlag!, $grad_L, $Z)
@btime FiniteDiff.finite_difference_jacobian!($hess_L, $gradlag!, $Z, $hessL_cache)
@btime FiniteDiff.finite_difference_jacobian!($hess_L, $gradlag!, $Z, $hessL_cache, 
    colorvec=$colorvec)
@btime forwarddiff_color_jacobian!($hess_L, $gradlag!, $Z, dx=$grad_L, colorvec=$colorvec)
@btime hess_lagrangian!($nlp, $hess_L, $Z, $λ)


## modelling toolkit
using Model
@variables Zsym[1:N]
costexpr = eval_f(nlp, Zsym)
costexpr2 = simplify(costexpr)
eval_f2 = eval(build_function(costexpr, Zsym))
grad = ModelingToolkit.gradient(costexpr, Zsym, simplify=true)
eval_grad = eval(build_function(grad, Zsym)[2])
g = zeros(N)

@btime eval_f2($Z)
@btime eval_f($nlp, $Z)
@btime eval_grad($g, $Z)

@variables csym[1:M]
eval_dynamics_constraints!(nlp, csym, Zsym)
conexpr2 = simplify.(csym)
jacexpr = ModelingToolkit.sparsejacobian(conexpr2, Zsym)
jac_c = eval(build_function(jacexpr, Zsym)[2])
@btime jac_c($jac, $Z)
@btime FiniteDiff.finite_difference_jacobian!($jac, $con,Z, $jac_cache, colorvec=$colvec)
@btime forwarddiff_color_jacobian!($jac, $con, $Z, dx=$c, colorvec=$colvec)

