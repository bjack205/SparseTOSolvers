import Pkg; Pkg.activate(joinpath(@__DIR__,".."))
include(joinpath(@__DIR__, "..", "src", "nlp.jl"))
include("problems.jl")

using ForwardDiff
using FiniteDiff
using SparseArrays
using SparseDiffTools
using BenchmarkTools
using Test

prob = CartpoleProb()
Z = Vector(prob.Z)

## Build NLP
nlp = NLP(prob, true)
n,m,T = size(nlp)
N = num_primals(nlp)
M = num_duals(nlp)

############################################################################################
##                              COST EXPANSION
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
@test grad ≈ grad0
norm(hess - hess0)

# Analytical
hess = spzeros(N,N)
grad_f!(nlp, grad, Z)
hess_f!(nlp, hess, Z)
@test grad ≈ grad0
@test hess ≈ hess0

# Hessian-vector product
hvp = zero(grad)
hvp_f!(nlp, hvp, Z)
@test hvp ≈ hess*Z

## Timing results
@btime ForwardDiff.gradient!($grad, x->eval_f($nlp,x), $Z)
@btime FiniteDiff.finite_difference_gradient!($grad, x->eval_f($nlp,x), $Z, $grad_cache)
@btime grad_f!($nlp, $grad, $Z)

@btime begin
    hess_f!($nlp, $hess, $Z)
    $hess*$Z
end
@btime hvp_f!($nlp, $hvp, $Z, false)  # this is about 20x faster!

############################################################################################
##                                 CONSTRAINTS
############################################################################################
# constraints
Z = rand(N)
jac = spzeros(M,N)
c = zeros(M)
con(c,x) = eval_dynamics_constraints!(nlp,c,x)

# Test constraint
con(c,Z)
@test c[1:n] ≈ Z[1:n] - prob.x0
@test c[n+1:2n] ≈ discrete_dynamics(RK3, prob.model, 
    Z[nlp.xinds[1]] , Z[nlp.uinds[1]], 0.0, prob.Z[1].dt) - Z[nlp.xinds[2]]
if termcon(nlp)
    @test c[end-n+1:end] ≈ Z[nlp.xinds[end]] - prob.xf
end

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
@test jac ≈ jac0

# Analytical
jac_dynamics!(nlp, jac, Z)
@test jac ≈ jac0

## Timing results
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
jacvec_dynamics!(nlp, grad_L, Z, λ, true)
@test grad_L ≈ jac'λ

## Gradient of the Lagrangian ##
# ForwardDiff 
lag(x) = lagrangian(nlp, x, λ)
ForwardDiff.gradient!(grad_L, lag, Z)
grad_L0 = copy(grad_L)
@test lag(Z) ≈ eval_f(nlp,Z) - c'λ

# FiniteDiff
FiniteDiff.finite_difference_gradient!(grad_L, x->lagrangian(nlp,x,λ,c), Z, grad_cache)
@test grad_L ≈ grad_L0

# Analytical
grad_lagrangian!(nlp, grad_L, Z, λ)
@test grad_L ≈ grad_L0

# Timing
tmp = zeros(n+m)
@btime ForwardDiff.gradient!($grad, x->lagrangian($nlp,x,$λ), $Z)
@btime FiniteDiff.finite_difference_gradient!($grad, x->lagrangian($nlp,x,$λ,$c), $Z, $grad_cache)
@btime grad_lagrangian!($nlp, $grad_L, $Z, $λ, $tmp)

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
@test hess_L ≈ hess_L0

# Analytical
hess_lagrangian!(nlp, hess_L, Z, λ)
@test hess_L ≈ hess_L0

# Timing results
@btime ForwardDiff.jacobian!($hess_L, $gradlag!, $grad_L, $Z)
@btime FiniteDiff.finite_difference_jacobian!($hess_L, $gradlag!, $Z, $hessL_cache)
@btime FiniteDiff.finite_difference_jacobian!($hess_L, $gradlag!, $Z, $hessL_cache, 
    colorvec=$colorvec)
@btime forwarddiff_color_jacobian!($hess_L, $gradlag!, $Z, dx=$grad_L, colorvec=$colorvec)
@btime hess_lagrangian!($nlp, $hess_L, $Z, $λ)

############################################################################################
#                              AUGMENTED LAGRANGIAN
############################################################################################
# ρ = 7.0
# ρ = fill(7.0, length(λ))
ρ = rand(length(c)) * 7.0
Iρ = Diagonal(sqrt.(ρ))
λ = randn(length(c))
J = eval_f(nlp, Z)
eval_c!(nlp, c, Z)

@test aug_lagrangian(nlp, Z, λ, ρ) ≈ lagrangian(nlp, Z, λ) + 0.5*(Iρ*c)'*(Iρ*c)
@test aug_lagrangian(nlp, Z, λ, ρ) ≈ J - λ'c + 0.5*(Iρ*c)'*(Iρ*c)
@test aug_lagrangian(nlp, Z, λ, ρ) ≈ J - λ'c + 0.5*(Iρ*c)'*(Iρ*c)

λbar = λ - Iρ*Iρ * c
@test aug_lagrangian(nlp, Z, λ, ρ) ≈ J + 1/2*((Iρ\λbar)'*(Iρ\λbar) - (Iρ\λ)'*(Iρ\λ)) 
al(Z) = aug_lagrangian(nlp, Z, λ, ρ)

## Gradient ##
# ForwardDiff
ForwardDiff.gradient!(grad0, al,Z)

# FiniteDiff
FiniteDiff.finite_difference_gradient!(grad, al, Z, grad_cache)
norm(grad - grad0) < 1e-5

# Analytical
algrad!(nlp, grad, Z, λ, ρ)
@test grad ≈ grad0
algrad!(nlp, grad, Z, λ, ρ, c, tmp)

# Timing
@btime ForwardDiff.gradient!($grad0, x->aug_lagrangian($nlp, x, $λ, $ρ), $Z)
@btime FiniteDiff.finite_difference_gradient!($grad0, x->aug_lagrangian($nlp, x, $λ, $ρ), $Z, $grad_cache)
@btime algrad!($nlp, $grad, $Z, $λ, $ρ, $c, $tmp)

## Directional Derivative ##
dZ = randn(N)
grad_f!(nlp, grad0, Z)
@test dgrad_f(nlp, Z, dZ) ≈ grad0'dZ

algrad!(nlp, grad, Z, λ, ρ)
@test al_dgrad(nlp, Z, dZ, λ, ρ) ≈ grad'dZ

@btime begin
    algrad!($nlp, $grad, $Z, $λ, $ρ, $c, $tmp)
    $grad'*$dZ
end
@btime al_dgrad($nlp, $Z, $dZ, $λ, $ρ, $grad, $c, $tmp)

## Hessian ##
# ForwardDiff
hess0 = spzeros(N,N)
ForwardDiff.hessian!(hess0, al, Z)

# FiniteDiff
FiniteDiff.finite_difference_hessian!(hess, al, Z, hess_cache)
@test norm(hess - hess0) < 1e-1

# Analytical
alhess!(nlp, hess, Z, λ, ρ, false)
@test hess ≈ hess0
alhess!(nlp, hess, Z, λ, ρ, false, c, jac)
@test hess ≈ hess0

# Analytical Gauss-Newton
hessf = spzeros(N,N)
hess_f!(nlp, hessf, Z, true)

alhess!(nlp, hess, Z, λ, ρ)
jac_c!(nlp, jac, Z)
@test hess ≈ (hessf + ρ*jac'jac)

# Timing
@btime ForwardDiff.hessian!($hess, x->aug_lagrangian($nlp, x, $λ, $ρ), $Z)
@btime FiniteDiff.finite_difference_hessian!($hess, x->aug_lagrangian($nlp, x, $λ, $ρ), $Z, $hess_cache)

@btime alhess!($nlp, $hess, $Z, $λ, $ρ, false, $c, $jac)
@btime alhess!($nlp, $hess, $Z, $λ, $ρ, true, $c, $jac)


############################################################################################
##                                 TERMINAL CONSTRAINT
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
@test jac ≈ jac0

# Analytical
jac_dynamics!(nlp, jac, Z)
@test jac ≈ jac0

## Timing results
@btime ForwardDiff.jacobian!($jac, $con, $c, $Z)
@btime forwarddiff_color_jacobian!($jac, $con, $Z, dx=$c, colorvec=$colvec)
@btime FiniteDiff.finite_difference_jacobian!($jac,$con,$Z, $jac_cache, colorvec=$colvec)
@btime jac_dynamics!($nlp, $jac, $Z)


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

