include(joinpath(@__DIR__, "..", "src", "nlp.jl"))
include("problems.jl")

using ForwardDiff
using FiniteDiff
using SparseArrays
using SparseDiffTools
using BenchmarkTools

prob = Cartpole()
Z = Vector(prob.Z)

# Build NLP
nlp = NLP(prob)
n,m,T = size(nlp)
N = num_vars(nlp)
M = num_cons(nlp)

# Compare cost
J = eval_f(nlp, Z)
abs(cost(prob) - J)

# constraints
c = zeros(M)
eval_dynamics_constraints!(nlp, c, Z)
norm(c)

# cost expansion
grad = zeros(N)
hess = spzeros(N,N)
tad_grad = @elapsed ForwardDiff.gradient!(grad, x->eval_f(nlp,x), Z)
tad_hess = @elapsed ForwardDiff.hessian!(hess, x->eval_f(nlp,x), Z)

grad_cache = FiniteDiff.GradientCache(grad, Z)
hess_cache = FiniteDiff.HessianCache(Z)
tfd_grad = @elapsed FiniteDiff.finite_difference_gradient!(grad, x->eval_f(nlp,x), Z, grad_cache)
tfd_hess = @elapsed FiniteDiff.finite_difference_hessian!(hess, x->eval_f(nlp,x), Z, hess_cache)
tad_grad / tfd_grad
tad_hess / tfd_hess

grad2 = zero(grad)
grad_f!(nlp, grad2, Z)
norm(grad - grad2) < 1e-6

@btime ForwardDiff.gradient!($grad, x->eval_f($nlp,x), $Z)
@btime FiniteDiff.finite_difference_gradient!($grad, x->eval_f($nlp,x), $Z, $grad_cache)
@btime grad_f!($nlp, $grad, $Z)

hess2 = zero(hess)
hess_f!(nlp, hess2, Z)
norm(hess - hess2)


# constraints
Z = rand(N)
jac = spzeros(M,N)
con(c,x) = eval_dynamics_constraints!(nlp,c,x)

jac_cache = FiniteDiff.JacobianCache(Z,c)
FiniteDiff.finite_difference_jacobian!(jac,con,Z,jac_cache)
spar = jac .!= 0
colvec = matrix_colors(spar)

tfd = @elapsed FiniteDiff.finite_difference_jacobian!(jac,con,Z,jac_cache)
tfd2 = @elapsed FiniteDiff.finite_difference_jacobian!(jac,con,Z, jac_cache, colorvec=colvec)
tfd/tfd2

tad = @elapsed ForwardDiff.jacobian!(jac, con, c, Z)
tad2 = @elapsed forwarddiff_color_jacobian!(jac, con, Z, dx=c, colorvec=colvec)
tad/tad2

tad/tfd
tad2/tfd2

jac2 = zero(jac)
jac_dynamics!(nlp, jac2, Z)
norm(jac - jac2)

@btime forwarddiff_color_jacobian!($jac, $con, $Z, dx=$c, colorvec=$colvec)
@btime FiniteDiff.finite_difference_jacobian!($jac,$con,$Z, $jac_cache, colorvec=$colvec)
@btime jac_dynamics!($nlp, $jac2, $Z)

# lagrangian
λ = rand(M)
L = lagrangian(nlp, Z, λ, c)
grad_L = zeros(N)
jacvec_dynamics!(nlp, grad_L, Z, λ)
grad_L ≈ jac'λ

grad_lagrangian!(nlp, grad_L, Z, λ)
lag(x) = lagrangian(nlp, x, λ)
ForwardDiff.gradient!(grad, lag, Z)
grad_L ≈ grad
FiniteDiff.finite_difference_gradient!(grad, x->lagrangian(nlp,x,λ,c), Z, grad_cache)
grad_L ≈ grad

@btime ForwardDiff.gradient!($grad, x->lagrangian($nlp,x,$λ), $Z)
@btime FiniteDiff.finite_difference_gradient!($grad, x->lagrangian($nlp,x,$λ,$c), $Z, $grad_cache)
@btime grad_lagrangian!($nlp, $grad_L, $Z, $λ)

hess_L = spzeros(N,N)
gradlag!(g,x) = grad_lagrangian!(nlp, g, x, λ)
ForwardDiff.jacobian!(hess_L, gradlag!, grad_L, Z)
hess .= hess_L

pat = hess_L .!= 0
colorvec = matrix_colors(pat)
hessL_cache = FiniteDiff.JacobianCache(Z,grad_L)

FiniteDiff.finite_difference_jacobian!(hess_L, gradlag!, Z, hessL_cache)
FiniteDiff.finite_difference_jacobian!(hess_L, gradlag!, Z, hessL_cache, colorvec=colorvec)
hess_L ≈ hess

hess_lagrangian!(nlp, hess_L, Z, λ)
hess_L ≈ hess
Matrix(hess_L)
Matrix(hess)

hess_c = zero(hess)
hess_f = zero(hess)
# hess_f!(nlp, hess_c, Z, false)
hess_lagrangian!(nlp, hess_c, Z, λ)
hess_c ≈ hess


@btime ForwardDiff.jacobian!($hess_L, $gradlag!, $grad_L, $Z)
@btime FiniteDiff.finite_difference_jacobian!($hess_L, $gradlag!, $Z, $hessL_cache)
@btime FiniteDiff.finite_difference_jacobian!($hess_L, $gradlag!, $Z, $hessL_cache, 
    colorvec=$colorvec)
@btime hess_lagrangian!($nlp, $hess_c, $Z, $λ)

H = zeros(n+m,n+m)
λ_ = λ[SA[1,2,3,4]]
@btime RD.∇discrete_jacobian!(RK3, $H, $(prob.model), $(prob.Z[1]), $λ_)

# modelling toolkit
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

