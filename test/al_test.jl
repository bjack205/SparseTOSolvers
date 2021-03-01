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
# function al_solve(nlp, Z, λ=zeros(num_eq(nlp)); 
#         ρ=10.0,   # initial penalty
#         ϕ=10.0,   # penalty scaling
#         max_iters = 20, 
#         eps_constraint = 1e-6,
#         eps_inner = 1e-6,
#         newton_iters= 20,
#         newton_reg=1e-6,
#         eps_fn = sqrt(eps_inner),
#         verbose=2
#     )
    
#     c = zero(λ)
#     iters = 0
#     for i = 1:max_iters
#         # Solve unconstrained problem
#         iters += al_newton!(nlp, Z, λ, ρ, 
#             max_iters=newton_iters, eps=eps_inner, eps_fn=eps_fn, reg=newton_reg, 
#             verbose=verbose>1
#         ) 

#         # dual update
#         eval_c!(nlp, c, Z)
#         λ .= λ - ρ*c
        
#         # penalty update
#         ρ *= ϕ

#         # Convergence
#         pres = primal_residual(nlp, Z, λ, p=Inf)
#         printstyled("Outer Loop ", i, ": ", bold=true, color=:yellow)
#         verbose > 0 && @printf("viol: %0.2e, pres: %0.2e, ρ = %.1e, iters = %d\n", 
#             norm(c, Inf), pres, ρ, iters)
#         if norm(c, Inf) < eps_constraint
#             break
#         end
#     end

#     return Z, λ 
# end
# function al_newton!(nlp, Z, λ, ρ;
#         max_iters = 25,
#         eps = 1e-6,
#         reg = 1e-6,
#         eps_fn = sqrt(eps),
#         verbose=false
#     )
#     N,M = num_primals(nlp), num_duals(nlp)
#     hess = spzeros(N,N)
#     grad = zeros(N)

#     gn = true 
#     iters = 0
#     for iter = 1:max_iters
#         alhess!(nlp, hess, Z, λ, ρ, gn)
#         algrad!(nlp, grad, Z, λ, ρ)
#         if norm(grad) < eps
#             verbose && println("  Converged in $iter iterations")
#             break
#         elseif gn && norm(grad) < eps_fn 
#             verbose && println("  Switching to Full Newton")
#             gn = false
#         end
#         dZ = -(cholesky(Symmetric(hess) + I*reg)\grad)
#         phi0 = aug_lagrangian(nlp, Z, λ, ρ)
#         dphi0 = grad'dZ
#         phi = 0.0

#         c1 = 1e-3
#         c2 = 0.9
#         α = 1.0
#         for i = 1:10
#             phi = aug_lagrangian(nlp, Z + α*dZ, λ, ρ)
#             dphi = al_dgrad(nlp, Z + α*dZ, dZ, λ, ρ)
#             armijo = phi < phi0 + c1*α*dphi0
#             wolfe = dphi > c2*dphi0
#             if armijo && wolfe
#                 break
#             else
#                 α *= 0.5
#             end
#         end
#         if verbose
#             Z̄ = Z .+ α*dZ
#             J0 = eval_f(nlp, Z)
#             J = eval_f(nlp, Z̄)
#             algrad!(nlp, grad, Z̄, λ, ρ)
#             @printf("  J: %0.2f → %0.2f (%0.2e), ϕ: %0.2f → %0.2f (%0.2e), ϕ′: %0.2e, α: %0.2f, pres: %0.2e, dres: %0.2e, grad: %0.2e\n", 
#                 J0, J, J0-J, 
#                 phi0, phi, phi0-phi,
#                 dphi0, α,
#                 norm(primal_residual(nlp, Z̄, λ, p=Inf)),
#                 norm(dual_residual(nlp, Z̄, λ, p=Inf)),
#                 norm(grad)
#             )

#             iters += 1
#         end
#         # phi0 - aug_lagrangian(nlp, Z + α*dZ, λ, ρ)
#         # dual_residual(nlp, Z + α*dZ, λ)
#         # primal_residual(nlp, Z + 0*dZ, λ)
#         # primal_residual(nlp, Z + α*dZ, λ)
#         Z .+= α .* dZ
#     end
#     return iters
# end

##
prob = CartpoleProb()
# prob = DoubleIntegrator()
nlp = NLP(prob)
Z = Vector(prob.Z)
λ = zeros(num_duals(nlp))

al_solve(nlp, Z, eps_inner=1e-3, newton_reg=1e-4, eps_fn=1e-3, ρ=1e4)
al_solve(nlp, Z, eps_inner=1e-3, newton_reg=1e-4, eps_fn=1e-3, ρ=1e4, pd=true)

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

