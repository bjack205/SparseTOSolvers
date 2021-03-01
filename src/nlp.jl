using TrajectoryOptimization
using RobotDynamics
using StaticArrays
using RobotZoo
using MathOptInterface
const TO = TrajectoryOptimization
const RD = RobotDynamics
const MOI = MathOptInterface

import TrajectoryOptimization: num_vars

struct NLP{n,m,Q,L,C} <: MOI.AbstractNLPEvaluator
    model::L
    obj::Objective{C}
    T::Int  # number of knot points
    tf::Float64
    x0::Vector{Float64}  # initial condition
    xf::Vector{Float64}  # final condition
    xinds::Vector{SVector{n,Int}}
    uinds::Vector{SVector{m,Int}}
    times::Vector{Float64}
    function NLP{Q}(model::AbstractModel, obj::Objective, x0::Vector, tf::Real, xf=zero(x0)*NaN) where Q
        n,m = size(model)
        T = length(obj)
        xinds = [SVector{n}((k-1)*(n+m) .+ (1:n)) for k = 1:T]
        uinds = [SVector{m}((k-1)*(n+m) .+ (n+1:n+m)) for k = 1:T-1]
        times = collect(range(0, tf, length=T))
        new{n,m,Q,typeof(model),typeof(obj[1])}(
            model, obj, T, tf, x0, xf, xinds, uinds, times
        )
    end
end
Base.size(nlp::NLP{n,m}) where {n,m} = (n,m,nlp.T)
num_primals(nlp::NLP{n,m}) where {n,m} = n*nlp.T + m*(nlp.T-1)
num_duals(nlp::NLP) = num_eq(nlp) + num_ineq(nlp)
num_eq(nlp::NLP{n,m}) where {n,m} = n*nlp.T + n*termcon(nlp)
num_ineq(nlp::NLP) = 0
@inline num_vars(nlp::NLP) = num_primals(nlp)
termcon(nlp::NLP) = all(isfinite, nlp.xf)

function NLP(prob::Problem{Q}, termcon::Bool=false) where Q
    xf = termcon ? prob.xf : zero(prob.x0)*NaN
    NLP{Q}(prob.model, prob.obj, Vector(prob.x0), prob.tf, xf)
end

function eval_f(nlp::NLP, Z)
    J = 0.0
    xi,ui = nlp.xinds, nlp.uinds
    for k = 1:nlp.T-1
        x,u = Z[xi[k]], Z[ui[k]]
        dt = nlp.times[k+1] - nlp.times[k]
        J += TO.stage_cost(nlp.obj[k], x, u) * dt
    end
    J += TO.stage_cost(nlp.obj[end], Z[xi[end]])
    return J
end

function grad_f!(nlp::NLP{n,m,<:Any,<:Any,<:TO.QuadraticCostFunction}, grad, Z) where {n,m}
    xi,ui = nlp.xinds, nlp.uinds
    obj = nlp.obj
    for k = 1:nlp.T-1
        x,u = Z[xi[k]], Z[ui[k]]
        dt = nlp.times[k+1] - nlp.times[k]
        grad[xi[k]] = (obj[k].Q*x + obj[k].q)*dt
        grad[ui[k]] = (obj[k].R*u + obj[k].r)*dt
    end
    grad[xi[end]] = obj[end].Q*Z[xi[end]] + obj[end].q
    return nothing
end

function dgrad_f(nlp::NLP{n,m,<:Any,<:Any,<:TO.QuadraticCostFunction}, Z, dZ) where {n,m}
    xi,ui = nlp.xinds, nlp.uinds
    obj = nlp.obj
    out = zero(eltype(Z))
    for k = 1:nlp.T-1
        x,u = Z[xi[k]], Z[ui[k]]
        dx,du = dZ[xi[k]], dZ[ui[k]]
        dt = nlp.times[k+1] - nlp.times[k]
        out += dot(x, obj[k].Q, dx)*dt + dot(obj[k].q, dx)*dt
        out += dot(u, obj[k].R, du)*dt + dot(obj[k].r, du)*dt 
    end
    x, dx = Z[xi[end]], dZ[xi[end]]
    out += dot(x, obj[end].Q, dx) + dot(obj[end].q, dx)
    return out
end

function hess_f!(nlp::NLP{n,m,<:Any,<:Any,<:TO.DiagonalCost}, hess, Z, rezero=true) where {n,m}
    if rezero
        for i = 1:size(hess,1)
            hess[i,i] = 0
        end
    end
    xi,ui = nlp.xinds, nlp.uinds
    obj = nlp.obj
    i = 1
    for k = 1:nlp.T
        dt = k < nlp.T ? nlp.times[k+1] - nlp.times[k] : 1.0
        for j = 1:n
            hess[i,i] += nlp.obj[k].Q[j,j] * dt
            i += 1
        end
        if k < nlp.T
            for j = 1:m
                hess[i,i] += nlp.obj[k].R[j,j] * dt
                i += 1
            end
        end
    end
end

function hvp_f!(nlp::NLP{n,m,<:Any,<:Any,<:TO.DiagonalCost}, hvp::AbstractVector{T}, Z, rezero=true) where {n,m,T}
    if rezero
        for i = 1:length(hvp)
            hvp[i] = zero(T) 
        end
    end
    xi,ui = nlp.xinds, nlp.uinds
    obj = nlp.obj
    i = 1
    for k = 1:nlp.T
        dt = k < nlp.T ? nlp.times[k+1] - nlp.times[k] : 1.0
        for j = 1:n
            hvp[i] += nlp.obj[k].Q[j,j] * dt * Z[i]
            i += 1
        end
        if k < nlp.T
            for j = 1:m
                hvp[i] += nlp.obj[k].R[j,j] * dt * Z[i]
                i += 1
            end
        end
    end
end

############################################################################################
#                                 LAGRANGIAN
############################################################################################
function lagrangian(nlp::NLP{n,m}, Z, λ, c=zeros(eltype(Z),length(λ))) where {n,m}
    J = eval_f(nlp, Z)
    eval_dynamics_constraints!(nlp, c, Z)
    return J - dot(λ,c)
end

function grad_lagrangian!(nlp::NLP{n,m}, grad, Z, λ, tmp=zeros(eltype(Z), n+m)) where {n,m}
    grad_f!(nlp, grad, Z)
    grad .*= -1
    jacvec_dynamics!(nlp, grad, Z, λ, false, tmp)
    grad .*= -1
    return nothing
end

function hess_lagrangian!(nlp::NLP{n,m}, hess, Z, λ) where {n,m}
    ∇jacvec_dynamics!(nlp, hess, Z, λ)
    hess .*= -1
    hess_f!(nlp, hess, Z, false)
end

function primal_residual(nlp::NLP, Z, λ, g=zeros(num_primals(nlp)); p=2)
    grad_lagrangian!(nlp, g, Z, λ)
    return norm(g, p)
end

############################################################################################
#                            AUGMENTED LAGRANGIAN
############################################################################################
function aug_lagrangian(nlp::NLP, Z, λ, ρ, c=zeros(eltype(Z), length(λ)))
    J = lagrangian(nlp, Z, λ, c)
    return J + 1//2 * ρ * dot(c,c)
end

function algrad!(nlp::NLP, grad, Z, λ, ρ, 
        c=zeros(eltype(Z), length(λ)), 
        tmp=zeros(eltype(Z), sum(size(nlp)[1:2]))
    )
    # Gradient of the Lagrangian
    grad_lagrangian!(nlp, grad, Z, λ, tmp)
    grad ./= ρ

    # Add Gradient of penalty
    eval_c!(nlp, c, Z)
    jacvec_dynamics!(nlp, grad, Z, c, false, tmp) 
    grad .*= ρ
    return nothing
end

function al_dgrad(nlp::NLP, Z, dZ, λ, ρ,
        grad=zeros(eltype(Z), length(Z)),
        c=zeros(eltype(Z), length(λ)), 
        tmp=zeros(eltype(Z), sum(size(nlp)[1:2]))
    )
    dphi = dgrad_f(nlp, Z, dZ)

    eval_c!(nlp, c, Z)
    λbar = c
    λbar .*= ρ
    λbar .-= λ
    jacvec_dynamics!(nlp, grad, Z, λbar, true, tmp)
    dphi += dot(grad, dZ)
end

function alhess!(nlp::NLP, hess, Z, λ, ρ, gn::Bool=true, 
        c=zeros(eltype(Z), length(λ)),
        jac=zeros(eltype(Z), length(λ), length(Z))
    )
    hess .= 0
    if gn
        hess .= 0
    else
        # Add 2nd-Order dynanics derivatives to Hessian
        eval_c!(nlp, c, Z)
        λbar = λ - ρ*c
        ∇jacvec_dynamics!(nlp, hess, Z, λbar)
        hess .*= -1
    end
    hess_f!(nlp, hess, Z, false)

    # Add constraint penalty
    jac_c!(nlp, jac, Z)
    hess .+= ρ*jac'jac
    # mul!(hess, jac', jac, ρ, 1.0)  # avoids allocs but WAY slower
end

function pdal_sys!(nlp::NLP, hess, grad, Z, λtilde, λ, ρ, gn::Bool=true,
        c=zeros(eltype(Z), length(λ)),
        ∇c=zeros(eltype(Z), length(λ), length(Z))
    )
    N,M = num_primals(nlp), num_duals(nlp)
    iP,iD = 1:N, N .+ (1:M)

    # Hessian
    hess1 = view(hess, iP, iP) 
    hess2 = view(hess, iP, iD)
    hess3 = view(hess, iD, iP)
    hess4 = view(hess, iD, iD) 

    eval_c!(nlp, c, Z)
    jac_c!(nlp, ∇c, Z) 
    if gn
        hess_f!(nlp, hess1, Z)
    else
        hess_lagrangian!(nlp, hess1, Z, λtilde)
    end
    jac_c!(nlp, hess3, Z)
    hess3 .*= -1

    iρ = inv(ρ)
    for i in iD 
        hess[i,i] = -iρ
    end

    # Gradient
    grad1 = view(grad, iP)
    grad2 = view(grad, iD)
    grad_lagrangian!(nlp, grad1, Z, λtilde)
    
    grad2 .= (λ .- λtilde) .* iρ .- c
    return nothing
end

function dual_update!(nlp::NLP, Z, λ, ρ, c=zeros(eltype(Z), length(λ)))
    eval_c!(nlp, c, Z)
    λ .- ρ .* c
end

############################################################################################
#                                CONSTRAINTS
############################################################################################
eval_c!(nlp::NLP, c, Z) = eval_dynamics_constraints!(nlp, c, Z)
jac_c!(nlp::NLP, jac, Z) = jac_dynamics!(nlp, jac, Z)

function dual_residual(nlp::NLP, Z, λ, c=zeros(num_eq(nlp)); p=2)
    eval_c!(nlp, c, Z)
    norm(c, p)
end

function eval_dynamics_constraints!(nlp::NLP{n_,m_,Q}, c, Z) where {n_,m_,Q}
    n,m,T = size(nlp)
    xi,ui = nlp.xinds, nlp.uinds
    idx = xi[1]

    # initial condition
    c[idx] = Z[xi[1]] - nlp.x0

    # dynamics
    for k = 1:T-1
        idx = idx .+ n
        x,u = Z[xi[k]], Z[ui[k]]
        x⁺ = Z[xi[k+1]]
        dt = nlp.times[k+1] - nlp.times[k]
        c[idx] = discrete_dynamics(Q, nlp.model, x, u, nlp.times[k], dt) - x⁺
    end

    # terminal constraint
    if termcon(nlp)
        idx = idx .+ n
        c[idx] = Z[xi[T]] - nlp.xf
    end
    return nothing
end

function jac_dynamics!(nlp::NLP{n,m,Q}, jac, Z) where {n,m,Q}
    for i = 1:n
        jac[i,i] = 1
    end

    xi,ui = nlp.xinds, nlp.uinds
    idx = xi[1]
    for k = 1:nlp.T-1
        idx = idx .+ n 
        zi = [xi[k];ui[k]]
        dt = nlp.times[k+1] - nlp.times[k]
        z = StaticKnotPoint(Z[zi], xi[1], ui[1], dt, nlp.times[k])
        J = view(jac,idx,zi)
        discrete_jacobian!(Q, J, nlp.model, z, nothing)
        for i = 1:n
            jac[idx[i], zi[end]+i] = -1
        end
    end
    if termcon(nlp)
        idx = idx .+ n 
        for i = 1:n
            jac[idx[i], xi[end][i]] = 1
        end
    end
end

function jac_dynamics!(nlp::NLP{n,m,Q}, jac::AbstractVector, Z) where {n,m,Q}
    cnt = 1
    for i = 1:n
        jac[cnt] = 1
        cnt += 1
    end

    xi,ui = nlp.xinds, nlp.uinds
    idx = xi[1]
    nblk = n * (n+m)
    for k = 1:nlp.T-1
        idx = idx .+ n 
        zi = [xi[k];ui[k]]
        dt = nlp.times[k+1] - nlp.times[k]
        z = StaticKnotPoint(Z[zi], xi[1], ui[1], dt, nlp.times[k])
        # J = view(jac,idx,zi)
        Jvec = view(jac,cnt:cnt+nblk-1)
        J = reshape(Jvec, n, n+m)
        discrete_jacobian!(Q, J, nlp.model, z, nothing)
        cnt += nblk
        for i = 1:n
            jac[cnt] = -1
            cnt += 1
        end
    end
    if termcon(nlp)
        for i = 1:n
            jac[cnt+i] = 1
        end
    end
end

function jacvec_dynamics!(nlp::NLP{n,m,Q}, jac, Z, λ, rezero::Bool=true, tmp=zeros(n+m)) where {n,m,Q}
    for i = 1:n
        rezero && (jac[i] = 0)
        jac[i] += λ[i]
    end

    xi,ui = nlp.xinds, nlp.uinds
    idx = [xi[1]; ui[1]]
    idx2 = xi[1]
    for k = 1:nlp.T-1
        idx2 = idx2 .+ n
        zi = [xi[k];ui[k]]
        dt = nlp.times[k+1] - nlp.times[k]
        z = StaticKnotPoint(Z[zi], xi[1], ui[1], dt, nlp.times[k])
        λ_ = λ[idx2]
        RD.discrete_jvp!(Q, tmp, nlp.model, z, λ_)
        rezero && (jac[idx[end]] = 0)
        jac[idx] += tmp 

        idx = idx .+ (n + m)
        for i = 1:n
            rezero && (jac[idx[i]] = 0)
            jac[idx[i]] += -λ_[i]
        end
    end
    if termcon(nlp)
        λT = λ[idx2 .+ n]
        for i = 1:n
            jac[idx[i]] += λT[i]
        end
    end
end

function ∇jacvec_dynamics!(nlp::NLP{n,m,Q}, hess, Z, λ) where {n,m,Q}
    xi,ui = nlp.xinds, nlp.uinds
    idx = [xi[1]; ui[1]]
    idx2 = xi[1]
    for k = 1:nlp.T-1
        idx2 = idx2 .+ n
        zi = [xi[k];ui[k]]
        dt = nlp.times[k+1] - nlp.times[k]
        z = StaticKnotPoint(Z[zi], xi[1], ui[1], dt, nlp.times[k])
        λ_ = λ[idx2]
        ∇f = view(hess, idx, idx)
        RD.∇discrete_jacobian!(Q, ∇f, nlp.model, z, λ_)
        idx = idx .+ (n + m)
    end
    for i = 1:n
        hess[end-i+1,end-i+1] = 0
    end
end

############################################################################################
#                                  OTHER METHODS
############################################################################################
function Vector(Z::RD.AbstractTrajectory{n,m,elT}) where {n,m,elT}
    ix = SVector{n}(1:n)
    iu = SVector{m}(n .+ (1:m))
    T = length(Z)
    N = RD.num_vars(Z)  
    Z_ = zeros(elT,N)
    for k = 1:T
        Z_[ix] .= state(Z[k])
        if k < T || !RD.is_terminal(Z[T])
            Z_[iu] .= control(Z[k])
        end
        ix = ix .+ (n+m)
        iu = iu .+ (n+m)
    end
    return Z_
end

function TO.get_trajectory(nlp::NLP{n,m}, Z) where {n,m}
    ix,iu = nlp.xinds, nlp.uinds
    Z_ = map(1:nlp.T-1) do k 
        x,u = Z[ix[k]], Z[iu[k]]
        dt = nlp.times[k+1] - nlp.times[k]
        KnotPoint(x, u, dt, nlp.times[k])
    end
    # zterm = KnotPoint(Z[ix[end]], m, nlp.times[end])
    push!(Z_, KnotPoint(Z[ix[end]], m, nlp.times[end]))
    Traj(Z_)
end

function find_reg(A; step=0.05, iters=10)::Float64
    A = Symmetric(A)
    reg = 0.0
    F = cholesky(A, check=false)
    for i = 1:iters
        if issuccess(F)
            break
        else
            reg += step
            cholesky!(F, A, shift=reg, check=false)
        end
        i == 10 && (reg = -1.0)
    end
    return reg
end