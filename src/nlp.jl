using TrajectoryOptimization
using RobotDynamics
using StaticArrays
using RobotZoo
const TO = TrajectoryOptimization
const RD = RobotDynamics

import TrajectoryOptimization: num_vars

struct NLP{n,m,Q,L,C}
    model::L
    obj::Objective{C}
    T::Int  # number of knot points
    tf::Float64
    x0::Vector{Float64}  # initial condition
    xinds::Vector{SVector{n,Int}}
    uinds::Vector{SVector{m,Int}}
    times::Vector{Float64}
    function NLP{Q}(model::AbstractModel, obj::Objective, x0::Vector, tf::Real) where Q
        n,m = size(model)
        T = length(obj)
        @show T
        xinds = [SVector{n}((k-1)*(n+m) .+ (1:n)) for k = 1:T]
        uinds = [SVector{m}((k-1)*(n+m) .+ (n+1:n+m)) for k = 1:T-1]
        times = collect(range(0, tf, length=T))
        new{n,m,Q,typeof(model),typeof(obj[1])}(
            model, obj, T, tf, x0, xinds, uinds, times
        )
    end
end
Base.size(nlp::NLP{n,m}) where {n,m} = (n,m,nlp.T)
num_vars(nlp::NLP{n,m}) where {n,m} = n*nlp.T + m*(nlp.T-1)
num_cons(nlp::NLP{n,m}) where {n,m} = n*nlp.T
function NLP(prob::Problem{Q}) where Q
    NLP{Q}(prob.model, prob.obj, Vector(prob.x0), prob.tf)
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

function lagrangian(nlp::NLP{n,m}, Z, λ, c=zeros(eltype(Z),length(λ))) where {n,m}
    J = eval_f(nlp, Z)
    eval_dynamics_constraints!(nlp, c, Z)
    return J + dot(λ,c)
end

function grad_lagrangian!(nlp::NLP{n,m}, grad, Z, λ) where {n,m}
    grad_f!(nlp, grad, Z)
    jacvec_dynamics!(nlp, grad, Z, λ, false)
    return nothing
end

function hess_lagrangian!(nlp::NLP{n,m}, hess, Z, λ)
    ∇jacvec_dynamics!(nlp, hess, Z, λ)
    hess_f!(nlp, hess, Z, false)
end

function aug_lagrangian(nlp::NLP, Z, λ, μ, c=zero(λ))
    J = lagrangian(nlp, Z, λ)
    return J + 1//2 * μ * dot(c,c)
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
end

function jacvec_dynamics!(nlp::NLP{n,m,Q}, jac, Z, λ, rezero::Bool=true) where {n,m,Q}
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
        grad = RD.discrete_jacvec(Q, nlp.model, z, λ_)
        rezero && (jac[idx[end]] = 0)
        jac[idx] += grad

        idx = idx .+ (n + m)
        for i = 1:n
            rezero && (jac[idx[i]] = 0)
            jac[idx[i]] += -λ_[i]
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