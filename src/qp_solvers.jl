using OSQP
using SparseArrays
using LinearAlgebra
using QDLDL

struct TOQP
    Q::SparseMatrixCSC{Float64,Int}  # quadratic cost
    q::Vector{Float64}               # linear cost
    A::SparseMatrixCSC{Float64,Int}  # equality constraint Ax = b
    b::Vector{Float64}               # equality constraint 
    C::SparseMatrixCSC{Float64,Int}  # inequality constraint l ≤ Cx ≤ u
    l::Vector{Float64}               # inequality constraint lower bound
    u::Vector{Float64}               # inequality constraint upper bound
    n::Int
    m::Int
    T::Int

    # Quasi-Newton storage
    B::Symmetric{Float64, Matrix{Float64}}  # Hessian approximation
    x::Vector{Float64}  # previous x 
    g::Vector{Float64}  # previous gradient
    s::Vector{Float64}  # x⁺ - x⁻
    y::Vector{Float64}  # g⁺ - g⁻
    r::Vector{Float64}  # damped step

    opts::Dict{Symbol,Symbol}

    function TOQP(n,m,T,M,P; hess=:gauss_newton)
        N = n*T + (T-1)*m
        Q = spzeros(N,N)
        q = zeros(N)
        A = spzeros(M,N)
        b = zeros(M) 
        C = spzeros(P,N)
        l = fill(-Inf,P)
        u = fill(Inf,P)

        # BFGS
        B = Symmetric(zeros(N,N))
        x = zeros(N)
        g = zeros(N)
        s = zeros(N)
        y = zeros(N)
        r = zeros(N)

        # Options
        opts = Dict(
            :hess => hess,
            :structure => :unknown,
            :keep_pd => :hess
        )
        new(Q,q,A,b,C,l,u,n,m,T,
            B,x,g,s,y,r, 
            opts
        )
    end
end


function TOQP(nlp::NLP{n,m}; kwargs...) where {n,m}
    TOQP(n,m,nlp.T,num_eq(nlp), num_ineq(nlp); kwargs...)
end

num_ineq(qp::TOQP) = length(qp.l)
num_eq(qp::TOQP) = length(qp.b)
num_primals(qp::TOQP) = length(qp.q)
num_duals(qp::TOQP) = num_ineq(qp) + num_eq(qp)


"""
Build a QP from the NLP, optionally using either the Hessian of the cost function 
or the Hessian of the Lagrangian.
"""
function build_qp!(qp::TOQP, nlp::NLP, Z, λ, hess = qp.opts[:hess])
    jac_dynamics!(nlp, qp.A, Z)
    eval_dynamics_constraints!(nlp, qp.b, Z)
    qp.b .*= -1  # reverse sign
    grad_lagrangian!(nlp, qp.q, Z, λ)

    if hess == :gauss_newton 
        hess_f!(nlp, qp.Q, Z)
        if qp.opts[:structure] == :unknown && isdiag(qp.Q)
            qp.opts[:structure] = :diag
        end
    elseif hess == :newton
        hess_lagrangian!(nlp, qp.Q, Z, λ)
    elseif hess == :bfgs
        bfgs!(qp, Z, qp.q)
    end
    return nothing
end

function invertQ!(qp::TOQP)
    if qp.opts[:hess] == :bfgs
        Q = qp.B
        F = cholesky(Q) 
        Qq = F \ qp.q
        QA = F \ Matrix(qp.A')  # necessary since B is dense
    elseif qp.opts[:structure] == :diag
        F = qp.Q
        Qq = F \ qp.q
        QA = F \ qp.A'
    else
        F = qdldl(qp.Q)
        if qp.opts[:keep_pd] ∈ (:hess,)
            for i = 1:size(qp.Q,1)
                if F.Dinv[i] < 0
                    F.Dinv[i] = 1e-10
                end
            end
        end
        Qq = F \ qp.q
        QA = F \ qp.A'  # qdldl is slow for this TODO: speed this up
    end
    return Qq, QA
end

function gethess(qp::TOQP)
    qp.opts[:hess] == :bfgs ? qp.B : qp.Q
end

"""
Damped BFGS
"""
function bfgs!(qp::TOQP, x, g)
    B = qp.B
    s,y,r = qp.s, qp.y, qp.r

    # Calculate differences
    s .= x .- qp.x
    y .= g .- qp.g

    # Stash previous values
    qp.x .= x
    qp.g .= g

    # Calculate BFGS update
    d1 = dot(s,y)
    d2 = dot(s, B, s)
    # d2 ≈ 0 && return  # skip if gradient hasn't changed
    if d2 ≈ 0 
        # println("\nSkipping BFGS update")
        return
    end
    Bs = B*s
    println("Secant condition: ", d1)
    if d1 ≥ 0.2*d2 
        r .= y
    else
        θ = 0.8*d2/(d2-d1)
        r .= θ .* y .+ (1-θ) .* Bs
    end
    B.data .= B - (Bs*Bs')/(d2) + r*r' / dot(s,r)

    return nothing
end

function OSQP.Model(qp::TOQP; kwargs...)
    model = OSQP.Model()
    OSQP.setup!(model, P=qp.Q, q=qp.q, A=qp.A, l=qp.b, u=qp.b; kwargs...)
    return model
end

function solve_qp!(solver::OSQP.Model, qp::TOQP; setup=true, kwargs...)
    if num_ineq(qp) == 0
        setup && OSQP.setup!(solver, P=qp.Q, q=qp.q, A=qp.A, l=qp.b, u=qp.b; kwargs...)
    else
        A = [qp.A; qp.C]
        u = [qp.b; qp.u]
        l = [qp.b; qp.l]
        OSQP.setup!(solver, P=qp.Q, q=qp.q, A=A, l=l, u=u; kwargs...)
    end
    res = OSQP.solve!(solver)
    return res.x, res.y
end

struct ShurSolver end
ShurSolver(qp::TOQP; kwargs...) = ShurSolver() 
function solve_qp!(solver::ShurSolver, qp::TOQP; kwargs...)
    Q,q = qp.Q, qp.q
    A,b = qp.A, qp.b
    Qq,QA = invertQ!(qp)
    AQA = Symmetric(A*QA)
    if qp.opts[:keep_pd] ∈ (:hess,) || qp.opts[:hess] == :bfgs || qp.opts[:structure] == :diag
        S = cholesky(AQA)
    else
        S = ldlt(AQA)
    end
    s = A*(Qq) + b
    dλ = -(S\s)
    dZ = -(QA*dλ + Qq)
    return dZ, dλ
end

struct KKTSolver
    K::Symmetric{Float64,SparseMatrixCSC{Float64,Int}}
    t::Vector{Float64}
    Q::SubArray{Float64,2,SparseMatrixCSC{Float64,Int},Tuple{UnitRange{Int},UnitRange{Int}}, false}
    A::SubArray{Float64,2,SparseMatrixCSC{Float64,Int},Tuple{UnitRange{Int},UnitRange{Int}}, false}
    q::SubArray{Float64,1,Vector{Float64},Tuple{UnitRange{Int}},true}
    b::SubArray{Float64,1,Vector{Float64},Tuple{UnitRange{Int}},true}
    function KKTSolver(n::Int, m::Int)
        K = spzeros(n+m,n+m)
        t = zeros(n+m)
        Q = view(K,1:n,1:n)
        A = view(K,1:n,n+1:n+m)
        q = view(t, 1:n)
        b = view(t, n+1:n+m)
        new(Symmetric(K),t,Q,A,q,b)
    end
end

KKTSolver(qp::TOQP) = KKTSolver(num_primals(qp), num_duals(qp))

function solve_qp!(solver::KKTSolver, qp::TOQP; kwargs...)
    N,M = num_primals(qp), num_duals(qp)
    Q = gethess(qp)
    K = [Q qp.A'; qp.A spzeros(M,M)]
    t = [-qp.q; qp.b]
    dY = K \ t
    dZ = dY[1:N]
    dλ = dY[N+1:N+M]
    return dZ, dλ
end

struct QDLDLSolver 
    method::Symbol
end
QDLDLSolver(qp::TOQP; method=:kkt) = QDLDLSolver(method)

function solve_qp!(solver::QDLDLSolver, qp::TOQP; kwargs...)
    N,M = num_primals(qp), num_duals(qp)
    if solver.method == :kkt
        K = [qp.Q qp.A'; qp.A I*1e-12]
        F = qdldl(K)
        t = [-qp.q; qp.b]
        dY = F \ t
        dZ = dY[1:N]
        dλ = dY[N+1:N+M]
    else
        Q,q = qp.Q, qp.q
        A,b = qp.A, qp.b
        Qq,QA = invertQ!(qp)
        S = qdldl(A*sparse(QA))
        s = A*(Qq) + b
        dλ = -(S\s)
        dZ = -(QA*dλ + Qq)
            
    end
    return dZ, dλ
end
