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
end

function TOQP(n,m,T,M,P)
    N = n*T + (T-1)*m
    Q = spzeros(N,N)
    q = zeros(N)
    A = spzeros(M,N)
    b = zeros(M) 
    C = spzeros(P,N)
    l = fill(-Inf,P)
    u = fill(Inf,P)
    TOQP(Q,q,A,b,C,l,u,n,m,T)
end

function TOQP(nlp::NLP{n,m}) where {n,m}
    TOQP(n,m,nlp.T,num_eq(nlp), num_ineq(nlp))
end

num_ineq(qp::TOQP) = length(qp.l)
num_eq(qp::TOQP) = length(qp.b)
num_primals(qp::TOQP) = length(qp.q)
num_duals(qp::TOQP) = num_ineq(qp) + num_eq(qp)


"""
Build a QP from the NLP, optionally using either the Hessian of the cost function 
or the Hessian of the Lagrangian.
"""
function build_qp!(qp::TOQP, nlp::NLP, Z, λ, gn::Bool=false)
    jac_dynamics!(nlp, qp.A, Z)
    eval_dynamics_constraints!(nlp, qp.b, Z)
    qp.b .*= -1  # reverse sign
    grad_lagrangian!(nlp, qp.q, Z, λ)

    if gn
        hess_f!(nlp, qp.Q, Z)
    else
        hess_lagrangian!(nlp, qp.Q, Z, λ)
    end
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
function solve_qp!(solver::ShurSolver, qp::TOQP; gauss_newton::Bool=false, kwargs...)
    Q,q = qp.Q, qp.q
    A,b = qp.A, qp.b
    ispd = gauss_newton
    if isdiag(Q)
        QA = Q\A'
        Qq = Q\q
        ispd = true  # use cholesky if Q is diagonal
    else
        F = ldlt(Symmetric(Q))
        QA = F \ sparse(A')  # need to make it a new sparse matrix
        Qq = F \ q
    end
    AQA = Symmetric(A*QA)
    if ispd
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
    K = [qp.Q qp.A'; qp.A spzeros(M,M)]
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
    if solver.method == :kkt
        N,M = num_primals(qp), num_duals(qp)
        K = [qp.Q qp.A'; qp.A I*1e-12]
        F = qdldl(K)
        t = [-qp.q; qp.b]
        dY = F \ t
        dZ = dY[1:N]
        dλ = dY[N+1:N+M]
    else
        Q,q = qp.Q, qp.q
        A,b = qp.A, qp.b
        Q = qdldl(Q)
        QA = Q\A'
        Qq = Q\q
        S = qdldl(A*QA)
        s = A*(Qq) + b
        dλ = -(S\s)
        dZ = -(QA*dλ + Qq)
    end
    return dZ, dλ
end
