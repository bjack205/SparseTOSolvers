using TrajectoryOptimization
using LinearAlgebra, StaticArrays, SparseArrays
using RobotDynamics
using RobotZoo
const TO = TrajectoryOptimization
const RD = RobotDynamics
include("problems.jl")

##
struct ViewExpansion{T}
    hess::SubArray{T,2,SparseMatrixCSC{T,Int64},Tuple{UnitRange{Int64},UnitRange{Int64}},false}
    grad::SubArray{T,1,Vector{T},Tuple{UnitRange{Int}},true}
    x::SubArray{T,1,Vector{T},Tuple{UnitRange{Int}},true}
    u::SubArray{T,1,Vector{T},Tuple{UnitRange{Int}},true}
    xx::SubArray{T,2,SparseMatrixCSC{T,Int},Tuple{UnitRange{Int},UnitRange{Int}},false}
    uu::SubArray{T,2,SparseMatrixCSC{T,Int},Tuple{UnitRange{Int},UnitRange{Int}},false}
    ux::SubArray{T,2,SparseMatrixCSC{T,Int},Tuple{UnitRange{Int},UnitRange{Int}},false}
    function ViewExpansion{T}(n,m,hess::SubArray, grad::SubArray) where T 
        ix,iu = 1:n, n .+ (1:m)
        x = view(grad,ix)
        u = view(grad,iu)
        xx = view(hess,ix,ix)
        uu = view(hess,iu,iu)
		ux = view(hess,iu,ix)
		hess .= I(n+m)
		grad .= 0 
        new{T}(hess, grad, x, u, xx, uu, ux)
    end
end
function ViewExpansion(n::Int, m::Int, H::SparseMatrixCSC{T}, g::Vector, k::Integer) where {T}
    NN = length(g)
    N = (NN - n) ÷ (n+m)
    if k == N
        iz = NN-n+1:NN
    else
        iz = 1:(n+m) .+ (n+m)*(k-1)
    end
    hess = view(H,iz,iz)
    grad = view(g,iz)
    ViewExpansion{T}(n, m * (k<N), hess, grad)
end

function Base.getproperty(E::ViewExpansion, field::Symbol)
    if field == :q
        getfield(E, :x)
    elseif field == :r
        getfield(E, :u)
    elseif field == :Q
        getfield(E, :xx)
    elseif field == :R
        getfield(E, :uu)
    elseif field == :H
        getfield(E, :ux)
    else
        getfield(E, field)
    end
end

struct SProblem{n,m,Q,L,C}
    model::L
    obj::Objective{C}
    dt::Float64
    N::Int
    x0::Vector{Float64}
    E::Vector{ViewExpansion{Float64}}
    H::SparseMatrixCSC{Float64,Int}
    g::Vector{Float64}
    D::SparseMatrixCSC{Float64,Int}
    d::Vector{Float64}
    Z::Vector{Float64}
    λ::Vector{Float64}
    xinds::Vector{SVector{n,Float64}}
    uinds::Vector{SVector{m,Float64}}
end
function SProblem(model::L, obj0::Objective, x0, dt, N::Int, integration=RK4) where L
    n,m = size(model)
    @assert length(x0) == n

    NN = N*n + (N-1)*m
    P = N*n  

    H = spzeros(NN,NN)
    g = zeros(NN)

    D = spzeros(P,NN)
    d = zeros(P)

    Z = zeros(NN)
    λ = zeros(P)
    
    xinds = [SVector{n}((1:n) .+ (n+m)*(k-1)) for k = 1:N]
    xinds = [SVector{n}((1:m) .+ n .+ (n+m)*(k-1)) for k = 1:N-1]
    SProblem{n,m,integration,L}(model, obj, dt, N, x0, H, g, D, d, Z, λ, xinds, uinds)
end
function SProblem(prob::Problem)
    SProblem(prob.model, prob.obj, prob.x0, prob.Z[1].dt, prob.N, TO.integration(prob))
end

prob = DoubleIntegrator()
NN = TO.num_vars(prob.Z)
H = spzeros(NN,NN)
g = zeros(NN)
ViewExpansion(size(prob)[1:2]..., H,g,1)