using RobotDynamics
using RobotZoo
using LinearAlgebra, StaticArrays

function DoubleIntegrator()
    model = RobotZoo.DoubleIntegrator()
    n,m = size(model)

    # Task
    x0 = @SVector [0., 0.]
    xf = @SVector [1., 0]
    tf = 2.0

    # Discretization info
    N = 21
    dt = tf/(N-1)

    # Costs
    Q = 1.0*Diagonal(@SVector ones(n))
    Qf = 1.0*Diagonal(@SVector ones(n))
    R = 1.0e-1*Diagonal(@SVector ones(m))
    obj = LQRObjective(Q,R,Qf,xf,N)

    prob = Problem(model, obj, xf, tf, x0=x0, N=N)
    TO.rollout!(prob)
    return prob
end

function Cartpole(N=101)
    model = RobotZoo.Cartpole()
    n,m = size(model)
    tf = 5.
    dt = tf/(N-1)

    Q = 1.0e-3*Diagonal(@SVector ones(n))
    Qf = 10000.0*Diagonal(@SVector ones(n))
    R = 1.0e-3*Diagonal(@SVector ones(m))
    x0 = @SVector zeros(n)
    xf = @SVector [0, pi, 0, 0]
    obj = LQRObjective(Q,R,Qf,xf,N)

    conSet = ConstraintList(n,m,N)

    X0 = [@SVector fill(NaN,n) for k = 1:N]
    u0 = @SVector fill(0.01,m)
    U0 = [u0 for k = 1:N-1]
    Z = Traj(X0,U0,dt*ones(N))
    prob = Problem{RK3}(model, obj, conSet, x0, xf, Z, N, 0.0, tf)
    rollout!(prob)

    return prob
end

Base.@kwdef struct Cartpole2 <: AbstractModel 
    m1::Float64 = 2.0   # cart mass
    m2::Float64 = 0.5   # pole mass
    g::Float64 = 9.81   # gravity
    l::Float64 = 0.5    # pole length
end
RD.state_dim(::Cartpole2) = 4
RD.control_dim(::Cartpole2) = 1

function RD.dynamics(model::Cartpole2, x, u)
    m1,m2 = model.m1, model.m2
    g,l = model.g, model.l
    p = x[1]
    q = x[2]
    dp = x[3]
    dq = x[4]
    t2 = cos(q);
    t3 = sin(q);
    t4 = t2.^2;
    t5 = m1+m2- m2*t4;
    t6 = 1.0/t5;
    t7 = dq.^2;
    ddp = t6*(u[1]+g*m2*t2*t3 + l*m2*t3*t7);
    ddq = -(t6*(t2*u[1]+g*m1*t3+g*m2*t3+l*m2.*t2*t3*t7))/l;
    return SA[dp,dq,ddp,ddq]
end

function CartpoleProb(N=101, Qd=10.0; mc=2.0, mp=0.5, l=0.5)
    # model = Cartpole2()
    model = RobotZoo.Cartpole(mc, mp, l, 9.81)
    n,m = size(model)
    tf = 2.
    dt = tf/(N-1)

    Q = 1.0e-3*Diagonal(@SVector ones(n))
    Qf = Qd*Diagonal(@SVector ones(n))
    R = 1.0e-0*Diagonal(@SVector ones(m))
    x0 = @SVector zeros(n)
    xf = @SVector [0.8, pi, 0, 0]
    obj = LQRObjective(Q,R,Qf,xf,N)

    conSet = ConstraintList(n,m,N)

    X0 = [@SVector fill(NaN,n) for k = 1:N]
    u0 = @SVector fill(0.01,m)
    U0 = [u0 for k = 1:N-1]
    Z = Traj(X0,U0,dt*ones(N))
    prob = Problem{RK3}(model, obj, conSet, x0, xf, Z, N, 0.0, tf)
    rollout!(prob)

    return prob
end