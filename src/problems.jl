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
