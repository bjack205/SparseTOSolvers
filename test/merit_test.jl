
## Generate Problem
prob = Cartpole()
nlp = NLP(prob)

# Initialize solution
Z = Vector(prob.Z)
λ = zeros(num_eq(nlp))

# Build QP
qp = TOQP(size(nlp)..., num_eq(nlp), 0)
build_qp!(qp, nlp, Z, λ)

# Solve the QP
osqp = OSQP.Model()
dZ, λhat = solve_qp!(osqp, qp)
dλ = λhat - λ

## Line Search
ϕ = NormPenalty(10.0, 1, num_primals(nlp), num_eq(nlp))
ϕ(nlp,Z)
grad(ϕ, nlp, Z)
g0 = ForwardDiff.gradient(x->ϕ(nlp,x,zeros(eltype(x),num_eq(nlp))), Z)
@test g0'dZ ≈ grad(ϕ, nlp, Z)'dZ
@test qp.q'dZ - ϕ.ρ*norm(qp.b) ≈ g0'dZ
@test dgrad(ϕ, nlp, Z, dZ) ≈ grad(ϕ, nlp, Z)'dZ

@test (@allocated grad(ϕ, nlp, Z)) == 0
# @test (@allocated dgrad(ϕ, nlp, Z, dZ)) == 0