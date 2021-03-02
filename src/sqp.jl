LinearAlgebra.isdiag(fact::QDLDL.QDLDLFactorisation) = nnz(fact.L) == 0
function QDLDL.solve(fact::QDLDL.QDLDLFactorisation, B::AbstractMatrix)
    if isdiag(fact)
        X = fact.Dinv * B
    else
        X = copy(B) 
        QDLDL.solve!(fact, X)
    end
    return X
end

function QDLDL.solve!(fact::QDLDL.QDLDLFactorisation, X::AbstractMatrix)
    if isdiag(fact)
        X .= fact.Dinv * X
    else
        for x in eachcol(X)
            QDLDL.solve!(fact, x)
        end
    end
    return X
end


function solve_sqp!(nlp, Z, λ;
        iters=100,
        qp_solver=:osqp,
        adaptive_reg::Bool=false,
        verbose=0,
        eps_primal=1e-6,
        eps_dual=1e-6,
        eps_fn=sqrt(eps_primal),
        qpopts...
    )

    # Initialize solution
    qp = TOQP(size(nlp)..., num_eq(nlp), 0)
    ϕ = NormPenalty(10.0, 1, num_primals(nlp), num_eq(nlp))
    for opt in qpopts
        if opt.first ∈ keys(qp.opts)
            qp.opts[opt.first] = opt.second
        end
    end


    if qp_solver == :osqp
        qp_solver = OSQP.Model(qp, verbose=false)
    elseif qp_solver == :shur
        qp_solver = ShurSolver(qp)
    elseif qp_solver == :kkt
        qp_solver = KKTSolver(qp)
    elseif qp_solver == :qdldl_kkt
        qp_solver = QDLDLSolver(qp, method=:kkt)
    elseif qp_solver == :qdldl_shur
        qp_solver = QDLDLSolver(qp, method=:shur)
    end

    # Initialize BFGS w/ Hessian of Lagrangian
    if qp.opts[:hess] == :bfgs
        hess_f!(nlp, qp.B, Z)
        grad_lagrangian!(nlp, qp.g, Z, λ)
        qp.x .= Z
    end

    reg = 0.0
    dZ = zero(Z)

    for iter = 1:iters
        ## Check the residuals and cost
        res_p = primal_residual(nlp, Z, λ)
        res_d = dual_residual(nlp, Z, λ)
        J = eval_f(nlp, Z)
        verbose > 0 && @printf("Iteration %d: cost = %0.2f, res_p = %0.2e, res_d = %0.2e,", iter, J, res_p, res_d)

        if res_p < eps_primal && res_d < eps_dual 
            verbose > 0 && println()
            break
        end

        # Build QP
        build_qp!(qp, nlp, Z, λ)

        # Solve the QP
        if adaptive_reg
            # reg = find_reg(qp.Q, step=5, iters=20)
            qp.Q .+= I(num_primals(nlp))*reg
        end
        dZ, dλ = solve_qp!(qp_solver, qp, 
            verbose=false, polish=true, eps_rel=1e-10, eps_abs=1e-10) 
        qp_res_p = norm(qp.Q*dZ + qp.q + qp.A'dλ)
        qp_res_d = norm(qp.A*dZ - qp.b)
        verbose > 1 && @printf(" qp_res_p = %0.2e, qp_res_d = %0.2e, δλ = %0.2e", qp_res_p, qp_res_d, norm(dλ,Inf))

        ## Line Search

        # Update penalty paramter
        μ_min = minimum_penalty(ϕ, qp.Q, qp.q, qp.b, dZ)
        dot(dZ,qp.Q,dZ)
        if ϕ.μ < μ_min
            ϕ.μ = μ_min*1.1
        end

        # Actual line search
        α = 1.0
        phi0 = ϕ(nlp,Z)
        dphi0 = dgrad(ϕ, nlp, Z, dZ)
        phi = Inf

        τ = 0.5
        η = 1e-2
        Z̄ = copy(Z)
        dphi1 = gradient(ϕ, nlp, Z)'dZ
        verbose > 2 && @printf("\n   ϕ0: %0.2f, ϕ′: %0.2e, %0.2e\n", phi0, dphi0, dphi1)
        for i = 1:10
            Z̄ .= Z .+ α .* dZ
            λbar = λ - α*dλ
            phi = ϕ(nlp, Z̄)
            res_d = dual_residual(nlp, Z̄, λbar)
            res_p = primal_residual(nlp, Z̄, λbar)
            verbose > 2 && @printf("   ls iter: %d, Δϕ: %0.2e, ϕ′: %0.2e, res_p: %0.2e, res_d: %0.2e\n", 
                i, phi-phi0, η*α*dphi0, res_p, res_d)
            if phi < phi0 + η*α*dphi0
                reg = 0
                break
            else
                α *= τ
            end
            if i == 10
                # reg += 10
                α = 0
                Z̄ .= Z
            end
        end
        Z .= Z̄
        # λ .= -λhat
        λ .= λ - α*dλ
        A = qp.A
        # λ .= (A*A')\(A*qp.q)
        verbose > 0 && @printf("   α = %0.2f, ΔJ: %0.2e, Δϕ: %0.2e, ϕ′: %0.2e, reg: %0.2f, pen: %d\n", 
            α, J - eval_f(nlp, Z), phi0 - phi, dgrad(ϕ, nlp, Z, dZ), reg, ϕ.μ)
        if α == 0.0
            @warn "line search failed"
            break
        end

        if res_p < eps_fn && qp.opts[:hess] == :gauss_newton
            println("Switching to Full Newton")
            qp.opts[:hess] = :newton
            qp.opts[:structure] = :unknown
            ϕ.μ = min(μ_min*10, 1.0)
        end

    end 
    return Z, λ, qp 
end