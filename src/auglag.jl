using IterativeSolvers
using Preconditioners

function al_solve(nlp, Z, λ=zeros(num_eq(nlp)); 
        ρ=10.0,   # initial penalty
        ϕ=10.0,   # penalty scaling
        ρterm = ρ,
        max_iters = 20, 
        eps_primal = 1e-6,
        eps_dual = 1e-6,
        eps_inner = 1e-6,
        newton_iters= 20,
        newton_reg=1e-6,
        eps_fn = sqrt(eps_inner),
        pd::Bool=false,
        verbose=2,
        max_penalty=1e8,
        kwargs...
    )
    
    c = zero(λ)
    iters = 0
    linres = Float64[]
    viol = Float64[]

    ρ = fill(ρ, length(λ))
    if termcon(nlp)
        ρ[end] = ρterm
    end
    for i = 1:max_iters
        # Solve unconstrained problem
        if pd
            inner_iters, dres = pdal_newton!(nlp, Z, λ, ρ, 
                max_iters=newton_iters, eps=eps_inner, eps_fn=eps_fn, reg=newton_reg, 
                verbose=verbose>1
            ) 
        else
            inner_iters, dres = al_newton!(nlp, Z, λ, ρ, 
                max_iters=newton_iters, eps=eps_inner, eps_fn=eps_fn, reg=newton_reg, 
                verbose=verbose>1; kwargs... 
            ) 
        end
        iters += inner_iters
        append!(viol, dres)

        # dual update
        eval_c!(nlp, c, Z)
        if !pd
            λ .= λ - ρ .* c
        end
        
        # penalty update
        ρ .*= ϕ
        ρ .= min.(max_penalty, ρ)

        # Convergence
        pres = primal_residual(nlp, Z, λ, p=Inf)
        printstyled("Outer Loop ", i, ": ", bold=true, color=:yellow)
        verbose > 0 && @printf("viol: %0.2e, pres: %0.2e, ρ = %.1e, iters = %d\n", 
            norm(c, Inf), pres, ρ[1], iters)
        if norm(c, Inf) < eps_dual && pres < eps_primal
            break
        end
    end

    return Z, λ, viol 
end

function al_newton!(nlp, Z, λ, ρ;
        max_iters = 25,
        eps = 1e-6,
        reg = 1e-6,
        eps_fn = sqrt(eps),
        pd::Bool=false,
        verbose=false,
        linear_solve::Symbol = :cholesky,
        refine = 0,
        Pl = nothing
    )
    N,M = num_primals(nlp), num_duals(nlp)
    hess = spzeros(N,N)
    grad = zeros(N)
    Ireg = I*reg

    gn = true 
    iters = 0
    linres = Float64[]
    viol = Float64[]
    for iter = 1:max_iters
        alhess!(nlp, hess, Z, λ, ρ, gn)
        algrad!(nlp, grad, Z, λ, ρ)
        if norm(grad) < eps && iter > 1  # force at least 1 iteration
            verbose && println("  Converged in $iter iterations")
            break
        elseif gn && norm(grad) < eps_fn 
            verbose && println("  Switching to Full Newton")
            gn = false
        end

        H = Symmetric(hess,:L) + Ireg 
        if linear_solve == :cholesky
            dZ = -(cholesky(H)\grad)
        elseif linear_solve == :qdldl
            dZ = -pdsolve(H.data, grad)
        elseif linear_solve == :pcg
            if isnothing(Pl)
                dZ = cg(H, -grad)
            else
                p = Pl(H.data)
                dZ = cg(H, -grad, Pl=p)
            end
        end
        for i = 1:refine
            r = hess*dZ + grad
            println(norm(r))
            dZ .-= H\r
        end

        err = norm(H*dZ + grad)
        push!(linres, err)
        phi0 = aug_lagrangian(nlp, Z, λ, ρ)
        dphi0 = grad'dZ
        phi = 0.0

        c1 = 1e-3
        c2 = 0.9
        α = 1.0
        for i = 1:10
            phi = aug_lagrangian(nlp, Z + α*dZ, λ, ρ)
            dphi = al_dgrad(nlp, Z + α*dZ, dZ, λ, ρ)
            armijo = phi < phi0 + c1*α*dphi0
            wolfe = dphi > c2*dphi0
            if armijo && wolfe
                break
            else
                α *= 0.5
            end
        end
        Z̄ = Z .+ α*dZ
        dres = norm(dual_residual(nlp, Z̄, λ, p=Inf), Inf)
        push!(viol, dres)
        if verbose
            J0 = eval_f(nlp, Z)
            J = eval_f(nlp, Z̄)
            algrad!(nlp, grad, Z̄, λ, ρ)
            @printf("  J: %0.2f → %0.2f (% 0.2e), ϕ: %0.2f → %0.2f (% 0.2e), ϕ′: %0.2e, α: %0.2f, cond: %0.1e, pres: %0.2e, dres: %0.2e, grad: %0.2e, err: %0.2e\n", 
                J0, J, J0-J, 
                phi0, phi, phi0-phi,
                dphi0, α, 0, #cond(Matrix(hess)),
                dres,
                norm(dual_residual(nlp, Z̄, λ, p=Inf)),
                norm(grad),
                err
            )
        end
        iters += 1
        # phi0 - aug_lagrangian(nlp, Z + α*dZ, λ, ρ)
        # dual_residual(nlp, Z + α*dZ, λ)
        # primal_residual(nlp, Z + 0*dZ, λ)
        # primal_residual(nlp, Z + α*dZ, λ)
        Z .+= α .* dZ
    end
    return iters, viol 
end

function pdal_newton!(nlp, Z, λ, ρ;
        max_iters = 25,
        eps = 1e-6,
        reg = 1e-6,
        eps_fn = sqrt(eps),
        pd::Bool=false,
        verbose=false
    )
    N,M = num_primals(nlp), num_duals(nlp)
    hess = spzeros(N+M,N+M)
    grad = zeros(N+M)
    Ireg = Diagonal([i > N ? -reg : reg for i = 1:N+M]) 

    gn = true 
    iters = 0
    λtilde = copy(λ)
    viol = Float64[]
    for iter = 1:max_iters

        pdal_sys!(nlp, hess, grad, Z, λtilde, λ, ρ, gn)
        if norm(grad) < eps && iter > 1  # force at least 1 iteration
            verbose && println("  Converged in $iter iterations ($(norm(grad)))")
            break
        elseif gn && norm(grad) < eps_fn 
            verbose && println("  Switching to Full Newton")
            gn = false
        end

        ldl = qdldl(hess + Ireg)
        dY = -(ldl\grad)
        dY = kktsolve(hess, -grad, Ireg)
        # dY = -((Symmetric(hess,:L) + Ireg)\grad)
        dZ = @view dY[1:N]
        dλtilde = @view dY[1+N:end]
        phi0 = aug_lagrangian(nlp, Z, λtilde, ρ)
        # dphi0 = grad[1:N]'dZ
        dphi0 = al_dgrad(nlp, Z, dZ, λtilde, ρ)
        phi = 0.0

        c1 = 1e-3
        c2 = 0.9
        α = 1.0

        for i = 1:10
            phi = aug_lagrangian(nlp, Z + α*dZ, λtilde + α*dλtilde, ρ)
            dphi = al_dgrad(nlp, Z + α*dZ, dZ, λtilde + α*dλtilde, ρ)
            armijo = phi < phi0 + c1*α*dphi0
            wolfe = dphi > c2*dphi0
            @printf("    α = %0.3f phi0: %0.2e phi: %0.2e Δphi: %0.2e dphi0: %0.2e dphi: %0.2e armijo? %d wolfe? %d\n", 
                α, phi0, phi, phi0-phi,
                dphi0, dphi,
                armijo, wolfe)
            if armijo #&& wolfe
                break
            else
                α *= 0.5
            end
            i == 10 && (α = 0.0)
        end
        # α = 0.1

        Z̄ = Z .+ α*dZ
        pres = norm(primal_residual(nlp, Z̄, λ, p=Inf),Inf)
        dres = norm(dual_residual(nlp, Z̄, λ, p=Inf),Inf)
        push!(viol, dres)
        if verbose
            J0 = eval_f(nlp, Z)
            J = eval_f(nlp, Z̄)
            algrad!(nlp, grad, Z̄, λ, ρ)
            @printf("  J: %0.2f → %0.2f (%0.2e), ϕ: %0.2f → %0.2f (%0.2e), ϕ′: %0.2e, α: %0.2f, pres: %0.2e, dres: %0.2e, grad: %0.2e, inertia: %d\n", 
                J0, J, J0-J, 
                phi0, phi, phi0-phi,
                dphi0, α, 
                pres,dres,
                norm(grad),
                QDLDL.positive_inertia(ldl)
            )
        end
        iters += 1
        # phi0 - aug_lagrangian(nlp, Z + α*dZ, λ, ρ)
        # dual_residual(nlp, Z + α*dZ, λ)
        # primal_residual(nlp, Z + 0*dZ, λ)
        # primal_residual(nlp, Z + α*dZ, λ)
        positive_inertia(ldl) < N && break
        if α ≈ 0
            @warn "Line search failed"
            reg *= 10 
        end

        # Dual Update
        λtilde .+= α .* dλtilde
        Z .+= α .* dZ
        
    end
    λ .= λtilde
    return iters, viol
end

function pdsolve(A,b)
    ldl = qdldl(A)
    for i = 1:length(b)
        if ldl.Dinv[i,i] <= 0
            ldl.Dinv[i,i] = 1e-2
        end
    end
    ldl\b 
end

function kktsolve(A,b,Ireg)
    ldl = qdldl(A + Ireg)
    N = count(x->x>0, diag(Ireg))
    k = 1
    inertia = positive_inertia(ldl)
    while inertia < N
        ldl = qdldl(A + k*Ireg)
        k += 1
        inertia = positive_inertia(ldl)
        @show inertia
        @show maximum(abs.(ldl.Dinv))
        @show minimum(ldl.Dinv)
        if k >= 100
            @warn "kkt solve failed"
            break
        end
    end
    return ldl\b
end