MOI.features_available(nlp::NLP) = [:Grad, :Jac]
MOI.initialize(nlp::NLP, features) = nothing

function MOI.jacobian_structure(nlp::NLP)
    N,M = num_primals(nlp), num_duals(nlp)
    n,m = size(nlp)
    jac = spzeros(Int,M,N)
    cnt = 1
    for i = 1:n
        jac[i,i] = cnt 
        cnt += 1
    end

    xi,ui = nlp.xinds, nlp.uinds
    idx = xi[1]
    nblk = length(idx) * (n+m)
    for k = 1:nlp.T-1
        idx = idx .+ n 
        zi = [xi[k];ui[k]]
        J = view(jac,idx,zi)
        vec(J) .= cnt:cnt+nblk-1
        cnt += nblk 
        
        for i = 1:n
            jac[idx[i], zi[end]+i] = cnt 
            cnt += 1
        end
    end
    r,c = TO.get_rc(jac)
    collect(zip(r,c))
end

MOI.hessian_lagrangian_structure(nlp::NLP) = []
@inline MOI.eval_objective(nlp::NLP, Z) = eval_f(nlp, Z)
@inline MOI.eval_objective_gradient(nlp::NLP, grad_f, Z) = grad_f!(nlp, grad_f, Z)
@inline MOI.eval_constraint(nlp::NLP, c, Z) = eval_c!(nlp, c, Z)
@inline MOI.eval_constraint_jacobian(nlp::NLP, jac, Z) = jac_c!(nlp, jac, Z)
@inline MOI.eval_hessian_lagrangian(::NLP, H, x, σ, μ) = nothing

function build_MOI!(optimizer::MOI.AbstractOptimizer, nlp::NLP, Z)
    N,M = num_primals(nlp), num_duals(nlp)

    has_objective = true
    cL,cU = zeros(M), zeros(M)
    nlp_bounds = MOI.NLPBoundsPair.(cL, cU)
    block_data = MOI.NLPBlockData(nlp_bounds, nlp, has_objective)

    primals = MOI.add_variables(optimizer, N)
    MOI.set(optimizer, MOI.VariablePrimalStart(), primals, Z)
    MOI.set(optimizer, MOI.NLPBlock(), block_data)
    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    return optimizer
end
