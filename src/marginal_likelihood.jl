#=
marginal_likelihood.jl — Marginal likelihood and hyperparameter optimization
=#

"""
    compute_marginal_likelihood(y, p, prior::MinnesotaPrior; ...)

Compute log marginal data density for Minnesota prior.
"""
function compute_marginal_likelihood(y::AbstractMatrix, p::Int, prior::MinnesotaPrior;
                                     constant::Bool=true, firstobs::Int=p+1, presample::Int=0)
    T, K = size(y)
    nx = constant ? 1 : 0

    prior_info, posterior, _, _, _, _, _ =
        compute_prior_posterior(y, p, prior; constant=constant, presample=presample, firstobs=firstobs)

    post_int = matrictint(posterior.S, posterior.df, posterior.XXi)
    prior_int = matrictint(prior_info.S, prior_info.df, prior_info.XXi)
    lik_nobs = posterior.df - prior_info.df
    return post_int - prior_int - 0.5 * K * lik_nobs * log(2π)
end

"""
    optimize_hyperparameters(y, p; bounds, method=:grid)

Optimize Minnesota prior hyperparameters by maximizing marginal likelihood.

Returns `(best_prior::MinnesotaPrior, best_logml::Float64)`.
"""
function optimize_hyperparameters(y::AbstractMatrix, p::Int;
                                  tau_range::AbstractVector{Float64}=collect(0.5:0.5:5.0),
                                  decay_range::AbstractVector{Float64}=[0.5, 1.0],
                                  lambda_range::AbstractVector{Float64}=collect(1.0:1.0:10.0),
                                  mu_range::AbstractVector{Float64}=collect(1.0:1.0:5.0),
                                  omega_range::AbstractVector{Float64}=[1.0, 2.0, 3.0],
                                  constant::Bool=true)
    best_ml = -Inf
    best_prior = MinnesotaPrior()

    for tau in tau_range
        for decay in decay_range
            for lambda in lambda_range
                for mu in mu_range
                    for omega in omega_range
                        pr = MinnesotaPrior(tau=tau, decay=decay,
                                            lambda=lambda, mu=mu, omega=omega)
                        ml = try
                            compute_marginal_likelihood(y, p, pr; constant=constant)
                        catch
                            -Inf
                        end
                        if ml > best_ml
                            best_ml = ml
                            best_prior = pr
                        end
                    end
                end
            end
        end
    end

    return best_prior, best_ml
end

"""
    optimize_hyperparameters_optim(y, p; hyperpara, index_est, lb, ub, method, constant)

Optimize Minnesota prior hyperparameters via gradient‑free/gradient‑based
optimization of the marginal likelihood.

Optimization is performed in **log‑space** (matching MATLAB `bvar_max_hyper.m`).

# Keywords
- `hyperpara::Vector{Float64}`:  starting values `[tau, decay, lambda, mu, omega]`
                                  (default `[3.0, 0.5, 5.0, 2.0, 2.0]`).
- `index_est::Vector{Int}`:      indices of hyperparameters to optimize
                                  (default `1:5`, i.e. all).  Fixed entries keep
                                  their `hyperpara` values.
- `lb::Vector{Float64}`:         lower bounds in original space (length = |index_est|).
- `ub::Vector{Float64}`:         upper bounds in original space.
- `method::Symbol`:              `:nelder_mead` (default), `:lbfgs`, or `:bfgs`.
- `constant::Bool`:              include intercept (default `true`).

# Returns
`(best_prior::MinnesotaPrior, best_logml::Float64)`.
"""
function optimize_hyperparameters_optim(y::AbstractMatrix, p::Int;
                                        hyperpara::Vector{Float64}=[3.0, 0.5, 5.0, 2.0, 2.0],
                                        index_est::Vector{Int}=collect(1:5),
                                        lb::Vector{Float64}=fill(1e-4, length(index_est)),
                                        ub::Vector{Float64}=fill(50.0, length(index_est)),
                                        method::Symbol=:nelder_mead,
                                        constant::Bool=true)
    length(hyperpara) == 5 || throw(ArgumentError("hyperpara must have 5 elements"))
    all(1 .<= index_est .<= 5) || throw(ArgumentError("index_est entries must be in 1:5"))
    length(lb) == length(index_est) || throw(ArgumentError("lb length must match index_est"))
    length(ub) == length(index_est) || throw(ArgumentError("ub length must match index_est"))

    index_fixed = setdiff(1:5, index_est)
    fixed_vals = hyperpara[index_fixed]
    x0 = log.(hyperpara[index_est])
    log_lb = log.(lb)
    log_ub = log.(ub)

    # Clamp initial values to be within bounds
    x0 = clamp.(x0, log_lb, log_ub)

    names = [:tau, :decay, :lambda, :mu, :omega]

    function _neg_logml(x_log)
        full = copy(hyperpara)
        full[index_est] = exp.(x_log)
        pr = MinnesotaPrior(tau=full[1], decay=full[2], lambda=full[3],
                            mu=full[4], omega=full[5])
        ml = try
            compute_marginal_likelihood(y, p, pr; constant=constant)
        catch
            -Inf
        end
        return ml == -Inf ? 1e10 : -ml
    end

    if method == :nelder_mead
        opt_method = Optim.NelderMead()
    elseif method == :lbfgs
        opt_method = Optim.LBFGS()
    elseif method == :bfgs
        opt_method = Optim.BFGS()
    else
        throw(ArgumentError("method must be :nelder_mead, :lbfgs, or :bfgs"))
    end

    result = Optim.optimize(_neg_logml, log_lb, log_ub, x0,
                             Optim.Fminbox(opt_method),
                             Optim.Options(iterations=5000, g_tol=1e-8))

    xh = Optim.minimizer(result)
    best = copy(hyperpara)
    best[index_est] = exp.(xh)
    best_prior = MinnesotaPrior(tau=best[1], decay=best[2], lambda=best[3],
                                mu=best[4], omega=best[5])
    best_logml = -Optim.minimum(result)
    return best_prior, best_logml
end
