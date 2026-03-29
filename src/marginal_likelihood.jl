#=
marginal_likelihood.jl — Marginal likelihood and hyperparameter optimization
=#

"""
    compute_marginal_likelihood(y, p, prior::MinnesotaPrior; ...)

Compute log marginal data density for Minnesota prior.
"""
function compute_marginal_likelihood(y::AbstractMatrix, p::Int, prior::MinnesotaPrior;
        constant::Bool = true, firstobs::Int = p+1, presample::Int = 0)
    T, K = size(y)
    nx = constant ? 1 : 0

    prior_info, posterior,
    _,
    _,
    _,
    _,
    _ = compute_prior_posterior(
        y, p, prior; constant = constant, presample = presample, firstobs = firstobs)

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
        tau_range::AbstractVector{Float64} = collect(0.5:0.5:5.0),
        decay_range::AbstractVector{Float64} = [0.5, 1.0],
        lambda_range::AbstractVector{Float64} = collect(1.0:1.0:10.0),
        mu_range::AbstractVector{Float64} = collect(1.0:1.0:5.0),
        omega_range::AbstractVector{Float64} = [1.0, 2.0, 3.0],
        constant::Bool = true)
    best_ml = -Inf
    best_prior = MinnesotaPrior()

    for tau in tau_range
        for decay in decay_range
            for lambda in lambda_range
                for mu in mu_range
                    for omega in omega_range
                        pr = MinnesotaPrior(tau = tau, decay = decay,
                            lambda = lambda, mu = mu, omega = omega)
                        ml = try
                            compute_marginal_likelihood(y, p, pr; constant = constant)
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
