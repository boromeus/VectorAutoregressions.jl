#=
bayesian.jl — BVAR Gibbs sampler and classical bootstrap VAR
=#

"""
    bvar(y, p; prior, identification, K, hor, fhor, constant, trend,
         non_explosive, exogenous, verbose, rng)

Estimate a Bayesian VAR(p) and compute posterior draws.

# Arguments
- `y::Matrix`:              T × n_var data matrix.
- `p::Int`:                 lag order.
- `prior::AbstractPrior`:   prior specification (default `FlatPrior()`).
- `identification`:         identification scheme (default `CholeskyIdentification()`).
- `K::Int`:                 number of posterior draws (default 5000).
- `hor::Int`:               IRF horizon (default 24).
- `fhor::Int`:              forecast horizon (default 12).
- `constant::Bool`:         include intercept (default `true`).
- `trend::Bool`:            include time trend (default `false`).
- `non_explosive::Bool`:    reject explosive draws (default `false`).
- `verbose::Bool`:          print progress (default `true`).

# Returns
A `BVARResult` struct.
"""
function bvar(y::AbstractMatrix{<:Real}, p::Int;
        prior::AbstractPrior = FlatPrior(),
        identification::AbstractIdentification = CholeskyIdentification(),
        K::Int = 5000, hor::Int = 24, fhor::Int = 12,
        constant::Bool = true, trend::Bool = false,
        non_explosive::Bool = false,
        exogenous::Union{Nothing, AbstractMatrix{<:Real}} = nothing,
        verbose::Bool = true,
        rng::AbstractRNG = Random.default_rng())
    p > 0 || throw(ArgumentError("lag order p must be positive"))
    T_raw, ny = size(y)

    nexo = exogenous !== nothing ? size(exogenous, 2) : 0

    # ── Compute OLS estimate and prior/posterior moments ──
    prior_info, posterior,
    B_ols,
    u_ols,
    xxi_ols,
    y_ols,
    X_ols = compute_prior_posterior(
        y, p, prior; constant = constant, trend = trend, nexogenous = nexo)

    nobs = size(y_ols, 1) - p  # effective obs for the VAR part
    nk = size(X_ols, 2)
    nx = (constant ? 1 : 0) + (trend ? 1 : 0) + nexo

    # Build the actual‑data‑only OLS estimate for output
    var_ols = var_estimate(y, p; constant = constant, trend = trend, exogenous = exogenous)

    # ── Information criteria on OLS ──
    ic = information_criteria(var_ols)

    # ── Marginal likelihood ──
    log_dnsty = try
        post_int = matrictint(posterior.S, posterior.df, posterior.XXi)
        if prior isa MinnesotaPrior
            prior_int = matrictint(prior_info.S, prior_info.df, prior_info.XXi)
            lik_nobs = posterior.df - prior_info.df
            post_int - prior_int - 0.5 * ny * lik_nobs * log(2π)
        elseif prior isa FlatPrior
            lik_nobs = posterior.df
            post_int - 0.5 * ny * lik_nobs * log(2π)
        else
            NaN  # conjugate uses different formula
        end
    catch
        @warn "Could not compute marginal likelihood"
        NaN
    end

    # ── Prepare for posterior sampling ──
    S_inv_upper_chol = try
        cholesky(Hermitian(inv(posterior.S))).U
    catch
        @warn "Posterior covariance ill‑conditioned, adding ridge"
        cholesky(Hermitian(inv(posterior.S + 1e-5 * I(ny)))).U
    end

    XXi_lower_chol = cholesky(Hermitian(posterior.XXi)).L

    # ── Pre‑allocate storage ──
    Phi_draws = zeros(nk, ny, K)
    Sigma_draws = zeros(ny, ny, K)
    ir_draws = zeros(ny, hor, ny, K)
    irlr_draws = zeros(ny, hor, ny, K)
    irsign_draws = zeros(ny, hor, ny, K)
    irnarrsign_draws = zeros(ny, hor, ny, K)
    irzerosign_draws = zeros(ny, hor, ny, K)
    irproxy_draws = zeros(ny, hor, ny, K)

    # Actual‑data regressors for residuals
    YY = var_ols.Y
    XX = var_ols.X
    e_draws = zeros(size(YY, 1), ny, K)
    Omega_draws = zeros(ny, ny, K)

    # Forecasts
    yhatfut_no = fill(NaN, fhor, ny, K)
    yhatfut_with = fill(NaN, fhor, ny, K)
    yhatfut_cond = fill(NaN, fhor, ny, K)

    # Forecast initial values
    forecast_initval = y[(end - p + 1):end, :]
    forecast_xdata = ones(fhor, constant ? 1 : 0)
    if trend
        T_eff = size(YY, 1)
        forecast_xdata = hcat(forecast_xdata, collect((T_eff + 1):(T_eff + fhor)))
    end

    # Companion matrix template
    Companion = zeros(ny * p, ny * p)
    if p > 1
        Companion[(ny + 1):end, 1:(ny * (p - 1))] = I(ny*(p-1))
    end

    dd = 0
    rejected = 0
    Phi = zeros(nk, ny)
    Sigma = zeros(ny, ny)

    for d in 1:K
        dd = 0
        inner_iter = 0
        while dd == 0
            inner_iter += 1
            if inner_iter > 10000
                @warn "Could not find stable draw after 10000 attempts in iteration $d"
                dd = 1
                break
            end
            # Step 1: Draw Σ ~ IW
            Sigma = rand_inverse_wishart(posterior.df, posterior.S; rng = rng)
            Sigma_chol = cholesky(Hermitian(Sigma)).L

            # Step 2: Draw Φ | Σ ~ MN
            Phi1 = randn(rng, nk * ny)
            Phi2 = kron(Sigma_chol, XXi_lower_chol) * Phi1
            Phi3 = reshape(Phi2, nk, ny)
            Phi = Phi3 + posterior.PhiHat

            # Step 3: Check stability
            if non_explosive
                Companion[1:ny, :] = Phi[1:(ny * p), :]'
                maxeig = maximum(abs.(eigvals(Companion)))
                if maxeig > 1.01
                    rejected += 1
                    continue
                end
            end
            dd = 1
        end

        # Store draws
        Phi_draws[:, :, d] = Phi
        Sigma_draws[:, :, d] = Sigma
        e_draws[:, :, d] = YY - XX * Phi

        # ── Compute IRFs ──
        # Cholesky (always)
        ir_draws[:, :, :, d] = compute_irf(Phi[1:(ny * p), :], Sigma, hor)

        # Long‑run
        if identification isa LongRunIdentification
            irlr, Qlr = compute_irf_longrun(Phi[1:(ny * p), :], Sigma, hor, p)
            irlr_draws[:, :, :, d] = irlr
            Omega_draws[:, :, d] = Qlr
        end

        # Sign restrictions
        if identification isa SignRestriction
            irsign,
            Omega = irf_sign_restriction(Phi[1:(ny * p), :], Sigma, hor,
                identification.restrictions;
                max_rotations = identification.max_rotations, rng = rng)
            irsign_draws[:, :, :, d] = irsign
            Omega_draws[:, :, d] = Omega
        end

        # Narrative + sign
        if identification isa NarrativeSignRestriction
            irns,
            Omega = irf_narrative_sign(e_draws[:, :, d],
                Phi[1:(ny * p), :], Sigma, hor,
                identification.signs, identification.narrative;
                max_rotations = identification.max_rotations, rng = rng)
            irnarrsign_draws[:, :, :, d] = irns
            Omega_draws[:, :, d] = Omega
        end

        # Zero + sign
        if identification isa ZeroSignRestriction
            irzs,
            Omega = irf_zero_sign(Phi, Sigma, hor, p,
                identification.restrictions;
                var_pos = identification.var_pos, rng = rng)
            irzerosign_draws[:, :, :, d] = irzs
            Omega_draws[:, :, d] = Omega
        end

        # Proxy
        if identification isa ProxyIdentification
            irs_p, b1,
            _ = compute_irf_proxy(Phi, Sigma, e_draws[:, :, d],
                identification.instrument, hor, p;
                proxy_end = identification.proxy_end)
            irproxy_draws[:, 1:size(irs_p, 1), 1, d] = irs_p'
        end

        # ── Forecasts ──
        lags_data = copy(forecast_initval)
        for t in 1:fhor
            x_fwd = vcat(vec(reverse(lags_data, dims = 1)'), forecast_xdata[t, :])
            y_pred = x_fwd' * Phi
            lags_data[1:(end - 1), :] = lags_data[2:end, :]
            lags_data[end, :] = y_pred
            yhatfut_no[t, :, d] = y_pred
        end

        lags_data = copy(forecast_initval)
        for t in 1:fhor
            x_fwd = vcat(vec(reverse(lags_data, dims = 1)'), forecast_xdata[t, :])
            shock = (cholesky(Hermitian(Sigma)).L * randn(rng, ny))'
            y_pred = x_fwd' * Phi + shock
            lags_data[1:(end - 1), :] = lags_data[2:end, :]
            lags_data[end, :] = y_pred
            yhatfut_with[t, :, d] = y_pred
        end

        if verbose && d % 1000 == 0
            @info "BVAR: draw $d / $K (rejected: $rejected)"
        end
    end

    verbose && rejected > 0 && @info "Total explosive draws rejected: $rejected"

    return BVARResult(
        var_ols, prior, identification,
        Phi_draws, Sigma_draws,
        ir_draws, irlr_draws, irsign_draws,
        irnarrsign_draws, irzerosign_draws, irproxy_draws,
        e_draws, Omega_draws,
        yhatfut_no, yhatfut_with, yhatfut_cond,
        log_dnsty, ic, K, p, ny, hor, fhor
    )
end

"""
    classical_var(y, p; nboot, identification, hor, constant, rng)

Classical VAR with bootstrap confidence intervals.
"""
function classical_var(y::AbstractMatrix{<:Real}, p::Int;
        nboot::Int = 1000, identification::AbstractIdentification = CholeskyIdentification(),
        hor::Int = 24, constant::Bool = true,
        rng::AbstractRNG = Random.default_rng())
    v = var_estimate(y, p; constant = constant)
    K = v.nvar
    T = v.nobs

    # Point estimate IRF
    ir_point = compute_irf(v.Phi[1:(K * p), :], v.Sigma, hor)

    # Bootstrap
    ir_boot = zeros(K, hor, K, nboot)
    for b in 1:nboot
        # Resample residuals
        idx = rand(rng, 1:T, T)
        u_boot = v.residuals[idx, :]

        # Build bootstrap sample
        y_boot = zeros(T + p, K)
        start_idx = rand(rng, 1:T)
        y_boot[1:p, :] = v.Y[start_idx:min(start_idx + p - 1, T), :]
        if start_idx + p - 1 > T
            y_boot[1:p, :] = v.Y[(end - p + 1):end, :]
        end

        for t in (p + 1):(T + p)
            x_t = vec(y_boot[(t - 1):-1:(t - p), :]')
            if constant
                x_t = vcat(x_t, 1.0)
            end
            y_boot[t, :] = x_t' * v.Phi + u_boot[t - p, :]'
        end

        v_boot = var_estimate(y_boot, p; constant = constant)
        ir_boot[:, :, :, b] = compute_irf(v_boot.Phi[1:(K * p), :], v_boot.Sigma, hor)
    end

    return ir_point, ir_boot, v
end
