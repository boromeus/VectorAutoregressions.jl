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
              prior::AbstractPrior=FlatPrior(),
              identification::AbstractIdentification=CholeskyIdentification(),
              K::Int=5000, hor::Int=24, fhor::Int=12,
              constant::Bool=true, trend::Bool=false,
              non_explosive::Bool=false,
              exogenous::Union{Nothing,AbstractMatrix{<:Real}}=nothing,
              exogenous_block::Union{Nothing,AbstractMatrix{<:Real}}=nothing,
              heterosked_weights::Union{Nothing,AbstractVector{<:Real}}=nothing,
              robust_bayes::Int=0,
              K_shrinkage::Float64=NaN,
              verbose::Bool=true,
              rng::AbstractRNG=Random.default_rng())

    p > 0 || throw(ArgumentError("lag order p must be positive"))
    T_raw, ny = size(y)

    nexo = exogenous !== nothing ? size(exogenous, 2) : 0

    # ── Exogenous block setup ──
    nz = 0
    z_var = nothing
    z_posterior = nothing
    z_S_inv_upper_chol = nothing
    z_XXi_lower_chol = nothing
    z_exo_data = nothing   # T_raw × nz*p matrix of z-lag data for prior/posterior
    if exogenous_block !== nothing
        nz = size(exogenous_block, 2)
        size(exogenous_block, 1) == T_raw ||
            throw(ArgumentError("exogenous_block must have T rows"))

        # Estimate independent VAR on z
        z_var = var_estimate(exogenous_block, p; constant=constant, trend=trend)

        # Build z-lag matrix as exogenous regressors for y-equation
        z_lags = lagmatrix(exogenous_block, p)  # (T_raw - p) × nz*p
        # Pad to T_raw rows: zeros for initial p rows (rfvar3 skips them via sample selection)
        z_exo_data = zeros(T_raw, nz * p)
        z_exo_data[p+1:end, :] = z_lags

        # Also set exogenous for var_estimate (includes z-lags)
        if exogenous !== nothing
            exogenous = hcat(exogenous[p+1:end, :], z_lags)
        else
            exo_full = zeros(T_raw, nz * p)
            exo_full[p+1:end, :] = z_lags
            exogenous = exo_full
        end
        nexo = size(exogenous, 2)

        # z posterior moments (flat prior for z equation)
        z_u = z_var.residuals
        z_nobs = z_var.nobs
        z_nk = size(z_var.X, 2)
        z_df = z_nobs - z_nk + nz + 1
        z_S = z_u' * z_u
        z_XXi = z_var.XXi
        z_PhiHat = z_var.Phi

        z_posterior = (df=z_df, S=z_S, XXi=z_XXi, PhiHat=z_PhiHat)
        z_S_inv_upper_chol = try
            cholesky(Hermitian(inv(z_S))).U
        catch
            cholesky(Hermitian(inv(z_S + 1e-5 * I(nz)))).U
        end
        z_XXi_lower_chol = cholesky(Hermitian(z_XXi)).L
    end

    # ── Compute OLS estimate and prior/posterior moments ──
    prior_info, posterior, B_ols, u_ols, xxi_ols, y_ols, X_ols =
        compute_prior_posterior(y, p, prior; constant=constant, trend=trend,
                                nexogenous=nexo,
                                exogenous_data=z_exo_data,
                                ww=heterosked_weights)

    nobs = size(y_ols, 1) - p  # effective obs for the VAR part
    nk = size(X_ols, 2)
    nx = (constant ? 1 : 0) + (trend ? 1 : 0) + nexo

    # Build the actual‑data‑only OLS estimate for output
    var_ols = var_estimate(y, p; constant=constant, trend=trend, exogenous=exogenous)

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

    # ── Robust Bayes setup ──
    rb_setup = nothing
    rb_skew_setup = nothing
    if robust_bayes > 0
        nobs_rb = size(var_ols.residuals, 1)
        kshrink = isnan(K_shrinkage) ? Float64(ny) : K_shrinkage
        rb_setup = robust_bayes_setup(var_ols.residuals, nobs_rb, ny;
                                       K_shrinkage=kshrink)
        if robust_bayes > 1
            vech_Sig_cov_ = rb_setup.vech_Sig_cov_lower_chol *
                            rb_setup.vech_Sig_cov_lower_chol'
            rb_skew_setup = robust_bayes_setup_skewness(var_ols.residuals,
                                nobs_rb, ny, vech_Sig_cov_, rb_setup.Sig_ols;
                                K_shrinkage=kshrink)
        end
    end

    # ── Pre‑allocate storage ──
    # For exogenous block: combined system dimensions
    ny_total = exogenous_block !== nothing ? ny + nz : ny
    nk_total = exogenous_block !== nothing ? (ny + nz) * p + (constant ? 1 : 0) + (trend ? 1 : 0) : nk + nexo

    Phi_draws   = zeros(nk_total, ny_total, K)
    Sigma_draws = zeros(ny_total, ny_total, K)
    ir_draws    = zeros(ny_total, hor, ny_total, K)
    irlr_draws  = zeros(ny_total, hor, ny_total, K)
    irsign_draws = zeros(ny_total, hor, ny_total, K)
    irnarrsign_draws = zeros(ny_total, hor, ny_total, K)
    irzerosign_draws = zeros(ny_total, hor, ny_total, K)
    irproxy_draws = zeros(ny_total, hor, ny_total, K)
    irheterosked_draws = zeros(ny_total, hor, ny_total, K)

    # Actual‑data regressors for residuals
    YY = var_ols.Y
    XX = var_ols.X
    # Separate endogenous/deterministic columns from exogenous columns
    XX_endo = XX[:, 1:nk]
    XX_exo  = nexo > 0 ? XX[:, nk+1:end] : nothing
    Gamma_ols = nexo > 0 ? var_ols.Phi[nk+1:end, :] : nothing
    e_draws = zeros(size(YY, 1), ny, K)
    Omega_draws = zeros(ny_total, ny_total, K)

    # Forecasts
    yhatfut_no    = fill(NaN, fhor, ny_total, K)
    yhatfut_with  = fill(NaN, fhor, ny_total, K)
    yhatfut_cond  = fill(NaN, fhor, ny_total, K)

    # Forecast initial values
    forecast_initval = y[end-p+1:end, :]
    forecast_xdata = ones(fhor, constant ? 1 : 0)
    if trend
        T_eff = size(YY, 1)
        forecast_xdata = hcat(forecast_xdata, collect(T_eff+1:T_eff+fhor))
    end

    # Companion matrix template
    ncomp = ny_total * p
    Companion = zeros(ncomp, ncomp)
    if p > 1
        Companion[ny_total+1:end, 1:ny_total*(p-1)] = I(ny_total*(p-1))
    end

    dd = 0
    rejected = 0
    Phi = zeros(nk, ny)
    Sigma = zeros(ny, ny)
    Sig2_rb = nothing  # used for robust_bayes skewness correction

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
            # Step 1: Draw Σ
            if robust_bayes > 0 && rb_setup !== nothing
                # Robust draw (kurtosis-adjusted)
                Sigma_rb, Sigma_chol_rb, Sig2_rb = robust_sigma_draw(rb_setup; rng=rng)
                if Sigma_rb === nothing
                    continue  # retry if not PSD
                end
                Sigma = Sigma_rb
                Sigma_chol = Sigma_chol_rb
            else
                if exogenous_block !== nothing && z_posterior !== nothing
                    # Exogenous block: draw y and z Sigma separately
                    ySigma = rand_inverse_wishart(posterior.df, posterior.S; rng=rng)
                    zSigma = rand_inverse_wishart(z_posterior.df, z_posterior.S; rng=rng)
                    Sigma = [ySigma zeros(ny, nz); zeros(nz, ny) zSigma]
                    Sigma_chol = cholesky(Hermitian(Sigma)).L
                    ySigma_chol = cholesky(Hermitian(ySigma)).L
                    zSigma_chol = cholesky(Hermitian(zSigma)).L
                else
                    Sigma = rand_inverse_wishart(posterior.df, posterior.S; rng=rng)
                    Sigma_chol = cholesky(Hermitian(Sigma)).L
                end
            end

            # Step 2: Draw Φ | Σ ~ MN
            if exogenous_block !== nothing && z_posterior !== nothing
                # y-equation draw (includes z-lag coefficients)
                Phi1 = randn(rng, nk * ny)
                Phi2 = kron(ySigma_chol, XXi_lower_chol) * Phi1
                Phi3 = reshape(Phi2, nk, ny)
                yPhi = Phi3 + posterior.PhiHat

                # z-equation draw (independent VAR)
                z_nk = size(z_posterior.PhiHat, 1)
                zPhi1 = randn(rng, z_nk * nz)
                zPhi2 = kron(zSigma_chol, z_XXi_lower_chol) * zPhi1
                zPhi3 = reshape(zPhi2, z_nk, nz)
                zPhi = zPhi3 + z_posterior.PhiHat

                # Assemble combined Phi: [(ny+nz)*p + nx_det] × (ny+nz)
                # Layout: interleaved lags for companion form compatibility
                # Each lag block is (ny+nz) rows: [y_coeffs; z_coeffs]
                nx_det = (constant ? 1 : 0) + (trend ? 1 : 0)
                yPhic = yPhi[ny*p+1:ny*p+nx_det, :] # deterministics → y
                zPhic = zPhi[nz*p+1:nz*p+nx_det, :] # deterministics → z

                K_total = ny + nz
                Phi = zeros(K_total * p + nx_det, K_total)
                for ell in 1:p
                    # y-lag ℓ → y equation: rows (ell-1)*ny+1 : ell*ny of yPhi
                    yPhiy_ell = yPhi[(ell-1)*ny+1:ell*ny, :]
                    # z-lag ℓ → y equation: rows ny*p+nx_det+(ell-1)*nz+1 : ny*p+nx_det+ell*nz of yPhi
                    yPhiz_ell = yPhi[ny*p+nx_det+(ell-1)*nz+1:ny*p+nx_det+ell*nz, :]
                    # z-lag ℓ → z equation: rows (ell-1)*nz+1 : ell*nz of zPhi
                    zPhiz_ell = zPhi[(ell-1)*nz+1:ell*nz, :]

                    # y does not enter z equation (exogeneity)
                    zPhiy_ell = zeros(ny, nz)

                    # Interleaved block for lag ℓ: (ny+nz) × (ny+nz)
                    row_start = (ell-1)*K_total + 1
                    Phi[row_start:row_start+ny-1, 1:ny] = yPhiy_ell
                    Phi[row_start:row_start+ny-1, ny+1:end] = zPhiy_ell
                    Phi[row_start+ny:row_start+K_total-1, 1:ny] = yPhiz_ell
                    Phi[row_start+ny:row_start+K_total-1, ny+1:end] = zPhiz_ell
                end
                # Deterministics at the end
                Phi[K_total*p+1:end, 1:ny] = yPhic
                Phi[K_total*p+1:end, ny+1:end] = zPhic
            else
                Phi1 = randn(rng, nk * ny)
                Phi2 = kron(Sigma_chol, XXi_lower_chol) * Phi1
                Phi3 = reshape(Phi2, nk, ny)
                Phi = Phi3 + posterior.PhiHat
            end

            # Step 2b: Skewness correction for intercept
            if robust_bayes > 1 && rb_skew_setup !== nothing
                const_row = ny * p + 1
                if const_row <= nk
                    mu_rob = posterior.PhiHat[const_row, :] +
                             rb_skew_setup.Sstar * rb_skew_setup.ivech_Sig_cov *
                             (Sig2_rb - rb_setup.vech_Sig)
                    Phi[const_row, :] = mu_rob + rb_skew_setup.mu_cov_chol * randn(rng, ny)
                end
            end

            # Step 3: Check stability
            if non_explosive
                Companion[1:ny_total, :] = Phi[1:ny_total*p, :]'
                maxeig = maximum(abs.(eigvals(Companion)))
                if maxeig > 1.01
                    rejected += 1
                    continue
                end
            end
            dd = 1
        end

        # Store draws
        if exogenous_block !== nothing
            Phi_draws[:, :, d] = Phi
            Sigma_draws[:, :, d] = Sigma
            # Residuals only for y-equation part
            e_draws[:, :, d] = YY - XX_endo * posterior.PhiHat
        else
            Phi_draws[1:nk, :, d] = Phi
            if nexo > 0
                Phi_draws[nk+1:end, :, d] = Gamma_ols
            end
            Sigma_draws[:, :, d] = Sigma
            e_draws[:, :, d] = YY - XX_endo * Phi
            if XX_exo !== nothing
                e_draws[:, :, d] .-= XX_exo * Gamma_ols
            end
        end

        # ── Compute IRFs ──
        # Cholesky (always)
        ir_draws[:, :, :, d] = compute_irf(Phi[1:ny_total*p, :], Sigma, hor)

        # Long‑run
        if identification isa LongRunIdentification
            irlr, Qlr = compute_irf_longrun(Phi[1:ny_total*p, :], Sigma, hor, p)
            irlr_draws[:, :, :, d] = irlr
            Omega_draws[:, :, d] = Qlr
        end

        # Sign restrictions
        if identification isa SignRestriction
            irsign, Omega = irf_sign_restriction(Phi[1:ny_total*p, :], Sigma, hor,
                                                  identification.restrictions;
                                                  max_rotations=identification.max_rotations, rng=rng)
            irsign_draws[:, :, :, d] = irsign
            Omega_draws[:, :, d] = Omega
        end

        # Narrative + sign
        if identification isa NarrativeSignRestriction
            irns, Omega = irf_narrative_sign(e_draws[:, :, d],
                              Phi[1:ny_total*p, :], Sigma, hor,
                              identification.signs, identification.narrative;
                              max_rotations=identification.max_rotations, rng=rng)
            irnarrsign_draws[:, :, :, d] = irns
            Omega_draws[:, :, d] = Omega
        end

        # Zero + sign
        if identification isa ZeroSignRestriction
            irzs, Omega = irf_zero_sign(Phi, Sigma, hor, p,
                              identification.restrictions;
                              var_pos=identification.var_pos, rng=rng)
            irzerosign_draws[:, :, :, d] = irzs
            Omega_draws[:, :, d] = Omega
        end

        # Proxy
        if identification isa ProxyIdentification
            irs_p, b1, _ = compute_irf_proxy(Phi, Sigma, e_draws[:, :, d],
                                              identification.instrument, hor, p;
                                              proxy_end=identification.proxy_end)
            irproxy_draws[:, 1:size(irs_p, 1), 1, d] = irs_p'
        end

        # Heteroskedasticity
        if identification isa HeteroskedIdentification
            irhet, Ahet = compute_irf_heterosked(Phi, e_draws[:, :, d],
                              identification.regimes, hor, p)
            irheterosked_draws[:, :, :, d] = irhet
            Omega_draws[:, :, d] = Ahet
        end

        # ── Forecasts ──
        if exogenous_block === nothing
            # Standard forecasting for y-only VAR
            lags_data = copy(forecast_initval)
            for t in 1:fhor
                x_fwd = vcat(vec(reverse(lags_data, dims=1)'), forecast_xdata[t, :])
                y_pred = x_fwd' * Phi
                lags_data[1:end-1, :] = lags_data[2:end, :]
                lags_data[end, :] = y_pred
                yhatfut_no[t, :, d] = y_pred
            end

            lags_data = copy(forecast_initval)
            for t in 1:fhor
                x_fwd = vcat(vec(reverse(lags_data, dims=1)'), forecast_xdata[t, :])
                shock = (cholesky(Hermitian(Sigma)).L * randn(rng, ny))'
                y_pred = x_fwd' * Phi + shock
                lags_data[1:end-1, :] = lags_data[2:end, :]
                lags_data[end, :] = y_pred
                yhatfut_with[t, :, d] = y_pred
            end
        end
        # Exogenous block forecasting skipped for now (requires future z path)

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
        irheterosked_draws,
        e_draws, Omega_draws,
        yhatfut_no, yhatfut_with, yhatfut_cond,
        log_dnsty, ic, K, p, ny_total, hor, fhor
    )
end

"""
    classical_var(y, p; nboot, identification, hor, constant, rng)

Classical VAR with bootstrap confidence intervals.
"""
function classical_var(y::AbstractMatrix{<:Real}, p::Int;
                       nboot::Int=1000, identification::AbstractIdentification=CholeskyIdentification(),
                       hor::Int=24, constant::Bool=true,
                       rng::AbstractRNG=Random.default_rng())
    v = var_estimate(y, p; constant=constant)
    K = v.nvar
    T = v.nobs

    # Point estimate IRF
    ir_point = compute_irf(v.Phi[1:K*p, :], v.Sigma, hor)

    # Bootstrap
    ir_boot = zeros(K, hor, K, nboot)
    for b in 1:nboot
        # Resample residuals
        idx = rand(rng, 1:T, T)
        u_boot = v.residuals[idx, :]

        # Build bootstrap sample
        y_boot = zeros(T + p, K)
        start_idx = rand(rng, 1:T)
        if start_idx + p - 1 <= T
            y_boot[1:p, :] = v.Y[start_idx:start_idx+p-1, :]
        else
            y_boot[1:p, :] = v.Y[end-p+1:end, :]
        end

        for t in p+1:T+p
            x_t = vec(y_boot[t-1:-1:t-p, :]')
            if constant
                x_t = vcat(x_t, 1.0)
            end
            y_boot[t, :] = x_t' * v.Phi + u_boot[t-p, :]'
        end

        v_boot = var_estimate(y_boot, p; constant=constant)
        ir_boot[:, :, :, b] = compute_irf(v_boot.Phi[1:K*p, :], v_boot.Sigma, hor)
    end

    return ir_point, ir_boot, v
end
