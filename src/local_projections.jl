#=
local_projections.jl — Local projection IRFs and direct forecasts
Port of MATLAB directmethods.m + refactor of existing LP code
=#

"""
    lp_irf(y, p, H; constant=true, identification=:cholesky,
           Q=nothing, controls=nothing, proxy=nothing,
           robust_se=true, conf_level=0.90, clags=0)

Compute local‑projection impulse responses.

# Arguments
- `y`:              T × K data matrix.
- `p`:              lag‑length (scalar or Vector{Int} of length H).
- `H`:              IRF horizon.
- `identification`: `:cholesky`, `:reduced_form`, or `:proxy`.
- `Q`:              K × K rotation matrix (default I).
- `controls`:       T × n_c control variables (optional).
- `proxy`:          T × n_s proxy variables (optional, used with `:proxy`).
- `robust_se`:      use Newey–West standard errors (default `true`).
- `conf_level`:     confidence level for CIs (default 0.90).

# Returns
`LPResult`.
"""
function lp_irf(y::AbstractMatrix, p::Union{Int,Vector{Int}}, H::Int;
                constant::Bool=true, identification::Symbol=:cholesky,
                Q::Union{Nothing,AbstractMatrix}=nothing,
                controls::Union{Nothing,AbstractMatrix}=nothing,
                proxy::Union{Nothing,AbstractMatrix}=nothing,
                robust_se::Bool=true,
                conf_level::Float64=0.90,
                clags::Int=0)
    T, K = size(y)
    pvec = p isa Int ? fill(p, H) : p
    nx = constant ? 1 : 0
    ns = proxy !== nothing ? size(proxy, 2) : 0

    if Q === nothing
        Q = Matrix{Float64}(I, K, K)
    end

    alpha = 1 - conf_level
    talpha = abs(quantile(TDist(T - K - 1), alpha / 2))

    # Build base regressor for the first-period OLS
    pmax = maximum(pvec)

    # Storage
    irf_out = zeros(H + 1, K^2)
    lower_out = zeros(H + 1, K^2)
    upper_out = zeros(H + 1, K^2)
    std_out = zeros(H + 1, K^2)
    Omega = Matrix{Float64}(I, K, K)  # Impact matrix, set at hh=0

    for hh in 0:H
        ph = hh == 0 ? pmax : pvec[hh]

        # LHS: y_{t+h}
        ytmp = y[ph + hh + 1:T, :]
        Teff = size(ytmp, 1)

        # RHS
        Xr = _build_lp_rhs(y, T, K, ph, hh, constant, controls, proxy, clags)
        positions_nylags = 1:K*ph
        nk = size(Xr, 2)

        if identification == :iv && proxy !== nothing
            # ── LP‑IV: Two‑Stage Least Squares ──
            # Instrument the first K columns of lag‑1 (endogenous) with proxy
            n_endo = K  # instrument all K variables at lag 1
            endo_cols = 1:K
            exo_cols = (K + 1):nk
            X_endo = Xr[:, endo_cols]
            X_exo = Xr[:, exo_cols]
            Z = proxy[ph + hh + 1:T, :]  # instruments

            beta, se_beta = _tsls_estimate(ytmp, X_endo, X_exo, Z,
                                            robust_se, max(ph + hh + 1, 1))

            if hh == 0
                resid = ytmp - hcat(X_endo, Xr[:, exo_cols]) * beta
                Sigma_u = (resid' * resid) / Teff
                Omega = cholesky(Hermitian(Sigma_u)).L * Q
                irf_out[1, :] = vec(Omega')
            else
                Phi_h = beta[1:K*ph, :]
                ir = compute_irf(Phi_h, Matrix{Float64}(I, K, K), 2; Omega=Omega)
                irf_out[hh + 1, :] = vec(ir[:, 2, :]')
            end

            if hh == 0
                std_out[1, :] .= 0.0
            else
                se_Phi = se_beta[1:K*ph, :]
                ir_up = compute_irf(beta[1:K*ph, :] .+ talpha .* se_Phi,
                                    Matrix{Float64}(I, K, K), 2; Omega=Omega)
                ir_lo = compute_irf(beta[1:K*ph, :] .- talpha .* se_Phi,
                                    Matrix{Float64}(I, K, K), 2; Omega=Omega)
                std_out[hh + 1, :] = abs.(vec(ir_up[:, 2, :]') .-
                                          irf_out[hh + 1, :]) ./ talpha
            end
        else
            # ── Standard OLS identification (Cholesky / reduced_form) ──
            # OLS
            beta = Xr \ ytmp
            resid = ytmp - Xr * beta

            if hh == 0
                # Impact: get Σ_u and Cholesky
                Sigma_u = (resid' * resid) / Teff
                try
                    Omega = cholesky(Hermitian(Sigma_u)).L * Q
                catch
                    Omega = cholesky(Hermitian(cov(resid))).L * Q
                end
                irf_out[1, :] = vec(Omega')
            else
                # hh > 0: LP-IRF via one-step ahead propagation
                Phi_h = beta[1:K*ph, :]
                ir = compute_irf(Phi_h, Matrix{Float64}(I, K, K), 2; Omega=Omega)
                irf_out[hh + 1, :] = vec(ir[:, 2, :]')
            end

            # SE via Newey-West
            if robust_se
                se_beta = _newey_west_se(ytmp, Xr, beta, max(ph + hh + 1, 1))
            else
                se_beta = _ols_se(ytmp, Xr, beta)
            end

            if hh == 0
                # SE on impact comes from Omega uncertainty
                std_out[1, :] = vec(Omega') .* 0  # impact is exact given Omega
            else
                se_Phi = se_beta[1:K*ph, :]
                ir_up = compute_irf(beta[1:K*ph, :] .+ talpha .* se_Phi,
                                    Matrix{Float64}(I, K, K), 2; Omega=Omega)
                ir_lo = compute_irf(beta[1:K*ph, :] .- talpha .* se_Phi,
                                    Matrix{Float64}(I, K, K), 2; Omega=Omega)
                upper_point = vec(ir_up[:, 2, :]')
                lower_point = vec(ir_lo[:, 2, :]')
                std_out[hh + 1, :] = abs.(upper_point .- irf_out[hh + 1, :]) ./ talpha
            end
        end

        lower_out[hh + 1, :] = irf_out[hh + 1, :] .- talpha .* std_out[hh + 1, :]
        upper_out[hh + 1, :] = irf_out[hh + 1, :] .+ talpha .* std_out[hh + 1, :]
    end

    return LPResult(irf_out, lower_out, upper_out, std_out, H, conf_level)
end

"""
    lp_lagorder(y, pbar, H, ic)

Select the LP lag‑order at each horizon by information criterion.
"""
function lp_lagorder(y::AbstractMatrix, pbar::Int, H::Int, ic::String)
    T, K = size(y)
    t = T - pbar
    vIC = zeros(Int, H)

    for h in 1:H
        IC = zeros(pbar)
        for m in 1:pbar
            # Build X with m lags and constant
            ytmp = y[pbar + h + 1:T, :]
            Teff = size(ytmp, 1)
            X = ones(Teff, 1)
            for lag in 1:m
                X = hcat(X, y[pbar + 1 - lag:T - h - lag, :])
            end
            yt = y[pbar:T - h, :]  # RHS variable of interest
            yt = yt[1:Teff, :]

            Mx = I - X / (X' * X) * X'
            beta = (yt' * Mx * yt) \ (yt' * Mx * ytmp)
            u = Mx * ytmp - Mx * yt * beta
            Sigma = u' * u / t

            IC[m] = _compute_ic(Sigma, K, m, t, ic)
        end
        vIC[h] = argmin(IC)
    end
    return vIC
end

# ─── Internal helpers ───────────────────────────────────────────────────────────

function _build_lp_rhs(y, T, K, ph, hh, constant, controls, proxy, clags)
    Teff = T - ph - hh
    X = Matrix{Float64}(undef, Teff, 0)

    # Lags of y
    for lag in 1:ph
        X = hcat(X, y[ph + 1 - lag:T - hh - lag, :])
    end

    # Controls
    if controls !== nothing
        for cl in 0:clags
            X = hcat(X, controls[ph + 1 - cl:T - hh - cl, :])
        end
    end

    # Proxy
    if proxy !== nothing
        X = hcat(X, proxy[ph + 1:T - hh, :])
    end

    # Constant
    if constant
        X = hcat(X, ones(Teff))
    end

    return X
end

function _newey_west_se(y, X, beta, bandwidth)
    T, K = size(y)
    nk = size(X, 2)
    u = y - X * beta
    M = bandwidth

    # HAC estimator
    u_mean = mean(u, dims=1)
    u0 = u .- u_mean
    S = (u0' * u0) / T

    for j in 1:M
        Rj = (u0[1:end-j, :]' * u0[j+1:end, :]) / T
        S = S .+ (1 - j / (M + 1)) * (Rj + Rj')
    end

    # SE of beta
    XXi = inv(X' * X)
    V = kron(S, XXi)
    se_full = reshape(sqrt.(max.(diag(V), 0.0)), nk, K)
    return se_full
end

function _ols_se(y, X, beta)
    T, K = size(y)
    nk = size(X, 2)
    u = y - X * beta
    Sigma_u = (u' * u) / (T - nk)
    XXi = inv(X' * X)
    V = kron(Sigma_u, XXi)
    se_full = reshape(sqrt.(max.(diag(V), 0.0)), nk, K)
    return se_full
end

function _compute_ic(Sigma, K, p, t, ic)
    ld = log(det(Sigma))
    if ic == "aic"
        return ld + 2 * p * K^2 / t
    elseif ic == "bic"
        return ld + K^2 * p * log(t) / t
    elseif ic == "aicc"
        b = t / (t - (p * K + K + 1))
        return t * (ld + K) + 2 * b * (K^2 * p + K * (K + 1) / 2)
    elseif ic == "hqc"
        return ld + 2 * log(log(t)) * K^2 * p / t
    else
        error("ic must be aic, bic, aicc or hqc")
    end
end

# ─── Two‑Stage Least Squares for LP‑IV ─────────────────────────────────────────

"""
    _tsls_estimate(Y, X_endo, X_exo, Z, robust_se, bandwidth)

Two‑stage least squares with HAC or OLS standard errors.

Stage 1:  X_endo = [Z  X_exo] γ + v   →   X̂_endo
Stage 2:  Y      = [X̂_endo  X_exo] β + e

Returns `(beta, se)` where `beta` is nk × K and `se` is nk × K.
"""
function _tsls_estimate(Y::AbstractMatrix, X_endo::AbstractMatrix,
                        X_exo::AbstractMatrix, Z::AbstractMatrix,
                        robust_se::Bool, bandwidth::Int)
    T, K = size(Y)
    n_endo = size(X_endo, 2)
    n_exo = size(X_exo, 2)

    # Align instrument length to match Y
    Tz = size(Z, 1)
    if Tz > T
        Z = Z[end-T+1:end, :]
    elseif Tz < T
        throw(ArgumentError("proxy has fewer observations than needed"))
    end

    # Stage 1: project endogenous onto instruments + exogenous
    W1 = hcat(Z, X_exo)
    gamma = W1 \ X_endo
    X_hat = W1 * gamma

    # Stage 2
    X_full = hcat(X_hat, X_exo)
    beta = X_full \ Y

    # SE
    # Residuals use actual X, not fitted
    X_actual = hcat(X_endo, X_exo)
    resid = Y - X_actual * beta
    nk = size(X_full, 2)

    if robust_se
        se = _newey_west_se(Y, X_full, beta, bandwidth)
    else
        se = _ols_se(Y, X_full, beta)
    end

    return beta, se
end

# ─── Bayesian Local Projections ──────────────────────────────────────────────────

"""
    lp_bayesian(y, p, H; prior=MinnesotaPrior(), K=1000, hor=nothing,
                constant=true, conf_level=0.90, rng=Random.default_rng())

Bayesian local projections with a Minnesota‑style prior.

At each horizon h = 0, …, H the LP regression

    yₜ₊ₕ = Xₜ βₕ + εₜ₊ₕ

is estimated with a Normal‑Inverse‑Wishart conjugate posterior
formed by augmenting the data with Minnesota dummy observations.

# Arguments
- `y`:          T × K data.
- `p`:          lag length.
- `H`:          maximum LP horizon.
- `prior`:      `MinnesotaPrior` or `FlatPrior`.
- `K`:          number of posterior draws.
- `constant`:   include intercept.
- `conf_level`: HPD coverage.
- `rng`:        random number generator.

# Returns
`LPBayesianResult`.
"""
function lp_bayesian(y::AbstractMatrix{<:Real}, p::Int, H::Int;
                     prior::AbstractPrior=MinnesotaPrior(),
                     K::Int=1000, constant::Bool=true,
                     conf_level::Float64=0.90,
                     rng::AbstractRNG=Random.default_rng())
    T, Kv = size(y)
    nk = Kv * p + (constant ? 1 : 0)

    # Storage for all draws across horizons
    Phi_all_draws = zeros(nk, Kv, H + 1, K)
    Sigma_all_draws = zeros(Kv, Kv, H + 1, K)
    irf_median = zeros(Kv, H + 1, Kv)
    irf_lower = zeros(Kv, H + 1, Kv)
    irf_upper = zeros(Kv, H + 1, Kv)

    alpha = 1 - conf_level
    Omega = Matrix{Float64}(I, Kv, Kv)

    for hh in 0:H
        # Build data for this horizon
        Yh = y[p + hh + 1:T, :]
        Teff = size(Yh, 1)
        Xh = Matrix{Float64}(undef, Teff, 0)

        # Lags
        for lag in 1:p
            Xh = hcat(Xh, y[p + 1 - lag:T - hh - lag, :])
        end
        if constant
            Xh = hcat(Xh, ones(Teff))
        end

        # Build Minnesota prior precision and augment
        if prior isa MinnesotaPrior
            mu_pr, sig_pr, delta_pr = get_prior_moments(y, p)
            tau = prior.tau
            decay = prior.decay

            # Construct diagonal prior precision for β (nk × Kv)
            # β is organized as [lag1_var1, lag1_var2, ..., lagp_varK, (constant)]
            # Prior variance for coeff of variable j at lag l in equation i:
            #   V_{l,j,i} = (τ / l^decay)^2 * (σ_i / σ_j)^2
            Omega_prior_diag = zeros(nk)
            for l in 1:p
                for j in 1:Kv
                    idx = (l - 1) * Kv + j
                    # Use average sigma ratio for the prior precision
                    Omega_prior_diag[idx] = (tau / l^decay)^2 * mean(sig_pr)^2 / sig_pr[j]^2
                end
            end
            if constant
                Omega_prior_diag[nk] = (tau * 10.0)^2  # loose prior on constant
            end
            # Prior precision = 1/variance
            Omega_prior_inv = Diagonal(1.0 ./ (Omega_prior_diag .+ 1e-12))

            # Posterior with Normal prior on β:
            # Ω_post = X'X + Ω₀⁻¹,  β_post = Ω_post⁻¹ X'Y
            Xh_Xh = Xh' * Xh + Omega_prior_inv
            Xh_Yh = Xh' * Yh  # prior mean is 0
            Phi_aug = Xh_Xh \ Xh_Yh
            resid_aug = Yh - Xh * Phi_aug
            S_a = resid_aug' * resid_aug
            xxi_a = inv(Hermitian(Xh_Xh + 1e-10 * I(nk)))
            Phi_ols = Phi_aug
            df_post = max(Teff, Kv + 1)
        else
            Phi_ols, resid_a, xxi_a = ols_svd(Yh, Xh)
            S_a = resid_a' * resid_a
            df_post = max(Teff - nk, Kv + 1)
        end

        # Draw K posterior samples
        for d in 1:K
            Sigma_d = rand_inverse_wishart(max(df_post, Kv + 1),
                                           Hermitian(S_a); rng=rng)
            # Draw Phi | Sigma
            L_sig = cholesky(Hermitian(Sigma_d)).L
            L_xxi = cholesky(Hermitian(xxi_a + 1e-10 * I(nk))).L
            Phi_d = Phi_ols + L_xxi * randn(rng, nk, Kv) * L_sig'

            Phi_all_draws[:, :, hh + 1, d] = Phi_d
            Sigma_all_draws[:, :, hh + 1, d] = Sigma_d
        end

        # Compute IRFs from draws
        if hh == 0
            # Impact: Cholesky of median Sigma
            Sigma_med = median(Sigma_all_draws[:, :, 1, :]; dims=3)[:, :, 1]
            Omega = cholesky(Hermitian(Sigma_med)).L
        end
    end

    # Compute IRFs across posterior draws
    ir_draws = zeros(Kv, H + 1, Kv, K)
    for d in 1:K
        # Impact from this draw's Sigma
        Sigma_d0 = Sigma_all_draws[:, :, 1, d]
        Omega_d = cholesky(Hermitian(Sigma_d0)).L

        ir_draws[:, 1, :, d] = Omega_d

        for hh in 1:H
            Phi_h = Phi_all_draws[1:Kv*p, :, hh + 1, d]
            ir_h = compute_irf(Phi_h, Matrix{Float64}(I, Kv, Kv), 2;
                               Omega=Omega_d)
            ir_draws[:, hh + 1, :, d] = ir_h[:, 2, :]
        end
    end

    # Compute pointwise median and bands
    for k in 1:Kv, h in 1:(H + 1), j in 1:Kv
        vals = sort(ir_draws[k, h, j, :])
        lo_idx = max(1, round(Int, alpha / 2 * K))
        hi_idx = min(K, round(Int, (1 - alpha / 2) * K))
        irf_median[k, h, j] = median(vals)
        irf_lower[k, h, j] = vals[lo_idx]
        irf_upper[k, h, j] = vals[hi_idx]
    end

    return LPBayesianResult(irf_median, irf_lower, irf_upper,
                            Phi_all_draws, Sigma_all_draws,
                            H, conf_level, K)
end

"""
    lp_marginal_likelihood(y, p, H, prior::MinnesotaPrior; constant=true)

Compute log marginal likelihood for Bayesian LP across horizons.
Uses the Minnesota prior precision directly.
"""
function lp_marginal_likelihood(y::AbstractMatrix, p::Int, H::Int,
                                prior::MinnesotaPrior; constant::Bool=true)
    T, K = size(y)
    nk = K * p + (constant ? 1 : 0)

    mu_pr, sig_pr, delta_pr = get_prior_moments(y, p)
    tau = prior.tau
    decay = prior.decay

    # Build Minnesota prior precision (same as in lp_bayesian)
    Omega_prior_diag = zeros(nk)
    for l in 1:p
        for j in 1:K
            idx = (l - 1) * K + j
            Omega_prior_diag[idx] = (tau / l^decay)^2 * mean(sig_pr)^2 / sig_pr[j]^2
        end
    end
    if constant
        Omega_prior_diag[nk] = (tau * 10.0)^2
    end
    Omega_prior_inv = Diagonal(1.0 ./ (Omega_prior_diag .+ 1e-12))

    total_ml = 0.0
    for hh in 0:H
        Yh = y[p + hh + 1:T, :]
        Teff = size(Yh, 1)
        Xh = Matrix{Float64}(undef, Teff, 0)
        for lag in 1:p
            Xh = hcat(Xh, y[p + 1 - lag:T - hh - lag, :])
        end
        if constant
            Xh = hcat(Xh, ones(Teff))
        end

        # Prior moments
        xxi_d = inv(Hermitian(Omega_prior_inv + 1e-10 * I(nk)))
        Sd = zeros(K, K) + I(K) * 1e-6  # prior scale
        df_d = K + 2  # minimal prior df

        # Posterior
        Xh_Xh = Xh' * Xh + Omega_prior_inv
        Phi_post = Xh_Xh \ (Xh' * Yh)
        resid_post = Yh - Xh * Phi_post
        Sa = resid_post' * resid_post
        xxi_a = inv(Hermitian(Xh_Xh + 1e-10 * I(nk)))
        df_a = max(Teff, K + 1)

        post_int = matrictint(Sa + I(K) * 1e-6, max(df_a, K + 1), xxi_a)
        prior_int = matrictint(Sd, max(df_d, K + 1), xxi_d)
        lik_nobs = df_a - df_d
        total_ml += post_int - prior_int - 0.5 * K * max(lik_nobs, 1) * log(2π)
    end

    return total_ml
end
