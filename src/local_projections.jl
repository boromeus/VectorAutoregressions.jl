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
            if identification == :proxy && proxy !== nothing
                proxy_pos = K * ph + (controls !== nothing ? size(controls, 2) * (clags + 1) : 0) + 1
                Omega_proxy = beta[proxy_pos:proxy_pos+ns-1, :]'
                irf_proxy_impact = Omega_proxy
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
