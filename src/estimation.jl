#=
estimation.jl — OLS VAR estimation, lag selection, information criteria
=#

"""
    var_estimate(y, p; constant=true, trend=false, exogenous=nothing)

Estimate a reduced‑form VAR(p) by OLS.

# Arguments
- `y::Matrix{Float64}`: T × K data matrix.
- `p::Int`:              lag order.
- `constant::Bool`:      include intercept (default `true`).
- `trend::Bool`:         include linear time trend (default `false`).
- `exogenous::Union{Nothing,Matrix{Float64}}`: T × n_exo exogenous regressors.

# Returns
A `VAREstimate` struct.
"""
function var_estimate(y::AbstractMatrix{<:Real}, p::Int;
                      constant::Bool=true, trend::Bool=false,
                      exogenous::Union{Nothing,AbstractMatrix{<:Real}}=nothing,
                      regularization::Symbol=:none,
                      lambda::Float64=0.0,
                      alpha::Float64=1.0)
    p > 0 || throw(ArgumentError("lag order p must be positive"))
    T, K = size(y)
    T > K || throw(ArgumentError("more variables than observations"))

    # Build regressor matrix: [y_{t-1} … y_{t-p}  constant  trend  exogenous]
    Y = y[p+1:T, :]
    Teff = T - p
    X = lagmatrix(y, p; constant=constant, trend=trend)

    nexo = 0
    if exogenous !== nothing
        nexo = size(exogenous, 2)
        size(exogenous, 1) >= T || throw(ArgumentError("exogenous must have at least T rows"))
        X = hcat(X, exogenous[p+1:T, :])
    end

    nk = size(X, 2)

    if regularization ∉ (:none, :ridge, :lasso, :elastic_net)
        throw(ArgumentError("regularization must be :none, :ridge, :lasso, or :elastic_net"))
    end

    if regularization == :none || lambda == 0.0
        # OLS via SVD
        Phi, residuals, xxi = ols_svd(Y, X)
    elseif regularization == :ridge
        Phi, residuals, xxi = _ridge_estimate(Y, X, lambda)
    elseif regularization == :lasso
        Phi, residuals, xxi = _lasso_estimate(Y, X, lambda)
    elseif regularization == :elastic_net
        Phi, residuals, xxi = _elastic_net_estimate(Y, X, lambda, alpha)
    end

    # Covariance: use dfₑ = T - (Kp + nx)
    Sigma = residuals' * residuals / (Teff - nk)

    return VAREstimate(Float64.(y), Y, X, Phi, Sigma, residuals, xxi,
                       Teff, K, p, constant, trend, nexo)
end

"""
    var_lagorder(y, pbar; ic="bic", verbose=true)

Select optimal VAR lag length by information criterion.

# Arguments
- `y`:    T × K data matrix.
- `pbar`: maximum lag to consider.
- `ic`:   `"aic"`, `"bic"`, `"aicc"`, or `"hqc"`.

# Returns
Optimal lag order (Int).
"""
function var_lagorder(y::AbstractMatrix{<:Real}, pbar::Int; ic::String="bic", verbose::Bool=true)
    T, K = size(y)
    t = T - pbar
    IC_vals = zeros(pbar)
    Y = y[pbar+1:T, :]
    for p in 1:pbar
        X = ones(t, 1)
        for i in 1:p
            X = hcat(X, y[pbar+1-i:T-i, :])
        end
        β = X \ Y
        u = Y - X * β
        Σ = u' * u / t
        if ic == "aic"
            IC_vals[p] = log(det(Σ)) + 2 * p * K^2 / t
        elseif ic == "bic"
            IC_vals[p] = log(det(Σ)) + K^2 * p * log(t) / t
        elseif ic == "aicc"
            b = t / (t - (p * K + K + 1))
            IC_vals[p] = t * (log(det(Σ)) + K) + 2 * b * (K^2 * p + K * (K + 1) / 2)
        elseif ic == "hqc"
            IC_vals[p] = log(det(Σ)) + 2 * log(log(t)) * K^2 * p / t
        else
            throw(ArgumentError("ic must be aic, bic, aicc or hqc"))
        end
    end
    best = argmin(IC_vals)
    verbose && @info "Using $ic: best lag‑length = $best"
    return best
end

"""
    information_criteria(var::VAREstimate)

Compute AIC, BIC, HQIC for a `VAREstimate`.
"""
function information_criteria(v::VAREstimate)
    T = v.nobs
    K = v.nvar
    nk = size(v.X, 2)
    S = v.Sigma
    E = v.residuals
    iS = pinv(S)
    llf = -(T * K / 2) * (1 + log(2π)) - T / 2 * log(det(S)) -
          0.5 * tr(iS * E' * E)
    aic  = -2 * llf / T + 2 * nk / T
    bic  = -2 * llf / T + nk * log(T) / T
    hqic = -2 * llf / T + 2 * nk * log(log(T)) / T
    return InfoCriteria(aic, bic, hqic)
end

"""
    rfvar3(ydata, lags, xdata, breaks, lambda, mu)

Reduced‑form VAR with dummy‑observation priors for sum‑of‑coefficients
and co‑persistence (port of Sims's rfvar3.m).

Returns `(B, u, xxi, y_out, X_out)`.
"""
function rfvar3(ydata::AbstractMatrix, lags::Int, xdata::AbstractMatrix,
                breaks::Vector{Int}, lambda::Float64, mu::Float64)
    T, nvar = size(ydata)
    nx = size(xdata, 2)

    # Build sample indices respecting breaks
    all_breaks = vcat(0, breaks, T)
    smpl = Int[]
    for nb in 1:(length(all_breaks)-1)
        append!(smpl, (all_breaks[nb]+lags+1):all_breaks[nb+1])
    end
    Tsmpl = length(smpl)

    # Build X = [y_{t-1} … y_{t-p}  xdata_t]
    X = Matrix{Float64}(undef, Tsmpl, nvar * lags + nx)
    for (idx, t) in enumerate(smpl)
        for lag in 1:lags
            X[idx, (lag-1)*nvar+1:lag*nvar] = ydata[t-lag, :]
        end
        X[idx, nvar*lags+1:end] = xdata[t, :]
    end
    y = ydata[smpl, :]

    # Persistence dummies
    if lambda != 0 || mu > 0
        ybar = mean(ydata[1:lags, :], dims=1)
        xbar = nx > 0 ? mean(xdata[1:lags, :], dims=1) : zeros(1, 0)
        if lambda != 0
            abslam = abs(lambda)
            if lambda > 0
                xdum = abslam * hcat(repeat(ybar, 1, lags), xbar)
            else
                xdum = abslam * hcat(repeat(ybar, 1, lags), zeros(size(xbar)))
            end
            ydum = abslam * ybar
            y = vcat(y, ydum)
            X = vcat(X, xdum)
        end
        if mu > 0
            xdum_mu = hcat(repeat(diagm(vec(ybar)), 1, lags), zeros(nvar, nx)) .* mu
            ydum_mu = mu * diagm(vec(ybar))
            X = vcat(X, xdum_mu)
            y = vcat(y, ydum_mu)
        end
    end

    # OLS via SVD
    B, u, xxi = ols_svd(y, X)
    return B, u, xxi, y, X
end

# ─── Regularization helpers ─────────────────────────────────────────────────────

"""
    _ridge_estimate(Y, X, lambda) → (Phi, residuals, XXi)

Ridge (L₂) regression: Phi = (X'X + λI)⁻¹ X'Y.
"""
function _ridge_estimate(Y::AbstractMatrix, X::AbstractMatrix, lambda::Float64)
    nk = size(X, 2)
    XtX_ridge = X' * X + lambda * I(nk)
    xxi = inv(XtX_ridge)
    Phi = xxi * (X' * Y)
    residuals = Y - X * Phi
    return Phi, residuals, xxi
end

"""
    _soft_threshold(z, λ)

Soft‑thresholding operator: sign(z) max(|z| − λ, 0).
"""
_soft_threshold(z::Real, λ::Real) = sign(z) * max(abs(z) - λ, 0.0)

"""
    _lasso_estimate(Y, X, lambda; max_iter=1000, tol=1e-6) → (Phi, residuals, XXi)

Lasso (L₁) regression via coordinate descent (Friedman et al. 2010).
Solves column‑by‑column: min ‖Yₖ − Xβₖ‖² + λ‖βₖ‖₁.
"""
function _lasso_estimate(Y::AbstractMatrix, X::AbstractMatrix, lambda::Float64;
                         max_iter::Int=1000, tol::Float64=1e-6)
    T, nk = size(X)
    K = size(Y, 2)
    Phi = zeros(nk, K)

    # Pre‑compute X'X diagonal and X'Y for speed
    XtX_diag = vec(sum(X .^ 2, dims=1))

    for k in 1:K
        beta = zeros(nk)
        y_k = Y[:, k]

        for iter in 1:max_iter
            beta_old = copy(beta)
            for j in 1:nk
                # Partial residual excluding predictor j
                r_j = y_k - X * beta + X[:, j] * beta[j]
                z_j = dot(X[:, j], r_j)
                beta[j] = _soft_threshold(z_j, lambda) / XtX_diag[j]
            end
            if maximum(abs.(beta - beta_old)) < tol
                break
            end
        end
        Phi[:, k] = beta
    end

    residuals = Y - X * Phi
    xxi = pinv(X' * X)   # pseudo‑inverse for informational purposes
    return Phi, residuals, xxi
end

"""
    _elastic_net_estimate(Y, X, lambda, alpha; max_iter=1000, tol=1e-6)

Elastic net regression via coordinate descent.
Penalty: α λ ‖β‖₁ + (1−α) λ/2 ‖β‖².
"""
function _elastic_net_estimate(Y::AbstractMatrix, X::AbstractMatrix,
                               lambda::Float64, alpha::Float64;
                               max_iter::Int=1000, tol::Float64=1e-6)
    T, nk = size(X)
    K = size(Y, 2)
    Phi = zeros(nk, K)

    XtX_diag = vec(sum(X .^ 2, dims=1))
    l1_pen = alpha * lambda
    l2_pen = (1.0 - alpha) * lambda

    for k in 1:K
        beta = zeros(nk)
        y_k = Y[:, k]

        for iter in 1:max_iter
            beta_old = copy(beta)
            for j in 1:nk
                r_j = y_k - X * beta + X[:, j] * beta[j]
                z_j = dot(X[:, j], r_j)
                beta[j] = _soft_threshold(z_j, l1_pen) / (XtX_diag[j] + l2_pen)
            end
            if maximum(abs.(beta - beta_old)) < tol
                break
            end
        end
        Phi[:, k] = beta
    end

    residuals = Y - X * Phi
    xxi = pinv(X' * X)
    return Phi, residuals, xxi
end
