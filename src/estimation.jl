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
                      exogenous::Union{Nothing,AbstractMatrix{<:Real}}=nothing)
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

    # OLS via SVD
    Phi, residuals, xxi = ols_svd(Y, X)

    # Covariance: use dfₑ = T - (Kp + nx)
    nk = size(X, 2)
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
