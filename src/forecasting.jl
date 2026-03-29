#=
forecasting.jl — Unconditional and conditional forecasts
=#

"""
    forecast_unconditional(forecast_initval, forecast_xdata, Phi, Sigma, fhor, p;
                           rng, with_shocks=true)

Compute out‑of‑sample forecasts.

# Returns
`(frcst_no_shock, frcst_with_shocks)` each `fhor × K`.
"""
function forecast_unconditional(forecast_initval::AbstractMatrix,
                                forecast_xdata::AbstractMatrix,
                                Phi::AbstractMatrix, Sigma::AbstractMatrix,
                                fhor::Int, p::Int;
                                rng::AbstractRNG=Random.default_rng())
    K = size(Sigma, 1)
    Sigma_chol = cholesky(Hermitian(Sigma)).L

    frcst_no = zeros(fhor, K)
    frcst_with = zeros(fhor, K)

    # No shocks
    lags_data = copy(forecast_initval)
    for t in 1:fhor
        x = vcat(vec(reverse(lags_data, dims=1)'), forecast_xdata[t, :])
        y = x' * Phi
        lags_data[1:end-1, :] = lags_data[2:end, :]
        lags_data[end, :] = y
        frcst_no[t, :] = y
    end

    # With shocks
    lags_data = copy(forecast_initval)
    for t in 1:fhor
        x = vcat(vec(reverse(lags_data, dims=1)'), forecast_xdata[t, :])
        shock = (Sigma_chol * randn(rng, K))'
        y = x' * Phi + shock
        lags_data[1:end-1, :] = lags_data[2:end, :]
        lags_data[end, :] = y
        frcst_with[t, :] = y
    end

    return frcst_no, frcst_with
end

"""
    forecast_conditional(endo_path, endo_index, forecast_initval, forecast_xdata,
                         Phi, Sigma, fhor, p; rng)

Conditional forecasting: forecast path consistent with observed
endogenous variable path using Waggoner & Zha (1999).

# Arguments
- `endo_path`:     fhor × n_cond matrix of conditioned values.
- `endo_index`:    vector of indices of conditioned variables.
- `forecast_initval`: p × K initial lags.
- `Phi`:           coefficient matrix.
- `Sigma`:         covariance matrix.

# Returns
`(conditional_forecast, structural_shocks)` each `fhor × K`.
"""
function forecast_conditional(endo_path::AbstractMatrix,
                              endo_index::AbstractVector{Int},
                              forecast_initval::AbstractMatrix,
                              forecast_xdata::AbstractMatrix,
                              Phi::AbstractMatrix, Sigma::AbstractMatrix,
                              fhor::Int, p::Int;
                              rng::AbstractRNG=Random.default_rng())
    K = size(Sigma, 1)
    Ncondvar = length(endo_index)
    Nres = Ncondvar * fhor

    # Companion form
    F_comp = companion_form(Phi, K, p)
    Kp = K * p

    # No‑shock forecast
    frcst_no, _ = forecast_unconditional(forecast_initval, forecast_xdata,
                                          Phi, Sigma, fhor, p; rng=rng)

    # Deviation from no‑shock forecast
    err_mat = endo_path .- frcst_no[:, endo_index]
    err = reshape(err_mat', Nres)

    # Build response matrix R
    C = cholesky(Hermitian(Sigma)).L
    G = zeros(Kp, K); G[1:K, :] = I(K)

    R = zeros(K * fhor, K * fhor)
    for ff in 1:fhor
        tmp = zeros(K, K * ff)
        for hh in 0:ff-1
            Fh = Matrix{Float64}(I, Kp, Kp)
            for _ in 1:hh
                Fh = F_comp * Fh
            end
            block = G' * Fh * G * C
            tmp[:, (ff-hh-1)*K+1:(ff-hh)*K] = block
        end
        R[(ff-1)*K+1:ff*K, 1:K*ff] = tmp
    end

    # Select rows corresponding to conditioned variables
    index = zeros(Int, fhor, Ncondvar)
    for ff in 1:fhor
        index[ff, :] = (ff-1) * K .+ endo_index
    end
    idx_vec = vec(index')
    Rtilde = R[idx_vec, :]

    # Solve via SVD (Waggoner & Zha)
    U_svd, D_svd, V_svd = svd(Rtilde)
    V1 = V_svd[:, 1:Nres]
    V2 = V_svd[:, Nres+1:end]

    eps_vec = V1 * (Diagonal(D_svd[1:Nres]) \ (U_svd' * err)) +
              V2 * randn(rng, size(R, 1) - Nres)

    # Orthogonal shocks
    EPSn = reshape(eps_vec, K, fhor)
    EPS = C * EPSn

    # Recompute forecast with these specific shocks
    lags_data = copy(forecast_initval)
    cond_forecast = zeros(fhor, K)
    for t in 1:fhor
        x = vcat(vec(reverse(lags_data, dims=1)'), forecast_xdata[t, :])
        y = x' * Phi + EPS[:, t]'
        lags_data[1:end-1, :] = lags_data[2:end, :]
        lags_data[end, :] = y
        cond_forecast[t, :] = y
    end

    return cond_forecast, EPSn'
end
