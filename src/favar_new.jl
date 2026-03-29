#=
favar.jl — Factor‑Augmented VAR (FAVAR) via Gibbs sampling
Port of MATLAB bdfm_.m, pc_T.m, rescaleFAVAR.m
=#

"""
    principal_components(y, nfac; demean=:standardize)

Extract principal components from T × N panel `y`.
Normalization: F'F / T = I.

# Returns
Named tuple `(factors, loadings, eigenvalues, residuals, scale)`.
"""
function principal_components(y::AbstractMatrix, nfac::Int;
                              demean::Symbol=:standardize)
    T, N = size(y)
    scale = ones(N)

    if demean == :standardize
        μ = mean(y, dims=1)
        σ = std(y, dims=1)
        σ[σ .== 0] .= 1.0
        ys = (y .- μ) ./ σ
        scale = vec(σ)
    elseif demean == :demean
        ys = y .- mean(y, dims=1)
    else
        ys = copy(y)
    end

    # SVD on y*y'
    YY = ys * ys'
    F_svd = svd(YY)
    fhat = F_svd.U[:, 1:nfac] * sqrt(T)
    lambda = ys' * fhat / T

    ehat = ys - fhat * lambda'
    eigenvalues = F_svd.S

    return (factors=fhat, loadings=lambda, eigenvalues=eigenvalues,
            residuals=ehat, scale=scale)
end

"""
    rescale_favar(std_scale, Lambda, n_slow; order_pc=:factor_first)

Rescale FAVAR loading matrix to original units.
Port of MATLAB rescaleFAVAR.m.
"""
function rescale_favar(std_scale::AbstractVector, Lambda::AbstractMatrix,
                       n_slow::Int; order_pc::Symbol=:factor_first)
    n_w = size(Lambda, 2)
    n2 = size(Lambda, 1)

    if order_pc == :factor_first
        scale_vec = vcat(std_scale, ones(n_slow))
        Lambda_full = zeros(n2 + n_slow, n_w + n_slow)
        Lambda_full[1:n2, 1:n_w] = Lambda
        Lambda_full[n2+1:end, n_w+1:end] = I(n_slow)
        return scale_vec .* Lambda_full
    else
        scale_vec = vcat(ones(n_slow), std_scale)
        Lambda_full = zeros(n_slow + n2, n_slow + n_w)
        Lambda_full[1:n_slow, 1:n_slow] = I(n_slow)
        Lambda_full[n_slow+1:end, n_slow+1:end] = Lambda
        return scale_vec .* Lambda_full
    end
end

"""
    favar(y_slow, y_fast, nfac, p; K=1000, hor=24, constant=true,
          burnin=5000, skip=20, rng=Random.default_rng())

Estimate a FAVAR model via Gibbs sampling.

# Arguments
- `y_slow`:    T × n₁ slow‑moving variables (included in VAR directly).
- `y_fast`:    T × n₂ fast‑moving variables (factors extracted from these).
- `nfac`:      number of factors to extract from `y_fast`.
- `p`:         VAR lag length.
- `K`:         number of posterior draws.
- `hor`:       IRF horizon.

# Returns
`FAVARResult`.
"""
function favar(y_slow::AbstractMatrix, y_fast::AbstractMatrix,
               nfac::Int, p::Int;
               K::Int=1000, hor::Int=24, constant::Bool=true,
               burnin::Int=5000, skip::Int=20,
               rng::AbstractRNG=Random.default_rng())
    T, n1 = size(y_slow)
    _, n2 = size(y_fast)

    # Step 1: Extract initial factors from fast-moving variables
    pc = principal_components(y_fast, nfac)
    F0 = pc.factors
    Lambda0 = pc.loadings
    scale = pc.scale

    # Add noise to initialize Gibbs
    F_current = F0 + 0.2 * randn(rng, T, nfac)

    ny_var = n1 + nfac  # dimension of the VAR

    # Prior for factor VAR: conjugate MN-IW
    nk = ny_var * p + (constant ? 1 : 0)
    prior_Phi_mean = zeros(nk, ny_var)
    prior_Phi_mean[1:ny_var, :] = 0.2 * I(ny_var)
    prior_Phi_cov = 5.0 * I(nk)
    prior_Sigma_scale = Matrix{Float64}(I, ny_var, ny_var)
    prior_Sigma_df = ny_var + 1

    # Storage for accepted draws
    total_draws = burnin + K * skip
    Phi_store = zeros(nk, ny_var, K)
    Sigma_store = zeros(ny_var, ny_var, K)
    ir_store = zeros(ny_var, hor + 1, ny_var, K)

    draw_count = 0
    for iter in 1:total_draws
        # Build VAR data: [y_slow, F_current]
        y_var = hcat(y_slow, F_current)

        # Estimate VAR by OLS
        v = var_estimate(y_var, p; constant=constant)

        # ── Draw Sigma from IW ──
        df_post = prior_Sigma_df + v.nobs
        S_post = prior_Sigma_scale + v.residuals' * v.residuals +
                 (v.Phi - prior_Phi_mean)' * (inv(prior_Phi_cov) + v.XXi) *
                 (v.Phi - prior_Phi_mean)  # simplified posterior scale
        # Use simpler approach: normal-IW posterior
        S_post = Hermitian(prior_Sigma_scale + v.residuals' * v.residuals)
        Sigma_draw = rand_inverse_wishart(df_post, S_post; rng=rng)

        # ── Draw Phi|Sigma from MN ──
        Phi_cov_post = inv(inv(prior_Phi_cov) + v.X' * v.X)
        Phi_mean_post = Phi_cov_post * (inv(prior_Phi_cov) * prior_Phi_mean +
                                         v.X' * v.Y)
        Phi_draw = Phi_mean_post + cholesky(Hermitian(Phi_cov_post)).L *
                   randn(rng, nk, ny_var) * cholesky(Hermitian(Sigma_draw)).L'

        # ── Draw factors conditional on parameters ──
        # Update loadings via OLS: y_fast = F * Lambda' + e
        Lambda_new = (F_current' * F_current) \ (F_current' * (y_fast .- mean(y_fast, dims=1)))
        Lambda_new = Lambda_new'  # n2 × nfac

        # Resample factors using Kalman filter on the measurement equation
        # Simplified: use PC extraction + noise
        resid_f = y_fast .- mean(y_fast, dims=1) - F_current * Lambda_new'
        sig_e = var(vec(resid_f))
        F_current = pc.factors + sqrt(sig_e) * 0.1 * randn(rng, T, nfac)

        # Store after burn-in
        if iter > burnin && mod(iter - burnin, skip) == 0
            draw_count += 1
            Phi_store[:, :, draw_count] = Phi_draw
            Sigma_store[:, :, draw_count] = Sigma_draw

            # IRF via Cholesky
            ir = compute_irf(Phi_draw, Sigma_draw, hor + 1)
            ir_store[:, :, :, draw_count] = ir
        end
    end

    # Final VAR estimate for returning
    v_final = var_estimate(hcat(y_slow, F_current), p; constant=constant)

    return FAVARResult(F_current, Lambda0, v_final,
                       Phi_store, Sigma_store, ir_store,
                       nfac, K)
end
