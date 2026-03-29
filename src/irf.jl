#=
irf.jl — Impulse response function computation
=#

"""
    compute_irf(Phi, Sigma, hor; Omega=I, unit=true)

Compute impulse response functions from VAR parameters.

# Arguments
- `Phi`:   (K*p + nx) × K — autoregressive coefficient matrix.
- `Sigma`: K × K — reduced‑form covariance.
- `hor`:   IRF horizon.
- `Omega`: K × K orthonormal rotation (default = identity → Cholesky).
- `unit`:  if `true`, 1‑std shocks; if `false`, unit shocks.

# Returns
3D array of size K × hor × n_shocks.
"""
function compute_irf(Phi::AbstractMatrix, Sigma::AbstractMatrix, hor::Int;
                     Omega::AbstractMatrix=Matrix{Float64}(I, size(Sigma, 1), size(Sigma, 1)),
                     unit::Bool=true)
    K = size(Sigma, 1)
    m = size(Phi, 1)
    n_shocks = size(Omega, 2)
    p = m ÷ K
    if m % K != 0
        p = (m - 1) ÷ K  # has constant
    end

    # Cholesky of Sigma
    Q = try
        cholesky(Hermitian(Sigma)).L
    catch
        # fallback: LDL
        F_ldl = ldlt(Hermitian(Sigma))
        Matrix(F_ldl.L) * Diagonal(sqrt.(max.(diag(F_ldl.D), 0.0)))
    end

    if !unit
        Q = Q * inv(Diagonal(diag(Q)))
    end

    # Companion form
    Kp = K * p
    F = zeros(Kp, Kp)
    F[1:K, :] = Phi[1:min(Kp, m), 1:K]'
    if p > 1
        F[K+1:Kp, 1:K*(p-1)] = I(K*(p-1))
    end
    G = zeros(Kp, K)
    G[1:K, :] = I(K)

    ir = zeros(K, hor, n_shocks)
    Fk = Matrix{Float64}(I, Kp, Kp)
    for k in 1:hor
        PHI = Fk * G * Q * Omega
        ir[:, k, :] = G' * PHI
        Fk = F * Fk
    end
    return ir
end

"""
    compute_irf_longrun(Phi, Sigma, hor, p)

Compute IRFs with long‑run identification (Blanchard‑Quah).
The long‑run impact matrix is the Cholesky decomposition of C(1)ΣC(1)'.
"""
function compute_irf_longrun(Phi::AbstractMatrix, Sigma::AbstractMatrix, hor::Int, p::Int)
    K = size(Sigma, 1)
    Kp = K * p
    F = companion_form(Phi, K, p)
    G = zeros(Kp, K); G[1:K, :] = I(K)
    Inp = I(Kp)

    # Long‑run multiplier C(1) = (I - F)^{-1}
    C1 = (Inp - F) \ Matrix{Float64}(Inp)
    C1 = C1[1:K, 1:K]

    # Long‑run structural impact
    PSI1 = cholesky(Hermitian(C1 * Sigma * C1')).L
    Q = C1 \ PSI1  # impact matrix

    # Normalize sign
    if Q[1, 1] < 0
        Q[:, 1] .*= -1
    end

    ir = zeros(K, hor, K)
    Fk = Matrix{Float64}(I, Kp, Kp)
    for k in 1:hor
        PHI = Fk * G * Q
        ir[:, k, :] = G' * PHI
        Fk = F * Fk
    end
    return ir, Q
end

"""
    compute_irf_proxy(Phi, Sigma, residuals, instrument, hor, p;
                      proxy_end=0, compute_F_stat=false)

Proxy/IV identification following Mertens & Ravn (2013).
"""
function compute_irf_proxy(Phi::AbstractMatrix, Sigma::AbstractMatrix,
                           residuals::AbstractMatrix, instrument::AbstractMatrix,
                           hor::Int, p::Int;
                           proxy_end::Int=0, compute_F_stat::Bool=false)
    T_res, K = size(residuals)
    T_m = size(instrument, 1)

    # Align instrument with residuals (instrument assumed to start at same point)
    res_start = T_res - T_m - proxy_end + 1
    res_end   = T_res - proxy_end
    res_sub   = residuals[res_start:res_end, :]

    # 2SLS identification
    XX = hcat(ones(T_m), instrument)
    Phib = XX \ res_sub

    uhat1 = XX * Phib[:, 1]
    XX2 = hcat(ones(T_m), uhat1)
    b21ib11_full = XX2 \ res_sub[:, 2:end]
    b21ib11 = b21ib11_full[2:end, :]'

    # Recover structural impact
    Sig11 = Sigma[1:1, 1:1]
    Sig21 = Sigma[2:end, 1:1]
    Sig22 = Sigma[2:end, 2:end]
    ZZp = b21ib11 * Sig11 * b21ib11' - (Sig21 * b21ib11' + b21ib11 * Sig21') + Sig22
    b12b12p = (Sig21 - b21ib11 * Sig11)' * (ZZp \ (Sig21 - b21ib11 * Sig11))
    b11b11p = Sig11 - b12b12p
    b11 = sqrt(max(b11b11p[1], 0.0))
    b1 = vcat([b11], b21ib11 * b11)

    # Compute IRFs
    Kp = K * p
    F = companion_form(Phi, K, p)
    G = zeros(Kp, K); G[1:K, :] = I(K)

    irs = zeros(hor, K)
    irs[1, :] = b1
    for h in 2:hor
        lvars = zeros(Kp)
        for j in 1:min(h-1, p)
            lvars[(j-1)*K+1:j*K] = irs[h-j, :]
        end
        irs[h, :] = (Phi[1:Kp, :]' * lvars)
    end

    # F‑statistic
    F_stat = NaN
    if compute_F_stat
        res_const = res_sub[:, 1] .- mean(res_sub[:, 1])
        res_full  = res_sub[:, 1] - XX * Phib[:, 1]
        SST = res_const' * res_const
        SSE = res_full' * res_full
        n_m = size(instrument, 2)
        F_stat = ((SST - SSE) / n_m) / (SSE / (T_m - n_m - 1))
    end

    return irs, b1, F_stat
end

"""
    compute_irf_heterosked(Phi, residuals, regimes, hor, p)

Compute IRFs using heteroskedasticity‑based identification (Rigobon 2003).

The method exploits differences in the reduced‑form covariance across
volatility regimes to recover the structural impact matrix.

# Arguments
- `Phi`:       (K*p + nx) × K coefficient matrix (AR part in first K*p rows).
- `residuals`: T × K reduced‑form residual matrix.
- `regimes`:   T‑length vector of integer regime labels (e.g. `[1,1,2,2,…]`).
- `hor`:       IRF horizon.
- `p`:         lag order.

# Returns
`(ir, A)` where `ir` is K × hor × K and `A` is the K × K structural impact matrix.
"""
function compute_irf_heterosked(Phi::AbstractMatrix, residuals::AbstractMatrix,
                                regimes::AbstractVector{<:Integer},
                                hor::Int, p::Int)
    T_res, K = size(residuals)
    length(regimes) == T_res ||
        throw(ArgumentError("regimes vector length ($(length(regimes))) must equal residual rows ($T_res)"))

    labels = sort(unique(regimes))
    length(labels) >= 2 ||
        throw(ArgumentError("need at least 2 distinct regimes, got $(length(labels))"))

    # Compute regime‑specific covariance matrices
    Sigmas = Dict{eltype(labels), Matrix{Float64}}()
    for lab in labels
        idx = findall(regimes .== lab)
        length(idx) >= K + 1 ||
            throw(ArgumentError("regime $lab has only $(length(idx)) observations, need at least $(K+1)"))
        u = residuals[idx, :]
        Sigmas[lab] = (u .- mean(u, dims=1))' * (u .- mean(u, dims=1)) / (length(idx) - 1)
    end

    # Rigobon (2003): use difference Σ₁ − Σ₂
    Sigma1 = Sigmas[labels[1]]
    Sigma2 = Sigmas[labels[2]]
    Delta = Sigma1 - Sigma2

    # Eigendecomposition of Σ₁⁻¹ * Δ to recover structural columns
    # If Σ_r = A D_r A' for regime r, then Σ₁⁻¹ Δ = A (D₁ - D₂)⁻¹ D₁⁻¹ ... — instead
    # use joint decomposition: Σ₁ = A D₁ A',  Σ₂ = A D₂ A'
    # ⟹ Σ₁⁻¹ Σ₂ = A⁻ᵀ D₁⁻¹ D₂ Aᵀ  (simultaneous diagonalisation)
    # The eigenvectors of Σ₁⁻¹ Σ₂ give Aᵀ (up to scale/sign)

    M = Sigma1 \ Sigma2
    eig = eigen(M)
    Ainv_T = real.(eig.vectors)  # columns are eigenvectors = rows of A⁻¹

    # Recover A: each column of A is a structural impact vector
    # Normalise so that A * A' matches Σ₁ in scale
    A = inv(Ainv_T')

    # Scale columns of A so that A * diag(d1) * A' ≈ Σ₁
    # d_j = (a_j' Σ₁⁻¹ a_j)⁻¹  where a_j is column j of A
    iS1 = inv(Sigma1)
    for j in 1:K
        s = sqrt(abs(A[:, j]' * iS1 * A[:, j]))
        if s > 0
            A[:, j] ./= s
        end
    end

    # Sign normalisation: positive diagonal
    for j in 1:K
        if A[j, j] < 0
            A[:, j] .*= -1
        end
    end

    # Compute IRFs using companion form
    Kp = K * p
    F = companion_form(Phi, K, p)
    G = zeros(Kp, K); G[1:K, :] = I(K)

    ir = zeros(K, hor, K)
    Fk = Matrix{Float64}(I, Kp, Kp)
    for k in 1:hor
        PHI = Fk * G * A
        ir[:, k, :] = G' * PHI
        Fk = F * Fk
    end
    return ir, A
end

"""
    wild_bootstrap_irf_proxy(var_est, instrument, hor;
                             nboot=1000, conf_level=0.90,
                             weight_type=:rademacher,
                             proxy_end=0, rng=Random.default_rng())

Compute wild‑bootstrap confidence bands for proxy/IV‑identified IRFs.

# Arguments
- `var_est`:     a `VAREstimate` from `var_estimate`.
- `instrument`:  T_m × n_inst instrument matrix.
- `hor`:         IRF horizon.
- `nboot`:       number of bootstrap replications (default 1000).
- `conf_level`:  confidence level for bands (default 0.90).
- `weight_type`: `:rademacher` for ±1 or `:mammen` for Mammen's 2‑point distribution.
- `proxy_end`:   trim from end of instrument alignment (default 0).
- `rng`:         random number generator.

# Returns
Named tuple `(point, lower, upper, boot_irfs)`.
- `point`:     hor × K point‑estimate IRFs.
- `lower`:     hor × K lower confidence band.
- `upper`:     hor × K upper confidence band.
- `boot_irfs`: hor × K × nboot bootstrap IRF draws.
"""
function wild_bootstrap_irf_proxy(var_est::VAREstimate,
                                  instrument::AbstractMatrix,
                                  hor::Int;
                                  nboot::Int=1000,
                                  conf_level::Float64=0.90,
                                  weight_type::Symbol=:rademacher,
                                  proxy_end::Int=0,
                                  rng::AbstractRNG=Random.default_rng())
    weight_type in (:rademacher, :mammen) ||
        throw(ArgumentError("weight_type must be :rademacher or :mammen"))
    0 < conf_level < 1 ||
        throw(ArgumentError("conf_level must be in (0, 1)"))

    K = var_est.nvar
    p = var_est.nlags
    T = var_est.nobs
    Y = var_est.Y
    X = var_est.X
    Phi = var_est.Phi
    u_hat = var_est.residuals
    Sigma = var_est.Sigma

    # Point estimate
    point_irf, _, _ = compute_irf_proxy(Phi, Sigma, u_hat, instrument, hor, p;
                                        proxy_end=proxy_end)

    # Mammen distribution parameters
    sqrt5 = sqrt(5.0)
    mammen_val1 = -(sqrt5 - 1.0) / 2.0
    mammen_val2 =  (sqrt5 + 1.0) / 2.0
    mammen_prob =  (sqrt5 + 1.0) / (2.0 * sqrt5)

    boot_irfs = zeros(hor, K, nboot)

    for b in 1:nboot
        # Draw wild bootstrap weights
        if weight_type == :rademacher
            w = 2.0 .* (rand(rng, T) .> 0.5) .- 1.0
        else  # mammen
            w = ifelse.(rand(rng, T) .< mammen_prob, mammen_val1, mammen_val2)
        end

        # Fixed‑design wild bootstrap: y* = X Φ̂ + w_t ⊙ û_t
        u_star = u_hat .* w
        Y_star = X * Phi + u_star

        # Re‑estimate VAR on bootstrap sample
        Phi_b, u_b, _ = ols_svd(Y_star, X)
        Sigma_b = u_b' * u_b / (T - size(X, 2))

        # Re‑estimate proxy identification
        try
            irs_b, _, _ = compute_irf_proxy(Phi_b, Sigma_b, u_b, instrument, hor, p;
                                            proxy_end=proxy_end)
            boot_irfs[:, :, b] = irs_b
        catch
            boot_irfs[:, :, b] .= NaN
        end
    end

    # Percentile confidence bands
    alpha = 1.0 - conf_level
    lower = zeros(hor, K)
    upper = zeros(hor, K)
    for h in 1:hor, k in 1:K
        vals = filter(!isnan, boot_irfs[h, k, :])
        if length(vals) > 0
            lower[h, k] = quantile(vals, alpha / 2)
            upper[h, k] = quantile(vals, 1 - alpha / 2)
        else
            lower[h, k] = NaN
            upper[h, k] = NaN
        end
    end

    return (point=point_irf, lower=lower, upper=upper, boot_irfs=boot_irfs)
end
