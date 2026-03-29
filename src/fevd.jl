#=
fevd.jl — Forecast Error Variance Decomposition
=#

"""
    compute_fevd(Phi, Sigma, hor; Omega=I)

Compute FEVD at horizon `hor`.

Returns K × K matrix: `FEVD[i,j]` = percentage of variable `i`'s forecast
error variance explained by shock `j`.
"""
function compute_fevd(Phi::AbstractMatrix, Sigma::AbstractMatrix, hor::Int;
                      Omega::AbstractMatrix=Matrix{Float64}(I, size(Sigma, 1), size(Sigma, 1)))
    K = size(Sigma, 1)
    p = size(Phi, 1) ÷ K
    if size(Phi, 1) % K != 0
        p = (size(Phi, 1) - 1) ÷ K
    end

    # Companion form
    Kp = K * p
    F = zeros(Kp, Kp)
    ar_rows = min(Kp, size(Phi, 1))
    F[1:K, 1:ar_rows] = Phi[1:ar_rows, :]'
    if p > 1
        F[K+1:Kp, 1:K*(p-1)] = I(K*(p-1))
    end
    G = zeros(Kp, K); G[1:K, :] = I(K)

    A = cholesky(Hermitian(Sigma)).L
    Kappa = G * A * Omega

    # Total variance
    tmp_all = zeros(Kp, Kp)
    Fh = Matrix{Float64}(I, Kp, Kp)
    for h in 1:hor
        tmp_all += Fh * Kappa * Kappa' * Fh'
        Fh = F * Fh
    end
    all_var = diag(tmp_all[1:K, 1:K])

    # Per-shock variance
    var_by_shock = zeros(K, K)
    for s in 1:K
        Ind = zeros(K, K); Ind[s, s] = 1.0
        tmp_s = zeros(Kp, Kp)
        Fh = Matrix{Float64}(I, Kp, Kp)
        for h in 1:hor
            tmp_s += Fh * Kappa * Ind * Kappa' * Fh'
            Fh = F * Fh
        end
        var_by_shock[:, s] = diag(tmp_s[1:K, 1:K])
    end

    # Sanity check
    discrepancy = maximum(abs.(sum(var_by_shock, dims=2) .- all_var))
    if discrepancy > 1e-8
        @warn "FEVD consistency check failed: max discrepancy = $discrepancy"
    end

    FEVD = zeros(K, K)
    for s in 1:K
        FEVD[:, s] = var_by_shock[:, s] ./ all_var .* 100
    end
    return FEVDResult(FEVD, hor)
end

"""
    fevd_posterior(bvar_result; conf_level=0.68, horizons=1:bvar_result.hor)

Compute FEVD with posterior credible bands.
"""
function fevd_posterior(result::BVARResult;
                       conf_level::Float64=0.68,
                       horizons::AbstractVector{Int}=1:result.hor)
    K = result.nvar
    nhor = length(horizons)
    ndraws = result.ndraws

    fevd_all = zeros(K, K, nhor, ndraws)
    for d in 1:ndraws
        Phi = result.Phi_draws[:, :, d]
        Sigma = result.Sigma_draws[:, :, d]
        Omega = result.Omega_draws[:, :, d]
        if any(isnan.(Omega))
            Omega = I(K)
        end
        for (hi, h) in enumerate(horizons)
            fevd_r = compute_fevd(Phi[1:K*result.nlags, :], Sigma, h; Omega=Omega)
            fevd_all[:, :, hi, d] = fevd_r.decomposition
        end
    end

    alpha = (1 - conf_level) / 2
    med = mapslices(x -> quantile(x, 0.5), fevd_all, dims=4)[:, :, :, 1]
    lo  = mapslices(x -> quantile(x, alpha), fevd_all, dims=4)[:, :, :, 1]
    hi  = mapslices(x -> quantile(x, 1-alpha), fevd_all, dims=4)[:, :, :, 1]

    return FEVDPosteriorResult(med, lo, hi, conf_level)
end
