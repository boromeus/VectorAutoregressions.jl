#=
priors.jl — Prior construction (dummy observations, conjugate, etc.)
=#

"""
    get_prior_moments(y, p)

Estimate prior hyper‑parameters from OLS on each variable individually.
Returns `(mu, sigma, delta)` — means, residual std, AR(1) coefficients.
"""
function get_prior_moments(y::AbstractMatrix, p::Int=1)
    T, K = size(y)
    mu = mean(y, dims=1)'  # K×1
    sigma = Vector{Float64}(undef, K)
    delta = Vector{Float64}(undef, K)
    for i in 1:K
        yi = y[:, i]
        xi = lagmatrix(reshape(yi, :, 1), p)
        xi = hcat(xi, ones(size(xi, 1)))
        yi_dep = yi[p+1:end]
        b = xi \ yi_dep
        e = yi_dep - xi * b
        sigma[i] = sqrt(e' * e / length(yi_dep))
        delta[i] = clamp(b[1], -1.0, 1.0)
    end
    return vec(mu), sigma, delta
end

"""
    build_dummy_observations(prior::MinnesotaPrior, ny, p, sig, delta, mu)

Build Minnesota‑prior dummy observations following Sims's `varprior.m`.

`sig` = vector of prior modes for diagonal of r.f. covariance.
`delta` = AR(1) coefficients.
`mu` = sample means.

Returns `(ydum, xdum, pbreaks)`.
"""
function build_dummy_observations(prior::MinnesotaPrior, ny::Int, p::Int,
                                  sig::AbstractVector{<:Real}, delta::AbstractVector{<:Real},
                                  mu::AbstractVector{<:Real})
    nx = 1  # constant
    tight = prior.tau
    decay = prior.decay
    omega_w = round(Int, max(prior.omega, 0))

    # ── Tightness dummy observations ──
    ydum_blocks = Matrix{Float64}[]
    xdum_blocks = Matrix{Float64}[]
    for il in 1:p
        for iv in 1:ny
            yd = zeros(p + 1, ny)
            xd = zeros(p + 1, nx)
            yd[il+1, iv] = il^decay * sig[iv]
            push!(ydum_blocks, tight .* yd)
            push!(xdum_blocks, tight .* xd)
        end
    end

    # ── Scale dummy ──
    yd_scale = zeros(p + 1, ny)
    xd_scale = zeros(p + 1, nx)
    yd_scale[1, :] = sig
    push!(ydum_blocks, tight .* yd_scale)
    push!(xdum_blocks, tight .* xd_scale)

    ydum = vcat(ydum_blocks...)
    xdum = vcat(xdum_blocks...)
    pbreaks_list = collect((p+1):(p+1):(size(ydum, 1)))

    # ── Variance‑of‑sigma dummy observations ──
    if omega_w > 0
        for _ in 1:omega_w
            for iv in 1:ny
                yd2 = zeros(p + 1, ny)
                xd2 = zeros(p + 1, nx)
                yd2[end, iv] = sig[iv]
                ydum = vcat(ydum, yd2)
                xdum = vcat(xdum, xd2)
                push!(pbreaks_list, size(ydum, 1))
            end
        end
    end

    pbreaks = pbreaks_list[1:end-1]
    return ydum, xdum, pbreaks
end

"""
    compute_prior_posterior(y, p, prior::FlatPrior; constant, trend, nexogenous)

Flat‑prior posterior computation.
"""
function compute_prior_posterior(y::AbstractMatrix, p::Int, prior::FlatPrior;
                                constant::Bool=true, trend::Bool=false,
                                nexogenous::Int=0)
    T, K = size(y)
    nx = constant ? 1 : 0
    xdata = ones(T, nx)
    if trend
        xdata = hcat(xdata, collect(1.0-p:T-p))
    end
    idx = 1:T

    B, u, xxi, y_ols, X_ols = rfvar3(y[idx, :], p, xdata, [T, T], 0.0, 0.0)
    Tu = size(u, 1)

    flat_adj = K + 1
    post_df  = Tu - K * p - nx + flat_adj - nexogenous
    post_S   = u' * u
    post_XXi = xxi
    post_PhiHat = B

    prior_out = (name="Jeffrey", df=0, S=zeros(K, K), XXi=zeros(0, 0), PhiHat=zeros(0, 0))
    posterior = (df=post_df, S=post_S, XXi=post_XXi, PhiHat=post_PhiHat)
    return prior_out, posterior, B, u, xxi, y_ols, X_ols
end

"""
    compute_prior_posterior(y, p, prior::MinnesotaPrior; ...)

Minnesota prior: build dummy observations, compute prior and posterior moments.
"""
function compute_prior_posterior(y::AbstractMatrix, p::Int, prior::MinnesotaPrior;
                                constant::Bool=true, trend::Bool=false,
                                nexogenous::Int=0, presample::Int=0,
                                firstobs::Int=p+1)
    T, K = size(y)
    nx = constant ? 1 : 0

    # Ensure enough presample for y-bar computation
    if presample == 0
        presample = p
        firstobs = max(firstobs, 2 * p + 1)
    end

    # Prior hyper from pre‑sample
    pre_start = max(1, firstobs - p - presample + 1)
    pre_end   = firstobs - 1
    y_pre = y[pre_start:pre_end, :]
    sig = vec(std(y_pre, dims=1))
    # Guard against zero std (from tiny samples)
    sig[sig .== 0] .= 1.0
    sig[isnan.(sig)] .= 1.0

    _, sigma_ar, delta = get_prior_moments(y, 1)

    ydum, xdum, pbreaks = build_dummy_observations(prior, K, p, Float64.(sig), Float64.(delta),
                                                   Float64.(vec(mean(y_pre, dims=1))))

    # Actual data indices 
    idx = firstobs:T
    xdata_full = ones(length(idx), nx)
    if trend
        xdata_full = hcat(xdata_full, collect(1.0:length(idx)))
    end

    # OLS on actual data only
    B_ols, u_ols, xxi_ols, y_ols, X_ols = rfvar3(y[idx, :], p, xdata_full,
                                                   [length(idx), length(idx)], 0.0, 0.0)

    # Posterior (on actual + dummies) — dummies appended after actual data
    post_data = vcat(y[max(1, firstobs-p):T, :], ydum)
    post_xdata_raw = ones(T - max(1, firstobs-p) + 1, nx)
    if trend
        post_xdata_raw = hcat(post_xdata_raw, collect(1.0:size(post_xdata_raw, 1)))
    end
    post_xdata = vcat(post_xdata_raw, xdum)
    T_actual = T - max(1, firstobs-p) + 1
    pbreaks_post = vcat(T_actual, T_actual .+ pbreaks)

    B_post, u_post, xxi_post, _, _ = rfvar3(post_data, p, post_xdata,
                                              pbreaks_post, prior.lambda, prior.mu)
    Tu_post = size(u_post, 1)
    post_df  = Tu_post - K * p - nx
    post_S   = u_post' * u_post
    post_XXi = xxi_post
    post_PhiHat = B_post

    # Prior (on presample + dummies only)
    pre_data = vcat(y[pre_start:pre_end, :], ydum)
    pre_xdata_raw = ones(pre_end - pre_start + 1, nx)
    if trend
        pre_xdata_raw = hcat(pre_xdata_raw, collect(1.0:size(pre_xdata_raw, 1)))
    end
    pre_xdata = vcat(pre_xdata_raw, xdum)
    T_pre_actual = pre_end - pre_start + 1
    pbreaks_prior = vcat(T_pre_actual, T_pre_actual .+ pbreaks)

    B_prior, u_prior, xxi_prior, _, _ = rfvar3(pre_data, p, pre_xdata,
                                                 pbreaks_prior, prior.lambda, prior.mu)
    Tup = size(u_prior, 1)
    prior_df  = Tup - K * p - nx
    prior_S   = u_prior' * u_prior
    prior_XXi = xxi_prior
    prior_PhiHat = B_prior

    if prior_df < K
        @warn "Too few prior df ($prior_df < $K). Using post_df for IW."
        prior_df = K + 1
        prior_S = Matrix{Float64}(I, K, K)
        prior_XXi = 10.0 * I(K * p + nx)
    end

    prior_out = (name="Minnesota", df=prior_df, S=prior_S, XXi=prior_XXi, PhiHat=prior_PhiHat,
                 YYdum=ydum, XXdum=xdum)
    posterior = (df=post_df, S=post_S, XXi=post_XXi, PhiHat=post_PhiHat)
    return prior_out, posterior, B_ols, u_ols, xxi_ols, y_ols, X_ols
end

"""
    compute_prior_posterior(y, p, prior::ConjugatePrior; ...)

Conjugate MN‑IW prior posterior computation.
"""
function compute_prior_posterior(y::AbstractMatrix, p::Int, prior::ConjugatePrior;
                                constant::Bool=true, trend::Bool=false,
                                nexogenous::Int=0)
    T, K = size(y)
    nx = constant ? 1 : 0

    v = var_estimate(y, p; constant=constant, trend=trend)
    X_ols = v.X
    Y_ols = v.Y
    u_ols = v.residuals
    B_ols = v.Phi
    xxi_ols = v.XXi
    Tu = v.nobs

    Ai = inv(prior.Phi_cov)
    post_df = Tu + prior.Sigma_df
    post_XXi = inv(X_ols' * X_ols + Ai)
    post_PhiHat = post_XXi * (X_ols' * Y_ols + Ai * prior.Phi_mean)
    post_S = u_ols' * u_ols + prior.Sigma_scale +
             prior.Phi_mean' * Ai * prior.Phi_mean +
             B_ols' * (X_ols' * X_ols) * B_ols -
             post_PhiHat' * (X_ols' * X_ols + Ai) * post_PhiHat

    prior_out = (name="Conjugate", df=prior.Sigma_df, S=prior.Sigma_scale,
                 XXi=prior.Phi_cov, PhiHat=prior.Phi_mean)
    posterior = (df=post_df, S=post_S, XXi=post_XXi, PhiHat=post_PhiHat)
    return prior_out, posterior, B_ols, u_ols, xxi_ols, Y_ols, X_ols
end
