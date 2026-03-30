#=
robust_bayes.jl — Non-Gaussian / Robust Bayes posterior draws
Port of MATLAB bvar_.m robust_bayes code (Ferroni & Canova)
=#

"""
    robust_bayes_setup(u_ols, nobs, ny; K_shrinkage=ny)

Pre-compute robust Bayes quantities from OLS residuals.
Returns named tuple with fields needed for `robust_sigma_draw`.

# Arguments
- `u_ols`:         T × ny matrix of OLS residuals.
- `nobs`:          effective number of observations.
- `ny`:            number of variables.
- `K_shrinkage`:   shrinkage parameter toward Gaussian (higher → more Gaussian).
"""
function robust_bayes_setup(u_ols::AbstractMatrix, nobs::Int, ny::Int;
                            K_shrinkage::Float64=Float64(ny))
    Sig_ = (u_ols' * u_ols) / nobs
    vech_Sig_ = vech(Sig_)

    # Whiten residuals
    SigChol = cholesky(Hermitian(Sig_)).L
    iSigChol = inv(SigChol)
    white_u = u_ols * iSigChol'

    # Fourth moment of whitened residuals
    Khat_ = fourthmom(white_u)

    # Shrink toward Gaussian fourth moment
    I_ny = vech(Matrix{Float64}(I, ny, ny))
    Dpl = duplication_matrix(ny)
    Dplus = pinv(Dpl)
    Knn = commutation_matrix(ny, ny)

    K1_ = nobs / (nobs + K_shrinkage) * (Khat_ - I_ny * I_ny')
    K2_ = K_shrinkage / (nobs + K_shrinkage) * Dplus * (Matrix{Float64}(I, ny^2, ny^2) + Knn) * Dplus'

    Kstar_ = K1_ + K2_

    # Covariance of vech(Sigma)
    Left = Dplus * kron(SigChol, SigChol) * Dpl
    vech_Sig_cov_ = (1 / nobs) * Left * Kstar_ * Left'

    # Ensure symmetry and PSD
    vech_Sig_cov_ = Hermitian(vech_Sig_cov_)
    vech_Sig_cov_lower_chol = cholesky(vech_Sig_cov_).L

    return (vech_Sig=vech_Sig_, vech_Sig_cov_lower_chol=vech_Sig_cov_lower_chol,
            Sig_ols=Sig_, ny=ny)
end

"""
    robust_bayes_setup_skewness(u_ols, nobs, ny, vech_Sig_cov_; K_shrinkage=ny)

Pre-compute skewness correction quantities for `robust_bayes > 1`.
"""
function robust_bayes_setup_skewness(u_ols::AbstractMatrix, nobs::Int, ny::Int,
                                      vech_Sig_cov_::AbstractMatrix, Sig_ols::AbstractMatrix;
                                      K_shrinkage::Float64=Float64(ny))
    ivech_Sig_cov_ = pinv(vech_Sig_cov_)
    Sstar_ = (nobs / (nobs + K_shrinkage)) * thirdmom(u_ols)
    mu_cov_ = (1 / nobs) * Sstar_ * ivech_Sig_cov_ * Sstar_'

    # Ensure Sig_ols - mu_cov_ is PSD
    diff = Sig_ols - mu_cov_
    diff = Hermitian(diff)
    eigvals_diff = eigvals(diff)
    if any(eigvals_diff .<= 0)
        # Regularize: use only Sig_ols when mu_cov_ is too large
        mu_cov_chol = cholesky(Hermitian(Sig_ols * 0.99)).L
    else
        mu_cov_chol = cholesky(diff).L
    end

    return (Sstar=Sstar_, ivech_Sig_cov=ivech_Sig_cov_, mu_cov_chol=mu_cov_chol)
end

"""
    robust_sigma_draw(setup; rng=Random.default_rng())

Draw Sigma from the robust posterior (kurtosis-adjusted).
Returns the drawn covariance matrix Sigma and its lower Cholesky factor.
If the drawn Sigma is not PSD, returns `nothing` (caller should retry).
"""
function robust_sigma_draw(setup::NamedTuple; rng::AbstractRNG=Random.default_rng())
    ny = setup.ny
    m = ny * (ny + 1) ÷ 2

    Sig0 = randn(rng, m)
    Sig1 = setup.vech_Sig_cov_lower_chol * Sig0
    Sig2 = setup.vech_Sig + Sig1
    Sigma = ivech(Sig2)
    Sigma = Hermitian(Sigma)

    F = cholesky(Sigma; check=false)
    if !issuccess(F)
        return nothing, nothing, Sig2
    end
    return Matrix(Sigma), F.L, Sig2
end
