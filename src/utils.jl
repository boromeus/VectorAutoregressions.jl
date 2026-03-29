#=
utils.jl — Matrix utilities and helper functions
=#

"""
    lagmatrix(y, p; constant=false, trend=false)

Build the regressor matrix with `p` lags from data matrix `y` (T × K).
Returns matrix of size (T-p) × (K*p [+ 1] [+ 1]).
Lags are ordered as [y_{t-1} … y_{t-p} [constant] [trend]].
"""
function lagmatrix(y::AbstractMatrix, p::Int; constant::Bool=false, trend::Bool=false)
    T, K = size(y)
    Teff = T - p
    ncols = K * p + (constant ? 1 : 0) + (trend ? 1 : 0)
    X = Matrix{Float64}(undef, Teff, ncols)
    for lag in 1:p
        @inbounds for t in 1:Teff
            for k in 1:K
                X[t, (lag-1)*K + k] = y[p + t - lag, k]
            end
        end
    end
    col = K * p
    if constant
        col += 1
        @inbounds for t in 1:Teff
            X[t, col] = 1.0
        end
    end
    if trend
        col += 1
        @inbounds for t in 1:Teff
            X[t, col] = Float64(t)
        end
    end
    return X
end

"""
    companion_form(Phi, K, p)

Build the companion matrix from the AR coefficients.
`Phi` is (K*p + nx) × K;  only the first K*p rows (AR part) are used.
Returns (K*p) × (K*p) companion matrix.
"""
function companion_form(Phi::AbstractMatrix, K::Int, p::Int)
    Kp = K * p
    F = zeros(Kp, Kp)
    F[1:K, :] = Phi[1:K*p, :]'
    if p > 1
        F[K+1:Kp, 1:K*(p-1)] = I(K*(p-1))
    end
    return F
end

"""
    check_stability(Phi, K, p)

Return `true` if all eigenvalues of the companion matrix are inside the unit circle.
"""
function check_stability(Phi::AbstractMatrix, K::Int, p::Int)
    F = companion_form(Phi, K, p)
    return maximum(abs.(eigvals(F))) < 1.0
end

"""
    rand_inverse_wishart(df, scale; rng)

Draw from the Inverse‑Wishart distribution IW(df, scale).
Uses the Bartlett decomposition via SVD for numerical stability.
"""
function rand_inverse_wishart(df::Int, scale::AbstractMatrix; rng::AbstractRNG=Random.default_rng())
    K = size(scale, 1)
    H_inv = cholesky(Hermitian(inv(scale))).U  # upper Cholesky of inv(scale)
    X = randn(rng, df, K) * H_inv
    # inv(X'X) via SVD for stability
    U, S, V = svd(X)
    SSi = 1.0 ./ (S .^ 2)
    G = V * Diagonal(SSi) * V'
    return Symmetric(G)
end

"""
    vech(A)

Half‑vectorization: extract lower‑triangular elements column‑by‑column.
"""
function vech(A::AbstractMatrix)
    n = size(A, 1)
    v = Float64[]
    for j in 1:n
        for i in j:n
            push!(v, A[i, j])
        end
    end
    return v
end

"""
    ivech(v, n)

Inverse half‑vectorization: reconstruct symmetric n×n matrix from vector `v`.
"""
function ivech(v::AbstractVector, n::Int)
    A = zeros(n, n)
    idx = 1
    for j in 1:n
        for i in j:n
            A[i, j] = v[idx]
            A[j, i] = v[idx]
            idx += 1
        end
    end
    return A
end

"""
    ivech(v)

Inverse half‑vectorization: infer dimension from length of v.
"""
function ivech(v::AbstractVector)
    m = length(v)
    n = round(Int, (-1 + sqrt(1 + 8m)) / 2)
    return ivech(v, n)
end

"""
    commutation_matrix(n, m)

Magnus–Neudecker commutation matrix K_{n,m}.
"""
function commutation_matrix(n::Int, m::Int)
    K = zeros(n*m, n*m)
    for i in 1:n
        for j in 1:m
            # e_i^n ⊗ e_j^m  maps to  e_j^m ⊗ e_i^n
            row = (j-1)*n + i
            col = (i-1)*m + j
            K[row, col] = 1.0
        end
    end
    return K
end

"""
    duplication_matrix(n)

Magnus–Neudecker duplication matrix D_n such that D_n * vech(A) = vec(A) for symmetric A.
"""
function duplication_matrix(n::Int)
    m = n * (n + 1) ÷ 2
    D = zeros(n^2, m)
    col = 0
    for j in 1:n
        for i in j:n
            col += 1
            D[(j-1)*n + i, col] = 1.0
            D[(i-1)*n + j, col] = 1.0
        end
    end
    return D
end

"""
    elimination_matrix(n)

Elimination matrix L_n such that L_n * vec(A) = vech(A).
"""
function elimination_matrix(n::Int)
    m = n * (n + 1) ÷ 2
    L = zeros(m, n^2)
    col = 0
    for j in 1:n
        for i in j:n
            col += 1
            L[col, (j-1)*n + i] = 1.0
        end
    end
    return L
end

"""
    var2ma(Phi_ar, horizon)

Convert VAR AR coefficients to MA representation.
`Phi_ar` is (K*p) × K (AR part only, no constant).
Returns 3D array K × K × horizon.
"""
function var2ma(Phi_ar::AbstractMatrix, horizon::Int)
    K = size(Phi_ar, 2)
    nlags = size(Phi_ar, 1) ÷ K
    MA = zeros(K, K, horizon)
    MA[:, :, 1] = I(K)
    for h in 2:horizon
        tmp = zeros(K, K)
        kstop = min(h, nlags + 1)
        for l in 2:kstop
            span_ar = (l-2)*K+1 : (l-2)*K+K
            tmp += MA[:, :, h-l+1] * Phi_ar[span_ar, :]'
        end
        MA[:, :, h] = tmp
    end
    return MA
end

"""
    var2ss(Phi, Sigma, K, p; index=nothing)

Convert VAR to state‑space form.
Returns (A, B, C, const_vec, lags).
"""
function var2ss(Phi::AbstractMatrix, Sigma::AbstractMatrix, K::Int, p::Int)
    F = companion_form(Phi, K, p)
    G = zeros(K * p, K)
    G[1:K, 1:K] = I(K)
    C = zeros(K, K * p)
    C[1:K, 1:K] = I(K)
    # steady‑state mean
    IminusA = I(K)
    for ell in 1:p
        IminusA -= Phi[(ell-1)*K+1:ell*K, :]'
    end
    nx = size(Phi, 1) - K * p
    if nx >= 1
        const_vec = [IminusA \ Phi[end-nx+1, :]'; zeros(K*(p-1), 1)]
    else
        const_vec = zeros(K * p, 1)
    end
    return F, G, C, const_vec, p
end

"""
    ols_svd(y, X)

OLS via SVD for numerical stability. Returns (B, u, xxi) where
B = (X'X)^{-1}X'y,  u = y - X*B,  xxi = (X'X)^{-1}.
"""
function ols_svd(y::AbstractMatrix, X::AbstractMatrix)
    T = size(X, 1)
    nk = size(X, 2)
    if T == 0 || nk == 0
        B = zeros(nk, size(y, 2))
        u = copy(y)
        xxi = zeros(nk, nk)
        return B, u, xxi
    end
    U, S, V = svd(X)
    di = 1.0 ./ S
    B = (V .* di') * U' * y
    u = y - X * B
    xxi_factor = V .* di'
    xxi = xxi_factor * xxi_factor'
    return B, u, xxi
end

"""
    generate_rotation_matrix(K; rng)

Generate a random K×K orthonormal matrix via QR decomposition.
Normalises to positive diagonal entries in R (Rubio‑Ramírez et al. 2010).
"""
function generate_rotation_matrix(K::Int; rng::AbstractRNG=Random.default_rng())
    G = randn(rng, K, K)
    Q, R = qr(G)
    Q_mat = Matrix(Q)
    # Normalize to positive diagonal
    for j in 1:K
        if R[j, j] < 0
            Q_mat[:, j] .*= -1
        end
    end
    return Q_mat
end

"""
    matrictint(S, df, XXi)

Log of the integral of the normal‑inverse‑Wishart kernel.
Used for marginal likelihood computation.
"""
function matrictint(S::AbstractMatrix, df::Int, XXi::AbstractMatrix)
    k = size(XXi, 1)
    ny = size(S, 1)
    cx = cholesky(Hermitian(XXi)).U
    cs = cholesky(Hermitian(S)).U
    # Matrix‑normal component
    w1 = 0.5 * k * ny * log(2π) + ny * sum(log.(diag(cx)))
    # Inverse‑Wishart component
    w2 = -df * sum(log.(diag(cs))) + 0.5 * df * ny * log(2) +
         ny * (ny - 1) * 0.25 * log(π) + ggammaln(ny, df)
    return w1 + w2
end

function ggammaln(m::Int, df::Int)
    if df <= m - 1
        error("Too few df in ggammaln")
    end
    garg = [0.5 * (df + 1 - i) for i in 1:m]
    return sum(loggamma.(garg))
end
