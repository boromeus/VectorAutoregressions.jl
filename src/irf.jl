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
        Omega::AbstractMatrix = Matrix{Float64}(I, size(Sigma, 1), size(Sigma, 1)),
        unit::Bool = true)
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
        F[(K + 1):Kp, 1:(K * (p - 1))] = I(K*(p-1))
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
    G = zeros(Kp, K);
    G[1:K, :] = I(K)
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
        proxy_end::Int = 0, compute_F_stat::Bool = false)
    T_res, K = size(residuals)
    T_m = size(instrument, 1)

    # Align instrument with residuals (instrument assumed to start at same point)
    res_start = T_res - T_m - proxy_end + 1
    res_end = T_res - proxy_end
    res_sub = residuals[res_start:res_end, :]

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
    G = zeros(Kp, K);
    G[1:K, :] = I(K)

    irs = zeros(hor, K)
    irs[1, :] = b1
    for h in 2:hor
        lvars = zeros(Kp)
        for j in 1:min(h - 1, p)
            lvars[((j - 1) * K + 1):(j * K)] = irs[h - j, :]
        end
        irs[h, :] = (Phi[1:Kp, :]' * lvars)
    end

    # F‑statistic
    F_stat = NaN
    if compute_F_stat
        res_const = res_sub[:, 1] .- mean(res_sub[:, 1])
        res_full = res_sub[:, 1] - XX * Phib[:, 1]
        SST = res_const' * res_const
        SSE = res_full' * res_full
        n_m = size(instrument, 2)
        F_stat = ((SST - SSE) / n_m) / (SSE / (T_m - n_m - 1))
    end

    return irs, b1, F_stat
end
