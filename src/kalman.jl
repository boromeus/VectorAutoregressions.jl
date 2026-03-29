#=
kalman.jl — Kalman filter and smoother with NaN handling
Port of MATLAB kfilternan.m and kf_dk.m
=#

"""
    kalman_filter(Phi, Sigma, y; initial_cond=:diffuse, state_space=:var,
                  index=nothing, start=1)

Run the Kalman filter and RTS smoother on data `y` (T × K) using
the VAR(p) state‑space representation derived from `Phi` and `Sigma`.

Handles missing data (NaN) automatically.

# Returns
Named tuple `(logL, states, smoothed, yforecast)`.
"""
function kalman_filter(Phi::AbstractMatrix, Sigma::AbstractMatrix,
        y::AbstractMatrix;
        initial_cond::Symbol = :diffuse,
        index::Union{Nothing, Vector{Int}} = nothing,
        start::Int = 1)
    T, ny = size(y)
    data = y'

    # State-space form: x(t) = A x(t-1) + B u(t);  y(t) = C x(t) + const
    A, B, C, const_vec = _var2ss(Phi, Sigma, ny; index = index)
    ns = size(A, 1)

    # Pre-allocate
    stt = zeros(ns, T + 1)
    ptt = zeros(ns, ns, T + 1)
    sfor = zeros(ns, T)
    yfor = zeros(ny, T)
    logLnc = zeros(T)

    # Initialization
    if initial_cond == :stationary
        # Solve Lyapunov: P = A P A' + B Σ B'
        BΣB = B * Sigma * B'
        P0 = _solve_lyapunov(A, BΣB)
        stt[:, 1] .= 0.0
        ptt[:, :, 1] = P0
    else  # :diffuse
        ptt[:, :, 1] = 10.0 * I(ns)
        stt[:, 1] .= 0.0
    end

    # Forward filter
    for t in 1:T
        yt = data[:, t]

        # Handle missing observations
        obs_mask = .!isnan.(yt)
        obs_idx = findall(obs_mask)
        yt_obs = yt[obs_mask]
        Ct = C[obs_mask, :]
        const_obs = const_vec[obs_mask]

        # Demean
        yt_dm = yt_obs - Ct * const_vec  # simplified: use const within state

        # Forecast
        state_prior = A * stt[:, t]
        P_prior = A * ptt[:, :, t] * A' + B * Sigma * B'
        sfor[:, t] = state_prior

        if isempty(obs_idx)
            # All missing
            stt[:, t + 1] = state_prior
            ptt[:, :, t + 1] = P_prior
            continue
        end

        # Innovation
        yt_pred = Ct * state_prior + const_obs
        yfor[obs_idx, t] = yt_pred
        v = yt_obs - yt_pred

        # Innovation covariance
        F = Ct * P_prior * Ct'
        F = Hermitian(F)

        # Kalman gain
        Finv = inv(F)
        KG = P_prior * Ct' * Finv

        # Update
        stt[:, t + 1] = state_prior + KG * v
        ptt[:, :, t + 1] = P_prior - KG * Ct * P_prior

        # Log-likelihood contribution
        nd = length(obs_idx)
        logLnc[t] = -0.5 * (nd * log(2π) + logdet(F) + v' * Finv * v)
    end

    logL = sum(logLnc[start:end])

    # RTS smoother
    smoothed = _rts_smoother(stt, ptt, A, T, ns)

    return (logL = logL,
        states = stt[:, 2:end]',
        smoothed = smoothed',
        yforecast = yfor')
end

# ─── Internal helpers ───────────────────────────────────────────────────────────

"""
Build the VAR state-space representation matching MATLAB var2ss.m.
"""
function _var2ss(Phi::AbstractMatrix, Sigma::AbstractMatrix, ny::Int;
        index::Union{Nothing, Vector{Int}} = nothing)
    m, K = size(Phi)
    if K != ny
        throw(DimensionMismatch("Phi has $K columns but ny=$ny"))
    end
    lags = if rem(m, ny) == 0
        m ÷ ny
    else
        (m - 1) ÷ ny
    end

    ns = ny * lags  # state dimension

    # Transition: A (ns × ns)
    A = zeros(ns, ns)
    A[1:ny, :] = Phi[1:(ny * lags), :]'
    if lags > 1
        A[(ny + 1):ns, 1:(ny * (lags - 1))] = I(ny * (lags - 1))
    end

    # Shock loading: B (ns × ny)
    B = zeros(ns, ny)
    B[1:ny, :] = I(ny)

    # Observation: C (ny × ns) — observe the first ny states
    if index === nothing
        C = zeros(ny, ns)
        C[1:ny, 1:ny] = I(ny)
    else
        C = zeros(ny, ns)
        for (i, idx) in enumerate(index)
            if idx > 0
                C[i, idx] = 1.0
            end
        end
    end

    # Constant (intercept in state, if present)
    const_vec = zeros(ns)
    if m > ny * lags
        const_vec[1:ny] = Phi[ny * lags + 1, :]
    end

    return A, B, C, const_vec
end

"""
Solve Lyapunov equation P = A P A' + Q iteratively.
"""
function _solve_lyapunov(A::AbstractMatrix, Q::AbstractMatrix;
        maxiter::Int = 1000, tol::Float64 = 1e-12)
    n = size(A, 1)
    P = Matrix{Float64}(I, n, n) * 10.0
    for _ in 1:maxiter
        P_new = A * P * A' + Q
        if maximum(abs.(P_new - P)) < tol
            return P_new
        end
        P = P_new
    end
    return P
end

"""
Rauch–Tung–Striebel smoother.
"""
function _rts_smoother(stt, ptt, A, T, ns)
    # stt is ns × (T+1), ptt is ns × ns × (T+1)
    s_smooth = copy(stt)
    P_smooth = copy(ptt)

    for t in T:-1:1
        P_pred = A * ptt[:, :, t] * A' + I(ns) * 1e-10  # regularize
        J = ptt[:, :, t] * A' / Hermitian(P_pred)
        s_smooth[:, t] = stt[:, t] + J * (s_smooth[:, t + 1] - A * stt[:, t])
        P_smooth[:, :, t] = ptt[:, :, t] + J * (P_smooth[:, :, t + 1] - P_pred) * J'
    end

    return s_smooth[:, 1:T]
end
