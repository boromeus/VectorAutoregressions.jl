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
                       initial_cond::Symbol=:diffuse,
                       index::Union{Nothing,Vector{Int}}=nothing,
                       start::Int=1)
    T, ny = size(y)
    data = y'

    # State-space form: x(t) = A x(t-1) + B u(t);  y(t) = C x(t) + const
    A, B, C, const_vec = _var2ss(Phi, Sigma, ny; index=index)
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

        # Forecast
        state_prior = A * stt[:, t] + const_vec
        P_prior = A * ptt[:, :, t] * A' + B * Sigma * B'
        sfor[:, t] = state_prior

        if isempty(obs_idx)
            # All missing
            stt[:, t+1] = state_prior
            ptt[:, :, t+1] = P_prior
            continue
        end

        # Innovation
        yt_pred = Ct * state_prior
        yfor[obs_idx, t] = yt_pred
        v = yt_obs - yt_pred

        # Innovation covariance (regularize for numerical stability)
        F = Ct * P_prior * Ct' + 1e-10 * I(length(obs_idx))
        F = Hermitian(F)

        # Kalman gain
        Finv = inv(F)
        KG = P_prior * Ct' * Finv

        # Update
        stt[:, t+1] = state_prior + KG * v
        ptt[:, :, t+1] = P_prior - KG * Ct * P_prior

        # Log-likelihood contribution
        nd = length(obs_idx)
        logLnc[t] = -0.5 * (nd * log(2π) + logdet(F) + v' * Finv * v)
    end

    logL = sum(logLnc[start:end])

    # RTS smoother
    smoothed = _rts_smoother(stt, ptt, A, T, ns)

    return (logL=logL,
            states=stt[:, 2:end]',
            smoothed=smoothed',
            yforecast=yfor')
end

# ─── Internal helpers ───────────────────────────────────────────────────────────

"""
Build the VAR state-space representation matching MATLAB var2ss.m.
"""
function _var2ss(Phi::AbstractMatrix, Sigma::AbstractMatrix, ny::Int;
                 index::Union{Nothing,Vector{Int}}=nothing)
    m, K = size(Phi)
    if K != ny
        error("Dimension mismatch between Phi and ny")
    end
    lags = if rem(m, ny) == 0
        m ÷ ny
    else
        (m - 1) ÷ ny
    end

    ns = ny * lags  # state dimension

    # Transition: A (ns × ns)
    A = zeros(ns, ns)
    A[1:ny, :] = Phi[1:ny*lags, :]'
    if lags > 1
        A[ny+1:ns, 1:ny*(lags-1)] = I(ny * (lags - 1))
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
        const_vec[1:ny] = Phi[ny*lags+1, :]
    end

    return A, B, C, const_vec
end

"""
Solve Lyapunov equation P = A P A' + Q iteratively.
"""
function _solve_lyapunov(A::AbstractMatrix, Q::AbstractMatrix;
                         maxiter::Int=1000, tol::Float64=1e-12)
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
function _rts_smoother(stt, ptt, A, T, ns;
                       Q::AbstractMatrix=Matrix{Float64}(1e-10 * I(ns)))
    # stt is ns × (T+1), ptt is ns × ns × (T+1)
    s_smooth = copy(stt)
    P_smooth = copy(ptt)

    for t in T:-1:1
        P_pred = A * ptt[:, :, t] * A' + Q
        J = ptt[:, :, t] * A' / Hermitian(P_pred)
        s_smooth[:, t] = stt[:, t] + J * (s_smooth[:, t+1] - A * stt[:, t])
        P_smooth[:, :, t] = ptt[:, :, t] + J * (P_smooth[:, :, t+1] - P_pred) * J'
    end

    return s_smooth[:, 1:T]
end

# ─── Mixed‑Frequency VAR via EM ────────────────────────────────────────────────

"""
    mixed_freq_var(y, p, freq_map; constant=true, max_iter=100, tol=1e-6)

Estimate a VAR with mixed‑frequency data using the EM algorithm
(Kalman E‑step, OLS M‑step).

# Arguments
- `y::Matrix`:          T × K data at the highest frequency. Low‑frequency
                        variables have NaN in non‑observation periods.
- `p::Int`:             VAR lag order.
- `freq_map::Vector{Symbol}`:  length‑K vector, each element `:high`, `:stock`, or `:flow`.
    - `:high`  — observed every period.
    - `:stock` — point‑in‑time (level) observed only when not NaN.
    - `:flow`  — period sum/average; observed value = sum of underlying high‑freq values
                 between consecutive observations.
- `constant::Bool`:     include VAR intercept (default `true`).
- `max_iter::Int`:      maximum EM iterations (default 100).
- `tol::Float64`:       convergence tolerance on log‑likelihood (default 1e‑6).

# Returns
`MixedFreqVARResult`.
"""
function mixed_freq_var(y::AbstractMatrix{<:Real}, p::Int,
                        freq_map::Vector{Symbol};
                        constant::Bool=true,
                        max_iter::Int=100, tol::Float64=1e-6)
    T, K = size(y)
    length(freq_map) == K || throw(ArgumentError("freq_map length must equal K"))
    all(f -> f in (:high, :stock, :flow), freq_map) ||
        throw(ArgumentError("freq_map entries must be :high, :stock, or :flow"))

    # Identify flow variables — they need accumulator states
    flow_idx = findall(f -> f == :flow, freq_map)
    n_flow = length(flow_idx)

    # ── Initial VAR estimate: fill NaN with column mean, then run OLS ──
    y_fill = copy(Float64.(y))
    for k in 1:K
        valid = .!isnan.(y_fill[:, k])
        if any(valid)
            m = mean(y_fill[valid, k])
            y_fill[.!valid, k] .= m
        end
    end
    v = var_estimate(y_fill, p; constant=constant)
    Phi_curr = copy(v.Phi)
    Sigma_curr = copy(v.Sigma)

    logL_prev = -Inf
    converged = false
    niter = 0

    for iter in 1:max_iter
        niter = iter

        # ── E‑step: Kalman filter/smoother on mixed‑freq state‑space ──
        A_mf, B_mf, C_mf, const_mf, ns_mf = _build_mixed_freq_ss(
            Phi_curr, Sigma_curr, K, p, freq_map)

        # Build observation matrix for the Kalman filter
        # y_kf is T × K with NaNs for unobserved
        y_kf = Float64.(y)

        # For flow variables: the observation is the accumulated sum,
        # so we need to map it through the accumulator observation row.
        # The Kalman filter handles NaN automatically.
        # The observation equation is y_t = C_mf * state_t (+const), but
        # C_mf maps flow vars to their accumulator state.
        # We pass the raw y with NaN — the filter handles masking.

        kf = _kalman_filter_mf(A_mf, B_mf, C_mf, const_mf, Sigma_curr,
                               y_kf, ns_mf, K, p, freq_map, flow_idx)

        logL = kf.logL

        # Check convergence
        if abs(logL - logL_prev) < tol && iter > 1
            converged = true
            break
        end
        logL_prev = logL

        # ── M‑step: re‑estimate VAR from smoothed states ──
        # Extract smoothed high‑freq variables from states
        y_smooth = kf.smoothed[:, 1:K]

        # For flow variables, replace accumulated obs with the smoothed
        # underlying high‑freq values
        for k in flow_idx
            y_smooth[:, k] = kf.smoothed[:, k]
        end

        v_new = var_estimate(y_smooth, p; constant=constant)
        Phi_curr = copy(v_new.Phi)
        Sigma_curr = copy(v_new.Sigma)
    end

    # Final smoothed series
    A_mf, B_mf, C_mf, const_mf, ns_mf = _build_mixed_freq_ss(
        Phi_curr, Sigma_curr, K, p, freq_map)
    kf_final = _kalman_filter_mf(A_mf, B_mf, C_mf, const_mf, Sigma_curr,
                                  Float64.(y), ns_mf, K, p, freq_map, flow_idx)

    y_interp = kf_final.smoothed[:, 1:K]

    return MixedFreqVARResult(Phi_curr, Sigma_curr, y_interp,
                              kf_final.smoothed, kf_final.logL,
                              niter, converged)
end

"""
Build augmented state‑space for mixed‑frequency VAR.

State vector: [y₁(t), …, yₖ(t), y₁(t−1), …, yₖ(t−p+1), acc_flow₁, …, acc_flowₙ]

For stock variables: observed when not NaN; observation = state.
For flow variables: accumulator sums underlying values between observations;
    observation = accumulator when observed, then accumulator resets.
"""
function _build_mixed_freq_ss(Phi::AbstractMatrix, Sigma::AbstractMatrix,
                              K::Int, p::Int, freq_map::Vector{Symbol})
    flow_idx = findall(f -> f == :flow, freq_map)
    n_flow = length(flow_idx)

    ns_var = K * p           # VAR companion states
    ns = ns_var + n_flow     # + accumulator states for flow

    # ── Transition ──
    A = zeros(ns, ns)
    # VAR companion block
    A[1:K, 1:ns_var] = Phi[1:K*p, :]'
    if p > 1
        A[K+1:ns_var, 1:K*(p-1)] = I(K * (p - 1))
    end
    # Accumulator: acc(t) = acc(t−1) + y_flow(t)
    # This will be handled specially in the filter (reset on observation)
    for (i, k) in enumerate(flow_idx)
        A[ns_var+i, ns_var+i] = 1.0   # acc carries forward
        A[ns_var+i, k] = 1.0          # adds current y_k
    end

    # ── Shock loading ──
    B = zeros(ns, K)
    B[1:K, :] = I(K)

    # ── Observation ──
    C = zeros(K, ns)
    for k in 1:K
        if freq_map[k] == :flow
            # Observe the accumulator for flow variables
            fi = findfirst(==(k), flow_idx)
            C[k, ns_var+fi] = 1.0
        else
            # Stock or high‑freq: observe the state directly
            C[k, k] = 1.0
        end
    end

    # ── Constant ──
    const_vec = zeros(ns)
    if size(Phi, 1) > K * p
        const_vec[1:K] = Phi[K*p+1, :]
    end

    return A, B, C, const_vec, ns
end

"""
Kalman filter for mixed‑frequency state‑space with flow accumulator resets.
"""
function _kalman_filter_mf(A, B, C, const_vec, Sigma,
                           y, ns, K, p, freq_map, flow_idx)
    T = size(y, 1)
    ny = K
    data = y'

    # Pre‑allocate
    stt = zeros(ns, T + 1)
    ptt = zeros(ns, ns, T + 1)
    logLnc = zeros(T)

    # Diffuse initialization
    ptt[:, :, 1] = 10.0 * I(ns)

    ns_var = K * p
    n_flow = length(flow_idx)

    # Track accumulator: which flow variables had their last observation
    # We need to reset accumulator after each observation
    last_obs_flow = zeros(Int, n_flow)  # last time each flow var was observed

    BΣB = B * Sigma * B'

    for t in 1:T
        yt = data[:, t]

        # ── Reset accumulators for flow variables that were observed at t−1 ──
        # If flow var was observed last period, reset accumulator before prediction
        state_pred = A * stt[:, t]
        P_pred = A * ptt[:, :, t] * A' + BΣB

        # For flow variables: reset accumulator in prediction step
        # if the variable was observed at previous period
        for (i, k) in enumerate(flow_idx)
            if t > 1 && !isnan(data[k, t - 1])
                # Accumulator was observed; reset it
                state_pred[ns_var+i] = state_pred[k]  # start fresh: just current value
                # Zero out accumulator cross‑covariances via reset
                P_pred[ns_var+i, :] .= P_pred[k, :]
                P_pred[:, ns_var+i] .= P_pred[:, k]
                P_pred[ns_var+i, ns_var+i] = P_pred[k, k]
            end
        end

        # Handle missing observations
        obs_mask = .!isnan.(yt)
        obs_idx = findall(obs_mask)
        yt_obs = yt[obs_mask]
        Ct = C[obs_mask, :]
        const_obs = const_vec[1:ny][obs_mask]

        if isempty(obs_idx)
            stt[:, t+1] = state_pred
            ptt[:, :, t+1] = P_pred
            continue
        end

        # Innovation
        yt_pred = Ct * state_pred + const_obs
        v = yt_obs - yt_pred

        # Innovation covariance
        F = Ct * P_pred * Ct'
        F = Hermitian(F + 1e-12 * I(length(obs_idx)))

        # Kalman gain
        Finv = inv(F)
        KG = P_pred * Ct' * Finv

        # Update
        stt[:, t+1] = state_pred + KG * v
        ptt[:, :, t+1] = P_pred - KG * Ct * P_pred

        # Log‑likelihood
        nd = length(obs_idx)
        logLnc[t] = -0.5 * (nd * log(2π) + logdet(F) + v' * Finv * v)
    end

    logL = sum(logLnc)

    # RTS smoother on augmented state
    s_smooth = _rts_smoother(stt, ptt, A, T, ns; Q=BΣB + 1e-10 * I(ns))

    return (logL=logL,
            states=stt[:, 2:end]',
            smoothed=s_smooth')
end

# ─── Nowcasting via BVAR + Kalman ──────────────────────────────────────────────

"""
    nowcast_bvar(data_panels, bvar_result; n_draws=100, target_index=nothing, rng)

Produce nowcasts using the BVAR posterior and Kalman filter/smoother.

# Arguments
- `data_panels`:   T × K × n_datasets array of data, with NaN for missing observations.
- `bvar_result`:   BVARResult from `bvar()`.
- `n_draws`:       number of random posterior draws to use.
- `target_index`:  index of the target variable for nowcasting (default: 1).
- `rng`:           random number generator.

# Returns
Named tuple `(nowcast, forecasts_no_shock, forecasts_with_shock)` where
`nowcast` is T × n_draws × n_datasets.
"""
function nowcast_bvar(data_panels::AbstractArray{<:Real, 3},
                      bvar_result;
                      n_draws::Int=100,
                      target_index::Int=1,
                      rng::AbstractRNG=Random.default_rng())
    T, ny, nD = size(data_panels)
    lags = bvar_result.nlags
    nk = size(bvar_result.Phi_draws, 1)
    K_total = size(bvar_result.Phi_draws, 3)

    n_draws = min(n_draws, K_total)
    draw_indices = rand(rng, 1:K_total, n_draws)

    NowCast = fill(NaN, T, n_draws, nD)

    for j in 1:nD
        data = data_panels[:, :, j]
        for (i, idx) in enumerate(draw_indices)
            Phi = bvar_result.Phi_draws[1:nk, :, idx]
            Sigma = bvar_result.Sigma_draws[:, :, idx]

            # Run Kalman filter/smoother with NaN handling
            kf_result = kalman_filter(Phi, Sigma, data)
            # Extract smoothed values for target variable
            NowCast[:, i, j] = kf_result.smoothed[:, target_index]
        end
    end

    return (nowcast=NowCast, draw_indices=draw_indices)
end
