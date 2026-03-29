#=
business_cycle.jl — Peak/trough dating and Beveridge‑Nelson decomposition
=#

"""
    bry_boschan(y; window=5, min_phase=2, min_cycle=5)

Bry‑Boschan (1971) nonparametric turning‑point algorithm for a univariate
time series.

# Arguments
- `y::AbstractVector`:   univariate time series.
- `window::Int`:         half‑window for local extremum detection (default 5).
- `min_phase::Int`:      minimum expansion/contraction length (default 2).
- `min_cycle::Int`:      minimum full‑cycle length peak‑to‑peak or trough‑to‑trough (default 5).

# Returns
`BusinessCycleResult` with vectors of peak and trough indices (1‑based).
"""
function bry_boschan(y::AbstractVector{<:Real};
                     window::Int=5, min_phase::Int=2, min_cycle::Int=5)
    T = length(y)

    # ── Step 1: find candidate local maxima and minima ──
    cand_peaks = Int[]
    cand_troughs = Int[]
    for t in (window + 1):(T - window)
        lo = t - window
        hi = t + window
        if all(y[t] >= y[s] for s in lo:hi) && any(y[t] > y[s] for s in lo:hi if s != t)
            push!(cand_peaks, t)
        end
        if all(y[t] <= y[s] for s in lo:hi) && any(y[t] < y[s] for s in lo:hi if s != t)
            push!(cand_troughs, t)
        end
    end

    # ── Step 2: merge candidates into alternating sequence ──
    # Build combined list of (index, type) sorted by index
    events = Tuple{Int,Symbol}[]
    for p in cand_peaks
        push!(events, (p, :peak))
    end
    for t in cand_troughs
        push!(events, (t, :trough))
    end
    sort!(events; by=first)

    # Enforce alternation: if same type repeats, keep the more extreme
    alt = Tuple{Int,Symbol}[]
    for ev in events
        if isempty(alt) || alt[end][2] != ev[2]
            push!(alt, ev)
        else
            # Same type – keep the more extreme
            prev = alt[end]
            if ev[2] == :peak
                if y[ev[1]] > y[prev[1]]
                    alt[end] = ev
                end
            else  # trough
                if y[ev[1]] < y[prev[1]]
                    alt[end] = ev
                end
            end
        end
    end

    # ── Step 3: enforce minimum phase length ──
    changed = true
    while changed
        changed = false
        i = 1
        while i < length(alt)
            if alt[i + 1][1] - alt[i][1] < min_phase
                # Remove the less extreme of the pair
                if alt[i][2] == :peak
                    # peak followed by trough too quickly – drop the lower‑amplitude one
                    if y[alt[i][1]] >= y[alt[i + 1][1]]
                        deleteat!(alt, i + 1)
                    else
                        deleteat!(alt, i)
                    end
                else
                    if y[alt[i][1]] <= y[alt[i + 1][1]]
                        deleteat!(alt, i + 1)
                    else
                        deleteat!(alt, i)
                    end
                end
                changed = true
                # Re‑enforce alternation after removal
                alt = _enforce_alternation(alt, y)
            else
                i += 1
            end
        end
    end

    # ── Step 4: enforce minimum cycle length ──
    changed = true
    while changed
        changed = false
        # Check peak‑to‑peak and trough‑to‑trough distances
        i = 1
        while i + 2 <= length(alt)
            if alt[i][2] == alt[i + 2][2]  # same type
                dist = alt[i + 2][1] - alt[i][1]
                if dist < min_cycle
                    # Remove the less extreme pair
                    if alt[i][2] == :peak
                        drop = y[alt[i][1]] >= y[alt[i + 2][1]] ? i + 2 : i
                    else
                        drop = y[alt[i][1]] <= y[alt[i + 2][1]] ? i + 2 : i
                    end
                    # Also remove the intermediate event
                    mid = i + 1
                    if drop < mid
                        deleteat!(alt, [drop, mid])
                    else
                        deleteat!(alt, [mid, drop])
                    end
                    changed = true
                    alt = _enforce_alternation(alt, y)
                    break
                end
            end
            i += 1
        end
    end

    peaks = [ev[1] for ev in alt if ev[2] == :peak]
    troughs = [ev[1] for ev in alt if ev[2] == :trough]

    return BusinessCycleResult(peaks, troughs)
end

"""
Re‑enforce alternation after removing events.
"""
function _enforce_alternation(events::Vector{Tuple{Int,Symbol}}, y::AbstractVector)
    alt = Tuple{Int,Symbol}[]
    for ev in events
        if isempty(alt) || alt[end][2] != ev[2]
            push!(alt, ev)
        else
            prev = alt[end]
            if ev[2] == :peak
                if y[ev[1]] > y[prev[1]]
                    alt[end] = ev
                end
            else
                if y[ev[1]] < y[prev[1]]
                    alt[end] = ev
                end
            end
        end
    end
    return alt
end

"""
    bn_decomposition(y, p; constant=true)

Beveridge‑Nelson (1981) trend/cycle decomposition via a VAR on first
differences.

# Arguments
- `y::AbstractMatrix`:  T × K data matrix.
- `p::Int`:             VAR lag length on Δy.
- `constant::Bool`:     include constant in the VAR (default `true`).

# Returns
`BNDecompResult` with `permanent` and `transitory` components (T × K).

The permanent component is defined as:
    yᵗᴾ = yₜ + C(1) Σⱼ₌₁^∞ E[Δyₜ₊ⱼ | Ωₜ]
where C(1) = (I − Φ₁ − ⋯ − Φₚ)⁻¹ is the long‑run multiplier.
"""
function bn_decomposition(y::AbstractMatrix{<:Real}, p::Int;
                           constant::Bool=true)
    T, K = size(y)
    T > p + 1 || throw(ArgumentError("need T > p + 1"))

    # First differences
    dy = diff(y; dims=1)   # (T−1) × K

    # Estimate VAR(p) on differences
    v = var_estimate(dy, p; constant=constant)

    # Extract AR coefficient matrices Φ₁, …, Φₚ  (each K × K)
    Phi_ar = v.Phi[1:K*p, :]   # (Kp) × K

    # Long‑run multiplier C(1) = (I − Σ Φₗ)⁻¹
    sum_Phi = zeros(K, K)
    for l in 1:p
        sum_Phi += Phi_ar[(l-1)*K+1:l*K, :]'
    end
    C1 = inv(I(K) - sum_Phi)

    # ── Companion‑form analytical BN decomposition ──
    # State vector: sₜ = [Δyₜ', Δyₜ₋₁', …, Δyₜ₋ₚ₊₁']'
    # Transition: sₜ = c + F sₜ₋₁ + Gεₜ
    ns = K * p
    F = companion_form(Phi_ar, K, p)

    # Intercept in companion form
    c_comp = zeros(ns)
    if constant
        c_comp[1:K] = v.Phi[K*p+1, :]
    end

    # BN forecast revision: C(1) * J * (I − F)⁻¹ * (sₜ − long‑run mean)
    # where J = [Iₖ 0 … 0] selects first K rows
    J = zeros(K, ns)
    J[1:K, 1:K] = I(K)

    ImF = I(ns) - F
    ImF_inv = inv(ImF)

    # Long‑run mean of state (if constant)
    s_bar = ImF_inv * c_comp

    # Build state vectors from the VAR residuals
    # The VAR was on dy, so we have residuals and the fitted states
    # We need the state at each time t = p+1, …, T−1 (in Δy indices)
    Tdy = size(dy, 1)
    Tvar = v.nobs  # = Tdy − p

    permanent = copy(y)
    transitory = zeros(T, K)

    # For t = 1 to p (before VAR estimation starts), set transitory = 0
    # For t = p+1+1 to T (corresponds to Δy indices p+1:Tdy):
    for t_var in 1:Tvar
        # Build state vector at this time
        t_dy = t_var + p   # index in dy
        s_t = zeros(ns)
        for l in 1:p
            if t_dy - l + 1 >= 1
                s_t[(l-1)*K+1:l*K] = dy[t_dy-l+1, :]
            end
        end

        # BN revision: sum of expected future changes
        # = C(1) * J * (I−F)⁻¹ * F * (sₜ − s̄)
        revision = C1 * J * ImF_inv * F * (s_t - s_bar)

        t_y = t_dy + 1   # index in y (dy[t] = y[t+1] − y[t])
        transitory[t_y, :] = -revision
    end

    permanent = y - transitory

    return BNDecompResult(permanent, transitory)
end
