#=
filters.jl — Time‑series filters: HP, BK, CF, Hamilton
Port of MATLAB Hpfilter.m, bkfilter.m, cffilter.m, hamfilter.m
=#

"""
    hp_filter(y, λ=1600)

Hodrick–Prescott filter.

# Returns
Named tuple `(trend, cycle)`.
"""
function hp_filter(y::AbstractVector, λ::Real = 1600)
    T = length(y)
    # Build pentadiagonal matrix
    a = 6λ + 1
    b = -4λ
    c = λ

    d_main = fill(a, T)
    d_sup1 = fill(b, T - 1)
    d_sup2 = fill(c, T - 2)

    M = diagm(0 => d_main, 1 => d_sup1, -1 => d_sup1,
        2 => d_sup2, -2 => d_sup2)

    # Fix corners
    M[1, 1] = 1 + λ;
    M[1, 2] = -2λ
    M[2, 1] = -2λ;
    M[2, 2] = 5λ + 1
    M[T - 1, T - 1] = 5λ + 1;
    M[T - 1, T] = -2λ
    M[T, T - 1] = -2λ;
    M[T, T] = 1 + λ

    trend = M \ y
    cycle = y - trend
    return (trend = trend, cycle = cycle)
end

"""
    hp_filter(Y::AbstractMatrix, λ=1600)

Column‑wise HP filter for matrices.
"""
function hp_filter(Y::AbstractMatrix, λ::Real = 1600)
    T, K = size(Y)
    trend = similar(Y)
    cycle = similar(Y)
    for k in 1:K
        r = hp_filter(Y[:, k], λ)
        trend[:, k] = r.trend
        cycle[:, k] = r.cycle
    end
    return (trend = trend, cycle = cycle)
end

"""
    bk_filter(X, pl=6, pu=32)

Baxter–King band‑pass filter.

# Arguments
- `X`:  T × 1 vector (or T × K matrix).
- `pl`: minimum period of oscillation.
- `pu`: maximum period of oscillation.

# Returns
Filtered series (same dimensions as X).
"""
function bk_filter(X::AbstractVector, pl::Int = 6, pu::Int = 32)
    T = length(X)

    if pu <= pl
        throw(ArgumentError("pu must be larger than pl"))
    end
    if pl < 2
        pl = 2
    end

    # Remove drift
    drift = (X[T] - X[1]) / (T - 1)
    Xun = X .- (0:(T - 1)) .* drift

    b = 2π / pl
    a = 2π / pu
    bnot = (b - a) / π

    j = 1:T
    B = vcat(bnot, (sin.(j .* b) .- sin.(j .* a)) ./ (j .* π))

    # Build symmetric AA matrix
    AA = zeros(T, T)
    for i in 1:T
        for k in i:min(i + T - 1, T)
            idx = k - i + 1
            if idx <= length(B)
                AA[i, k] += B[idx]
                if i != k
                    AA[k, i] += B[idx]
                end
            end
        end
    end

    # Fix boundary conditions
    bhat = bnot / 2
    AA[1, 1] = bhat
    AA[T, T] = bhat

    for i in 1:(T - 1)
        AA[i + 1, 1] = AA[i, 1] - B[min(i+1, length(B))]
        AA[T - i, T] = AA[i, 1] - B[min(i+1, length(B))]
    end

    return AA * Xun
end

function bk_filter(X::AbstractMatrix, pl::Int = 6, pu::Int = 32)
    T, K = size(X)
    out = similar(X)
    for k in 1:K
        out[:, k] = bk_filter(X[:, k], pl, pu)
    end
    return out
end

"""
    cf_filter(X, pl=6, pu=32; root=true, drift=true)

Christiano–Fitzgerald (1999) asymmetric band‑pass filter (Random Walk default).

# Returns
Filtered series.
"""
function cf_filter(X::AbstractVector, pl::Int = 6, pu::Int = 32;
        root::Bool = true, drift::Bool = true)
    T = length(X)
    b = 2π / pl
    a = 2π / pu

    # Remove drift if requested
    if drift
        d = (X[T] - X[1]) / (T - 1)
        Xun = X .- (0:(T - 1)) .* d
    else
        Xun = copy(X)
    end

    # Ideal filter weights
    j = 1:2T
    B = vcat((b - a) / π, (sin.(j .* b) .- sin.(j .* a)) ./ (j .* π))

    # Construct AA for asymmetric filter (Default: Random Walk)
    AA = zeros(T, T)
    for i in 1:T
        np = i - 1      # observations before
        nf = T - i      # observations after

        for k in 1:T
            jj = abs(i - k)
            if jj < length(B)
                AA[i, k] = B[jj + 1]
            end
        end

        # Adjustment for unit root: last coefficient absorbs remainder
        if root
            # Sum of ideal weights from np onward
            remainder = 0.0
            for m in max(nf + 1, 1):2T
                if m + 1 <= length(B)
                    remainder += B[m + 1]
                end
            end
            AA[i, T] += remainder / (2π)
            AA[i, 1] += remainder / (2π)
        end
    end

    # Normalize rows
    for i in 1:T
        s = sum(AA[i, :])
        if abs(s) > 1e-10
            target = B[1]  # ideal at lag 0
            # adjust diagonal
            AA[i, i] += (target - s)
        end
    end

    return AA * Xun
end

function cf_filter(X::AbstractMatrix, pl::Int = 6, pu::Int = 32;
        root::Bool = true, drift::Bool = true)
    T, K = size(X)
    out = similar(X)
    for k in 1:K
        out[:, k] = cf_filter(X[:, k], pl, pu; root = root, drift = drift)
    end
    return out
end

"""
    hamilton_filter(X, h=8, d=4; constant=true)

Hamilton (2018) detrending filter using direct regression.

    y(t+h) = a₀ + a₁ y(t) + a₂ y(t-1) + ... + a_d y(t-d+1) + e(t+h)

# Returns
Named tuple `(trend, cycle)` with NaN‑padding for initial observations.
"""
function hamilton_filter(X::AbstractVector, h::Int = 8, d::Int = 4;
        constant::Bool = true)
    T = length(X)
    yh = X[(d + h):T]
    Teff = length(yh)

    # Build RHS
    R = constant ? ones(Teff, 1) : Matrix{Float64}(undef, Teff, 0)
    for j in 1:d
        R = hcat(R, X[(d + 1 - j):(T - h + 1 - j)])
    end

    # OLS
    beta = R \ yh
    trend_part = R * beta
    cycle_part = yh - trend_part

    # Pad with NaN
    trend = fill(NaN, T)
    cycle = fill(NaN, T)
    trend[(d + h):T] = trend_part
    cycle[(d + h):T] = cycle_part

    return (trend = trend, cycle = cycle)
end

function hamilton_filter(X::AbstractMatrix, h::Int = 8, d::Int = 4;
        constant::Bool = true)
    T, K = size(X)
    trend = fill(NaN, T, K)
    cycle = fill(NaN, T, K)
    for k in 1:K
        r = hamilton_filter(X[:, k], h, d; constant = constant)
        trend[:, k] = r.trend
        cycle[:, k] = r.cycle
    end
    return (trend = trend, cycle = cycle)
end
