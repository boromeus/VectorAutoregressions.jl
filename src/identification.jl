#=
identification.jl — Structural identification schemes
=#

"""
    parse_sign_restriction(s, K)

Parse a sign restriction string like `"y(1,2,3)>0"` into
`(var_idx, horizon, shock_idx, sign)`.
"""
function parse_sign_restriction(s::AbstractString, K::Int)
    s = strip(s)
    # Match patterns like y(1,2,3)>0  or  y(1,1:4,1)<0
    m_var = match(r"^y\((\d+)\s*,\s*(.+?)\s*,\s*(\d+)\)\s*([><])\s*0$", s)
    if m_var !== nothing
        var_idx = parse(Int, m_var.captures[1])
        hor_str = m_var.captures[2]
        shock_idx = parse(Int, m_var.captures[3])
        sign_char = m_var.captures[4]
        sign_val = sign_char == ">" ? 1 : -1

        # Parse horizon: could be "2" or "1:4"
        horizons = if contains(hor_str, ":")
            parts = split(hor_str, ":")
            collect(parse(Int, parts[1]):parse(Int, parts[2]))
        else
            [parse(Int, hor_str)]
        end
        return [(var=var_idx, horizon=h, shock=shock_idx, sign=sign_val) for h in horizons]
    end

    # Narrative restrictions: v(timeperiods, shock)>0  or  v(timeperiods, shock)<0
    m_narr = match(r"^v\((.+?)\s*,\s*(\d+)\)\s*([><])\s*0$", s)
    if m_narr !== nothing
        time_str = m_narr.captures[1]
        shock_idx = parse(Int, m_narr.captures[2])
        sign_char = m_narr.captures[3]
        sign_val = sign_char == ">" ? 1 : -1

        times = if contains(time_str, ":")
            parts = split(time_str, ":")
            collect(parse(Int, parts[1]):parse(Int, parts[2]))
        else
            [parse(Int, time_str)]
        end
        return [(type=:narrative, times=times, shock=shock_idx, sign=sign_val)]
    end

    error("Cannot parse restriction: '$s'")
end

"""
    check_sign_restrictions(ir, restrictions, K)

Check whether the IRF array satisfies all sign restrictions.
`ir` is K × hor × K. Returns `true` if all satisfied.
"""
function check_sign_restrictions(ir::AbstractArray{<:Real,3}, restrictions::Vector{String}, K::Int)
    for s in restrictions
        parsed = parse_sign_restriction(s, K)
        for r in parsed
            if !haskey(r, :type)  # standard sign restriction
                if r.horizon > size(ir, 2)
                    return false
                end
                val = ir[r.var, r.horizon, r.shock]
                if r.sign > 0 && val < 0
                    return false
                elseif r.sign < 0 && val > 0
                    return false
                end
            end
        end
    end
    return true
end

"""
    check_narrative_restrictions(structural_shocks, narrative, K)

Check narrative restrictions on structural shocks.
`structural_shocks` is T × K.
"""
function check_narrative_restrictions(structural_shocks::AbstractMatrix,
                                     narrative::Vector{String}, K::Int)
    for s in narrative
        parsed = parse_sign_restriction(s, K)
        for r in parsed
            if haskey(r, :type) && r.type == :narrative
                for t in r.times
                    if t < 1 || t > size(structural_shocks, 1)
                        return false
                    end
                    val = structural_shocks[t, r.shock]
                    if r.sign > 0 && val < 0
                        return false
                    elseif r.sign < 0 && val > 0
                        return false
                    end
                end
            end
        end
    end
    return true
end

"""
    max_horizon_sign(restrictions, K)

Find the maximum horizon referenced in sign restrictions.
"""
function max_horizon_sign(restrictions::Vector{String}, K::Int)
    max_h = 1
    for s in restrictions
        parsed = parse_sign_restriction(s, K)
        for r in parsed
            if !haskey(r, :type)
                max_h = max(max_h, r.horizon)
            end
        end
    end
    return max_h
end

"""
    _check_sign_with_flip(ir, restrictions, K)

Check sign restrictions on `ir` and also on `-ir` (matching MATLAB
`checkrestrictions2`).  Returns `(satisfied, fsign)` where `fsign` is
`1` if the original IRFs satisfy or `-1` if the negated IRFs satisfy.
"""
function _check_sign_with_flip(ir::AbstractArray{<:Real,3},
                               restrictions::Vector{String}, K::Int)
    if check_sign_restrictions(ir, restrictions, K)
        return true, 1
    end
    if check_sign_restrictions(-ir, restrictions, K)
        return true, -1
    end
    return false, 1
end

"""
    irf_sign_restriction(Phi, Sigma, hor, restrictions; max_rotations=30000, rng)

Compute IRFs under sign restrictions using acceptance sampling.

Matches MATLAB `iresponse_sign.m` / `checkrestrictions2`:
both the original and negated IRFs are checked, doubling the effective
acceptance rate.  When the negated set satisfies, the rotation is
flipped (`fsign * Omega`) and the IRFs are scaled by `fsign`.
"""
function irf_sign_restriction(Phi::AbstractMatrix, Sigma::AbstractMatrix,
                              hor::Int, restrictions::Vector{String};
                              max_rotations::Int=30000,
                              rng::AbstractRNG=Random.default_rng())
    K = size(Sigma, 1)
    hor0 = max_horizon_sign(restrictions, K)

    for tol in 1:max_rotations
        Omega = generate_rotation_matrix(K; rng=rng)
        ir = compute_irf(Phi, Sigma, max(hor, hor0); Omega=Omega)
        satisfied, fsign = _check_sign_with_flip(ir, restrictions, K)
        if satisfied
            Omeg = fsign * Omega
            ir_full = compute_irf(Phi, Sigma, hor; Omega=Omeg)
            return ir_full, Omeg
        end
    end

    @warn "Could not find rotation satisfying sign restrictions after $max_rotations attempts"
    return fill(NaN, K, hor, K), fill(NaN, K, K)
end

"""
    irf_narrative_sign(residuals, Phi, Sigma, hor, signs, narrative; ...)

Compute IRFs with combined sign and narrative restrictions.
"""
function irf_narrative_sign(residuals::AbstractMatrix,
                            Phi::AbstractMatrix, Sigma::AbstractMatrix,
                            hor::Int, signs::Vector{String}, narrative::Vector{String};
                            max_rotations::Int=30000,
                            rng::AbstractRNG=Random.default_rng())
    K = size(Sigma, 1)
    A = cholesky(Hermitian(Sigma)).L
    hor0 = max_horizon_sign(signs, K)

    for tol in 1:max_rotations
        Omega = generate_rotation_matrix(K; rng=rng)
        ir = compute_irf(Phi, Sigma, max(hor, hor0); Omega=Omega)

        if check_sign_restrictions(ir, signs, K)
            # Compute structural shocks
            v = residuals / (Omega' * A')
            if check_narrative_restrictions(v, narrative, K)
                ir_full = compute_irf(Phi, Sigma, hor; Omega=Omega)
                return ir_full, Omega
            end
        end
    end

    @warn "Could not find rotation satisfying narrative+sign restrictions after $max_rotations attempts"
    return fill(NaN, K, hor, K), fill(NaN, K, K)
end

# ─── Zero + Sign Restrictions ──────────────────────────────────────────────────

"""
    parse_zero_sign_restrictions(restrictions, K)

Parse zero+sign restriction strings into matrices `f` and `sr`.
Supports:
- `"y(a,b)=1"` or `"y(a,b)=-1"` for sign restrictions
- `"ys(a,b)=0"` for short‑run zero restrictions
- `"yr(a,1,b)=0"` for long‑run zero restrictions
"""
function parse_zero_sign_restrictions(restrictions::Vector{String}, K::Int)
    yr = ones(K, K)   # long‑run zeros: 0 where restricted
    ys = ones(K, K)   # short‑run zeros: 0 where restricted
    sr = fill(NaN, K, K)  # sign restrictions

    for s in restrictions
        s = strip(s)
        # Short‑run zero: ys(a,b)=0
        m = match(r"^ys\((\d+)\s*,\s*(\d+)\)\s*=\s*0$", s)
        if m !== nothing
            a, b = parse(Int, m.captures[1]), parse(Int, m.captures[2])
            ys[a, b] = 0.0
            continue
        end
        # Long‑run zero: yr(a,1,b)=0
        m = match(r"^yr\((\d+)\s*,\s*\d+\s*,\s*(\d+)\)\s*=\s*0$", s)
        if m !== nothing
            a, b = parse(Int, m.captures[1]), parse(Int, m.captures[2])
            yr[a, b] = 0.0
            continue
        end
        # Sign: y(a,b)=1 or y(a,b)=-1
        m = match(r"^y\((\d+)\s*,\s*(\d+)\)\s*=\s*(-?1)$", s)
        if m !== nothing
            a, b = parse(Int, m.captures[1]), parse(Int, m.captures[2])
            sr[a, b] = parse(Float64, m.captures[3])
            continue
        end
        error("Cannot parse zero/sign restriction: '$s'")
    end

    f = vcat(ys, yr)
    return f, sr
end

"""
    findQs(K, f)

Find the Q matrices describing linear restrictions on impact matrix columns.
Based on Arias et al. (2018).
"""
function findQs(K::Int, f::AbstractMatrix)
    E = I(K)
    Q_cells = Vector{Matrix{Float64}}(undef, K)
    ranks = Vector{Int}(undef, K)

    for ii in 1:K
        diag_vec = f * E[:, ii]
        Q_init = Diagonal(Float64.(diag_vec .== 0))
        ranks[ii] = rank(Q_init)
        # Keep only nonzero rows
        nonzero_rows = findall(vec(sum(abs.(Matrix(Q_init)), dims=2)) .> 0)
        Q_cells[ii] = Matrix(Q_init)[nonzero_rows, :]
    end

    ord = sortperm(ranks, rev=true)
    new_ranks = ranks[ord]

    # Check identification
    flag = 0
    for ii in 1:K
        if new_ranks[ii] > K - ii
            flag = 1  # over‑identified
            break
        elseif new_ranks[ii] < K - ii
            flag = -1  # under‑identified
        end
    end

    # Build index for reordering
    index = similar(ord)
    for ii in 1:K
        index[ord[ii]] = ii
    end

    return Q_cells[ord], index, flag
end

"""
    findP(C, B, Q_cells, p, K, index)

Find rotation matrix P satisfying zero restrictions.
"""
function findP(C::AbstractMatrix, B::AbstractMatrix, Q_cells::Vector{Matrix{Float64}},
               p::Int, K::Int, index::Vector{Int})
    L0 = C

    # Compute long‑run multiplier
    beta = zeros(K, K)
    for ii in 1:p
        beta += B[(ii-1)*K+1:ii*K, :]'
    end
    Linf = (I(K) - beta) \ C

    F_mat = vcat(L0, Linf)

    P = zeros(K, K)
    for ii in 1:K
        if ii == 1
            Qtilde = Q_cells[ii] * F_mat
        else
            Qtilde = vcat(Q_cells[ii] * F_mat, P')
        end
        QQ, RR = qr(Qtilde')
        # Must use full Q (Julia's Matrix(QQ) gives thin Q for non‑square input)
        Q_full = QQ * Matrix{Float64}(I, K, K)
        P_temp = Q_full[:, end]
        P[:, ii] = P_temp
    end

    return P[:, index]
end

"""
    irf_zero_sign(Phi, Sigma, hor, p, restrictions; var_pos, max_draws, max_attempts, rng)

Compute IRFs with zero and sign restrictions (Arias et al. 2018).
"""
function irf_zero_sign(Phi::AbstractMatrix, Sigma::AbstractMatrix,
                       hor::Int, p::Int, restrictions::Vector{String};
                       var_pos::Vector{Int}=ones(Int, size(Sigma, 1)),
                       max_draws::Int=1, max_attempts::Int=10000,
                       rng::AbstractRNG=Random.default_rng())
    K = size(Sigma, 1)
    f, sr = parse_zero_sign_restrictions(restrictions, K)
    C1 = cholesky(Hermitian(Sigma)).L

    Q_cells, index, flag = findQs(K, f)
    if flag == 1
        error("Rank condition not satisfied: model is over‑identified")
    end

    ir = fill(NaN, K, hor, K)
    Omeg = fill(NaN, K, K)

    Phi_ar = Phi[1:K*p, :]
    F_comp = companion_form(Phi_ar, K, p)

    for attempt in 1:max_attempts
        # Generate random factorization
        C = C1 * generate_rotation_matrix(K; rng=rng)
        P = findP(C, Phi_ar, Q_cells, p, K, index)
        W = C * P

        valid = true
        for jj in 1:K
            chk = W[:, jj]
            sr_idx = findall(.!isnan.(sr[:, jj]))
            if !isempty(sr_idx)
                tmp = sign.(chk[sr_idx]) .- sr[sr_idx, jj]
                if any(tmp .!= 0)
                    valid = false
                    break
                end
            end
        end

        if valid
            for jj in 1:K
                V = zeros(K * p, hor)
                V[1:K, 1] = W[:, jj]
                for ii in 2:hor
                    V[:, ii] = F_comp * V[:, ii-1]
                end
                ir[:, :, jj] = V[1:K, :]
            end
            Omeg = C1 \ W
            return ir, Omeg
        end
    end

    @warn "Could not find rotation satisfying zero+sign restrictions"
    return ir, Omeg
end
