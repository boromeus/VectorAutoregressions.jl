#=
historical_decomp.jl — Historical decomposition of observables into structural shocks
=#

"""
    historical_decomposition(bvar_result; Omega=I, draw=:mean, tol=1e-7)

Decompose observed data into contributions from each structural shock,
exogenous variables, deterministic components and initial conditions.

Matches MATLAB `histdecomp.m`: decomposition includes structural shocks,
then exogenous variable contributions (if any), then deterministic
(constant + initial conditions).

# Arguments
- `bvar_result::BVARResult`: estimated BVAR.
- `Omega::AbstractMatrix`:   rotation matrix (default identity = Cholesky).
- `draw`:                    `:mean`, `:median`, or an integer draw index.
- `tol::Float64`:            tolerance for consistency check.

# Returns
`HistDecompResult` with decomposition T × K × (K + nexog + 1) and structural shocks.
"""
function historical_decomposition(result::BVARResult;
                                  Omega::Union{Nothing,AbstractMatrix}=nothing,
                                  draw::Union{Symbol,Int}=:mean,
                                  tol::Float64=1e-7)
    K = result.nvar
    p = result.nlags

    # Select parameters
    if draw isa Int
        u     = result.e_draws[:, :, draw]
        alpha = result.Phi_draws[:, :, draw]
        Sigma = result.Sigma_draws[:, :, draw]
    elseif draw == :median
        u     = mapslices(x -> quantile(x, 0.5), result.e_draws, dims=3)[:, :, 1]
        alpha = mapslices(x -> quantile(x, 0.5), result.Phi_draws, dims=3)[:, :, 1]
        Sigma = mapslices(x -> quantile(x, 0.5), result.Sigma_draws, dims=3)[:, :, 1]
    else  # :mean
        u     = mean(result.e_draws, dims=3)[:, :, 1]
        alpha = mean(result.Phi_draws, dims=3)[:, :, 1]
        Sigma = mean(result.Sigma_draws, dims=3)[:, :, 1]
    end

    if Omega === nothing
        Omega = Matrix{Float64}(I, K, K)
    end

    Tu = size(u, 1)
    A  = cholesky(Hermitian(Sigma)).L

    # Structural innovations: e = A * Omega * eta  →  eta = (A*Omega)^{-1} * e
    ierror = u / (Omega' * A')

    # Companion form
    F = companion_form(alpha, K, p)
    Kp = K * p
    G = zeros(Kp, K); G[1:K, :] = I(K)

    # Deterministic part (constant)
    has_const = size(alpha, 1) > K * p
    C_vec = zeros(Kp)
    if has_const
        C_vec[1:K] = alpha[K*p+1, :]
    end

    # Exogenous variables (MATLAB: histdecomp.m)
    nexogenous = 0
    exogenous_ = false
    if size(alpha, 1) > K * p + (has_const ? 1 : 0)
        exogenous_ = true
        exo_start = K * p + (has_const ? 2 : 1)
        nexogenous = size(alpha, 1) - exo_start + 1
        # Companion form for exogenous coefficients
        Gamma = zeros(Kp, nexogenous)
        Gamma[1:K, :] = alpha[exo_start:end, :]'
        # Extract exogenous data from X matrix
        exogenous = result.var.X[1:Tu, (K*p + (has_const ? 2 : 1)):end]
    end

    # Initial conditions from data
    yo = result.var.X[1, 1:K*p]

    # Deterministic + initial conditions
    B_ = zeros(Tu, K)
    for t in 1:Tu
        Aa = zeros(Kp)
        for tau in 1:t
            Aa += F^(tau-1) * C_vec
        end
        B = Aa + F^t * yo
        B_[t, :] = B[1:K]
    end

    # Structural part
    Kappa = G * A * Omega
    E_ = zeros(Tu, K, K)
    for shock in 1:K
        Ind = zeros(K, K); Ind[shock, shock] = 1.0
        for t in 1:Tu
            D_ = zeros(Kp)
            for tau in 1:t
                D_ += F^(t-tau) * Kappa * Ind * ierror[tau, :]
            end
            E_[t, :, shock] = D_[1:K]
        end
    end

    # Exogenous part (if any) — matches MATLAB histdecomp.m
    Q_ = zeros(Tu, K, max(nexogenous, 0))
    if exogenous_
        for exo in 1:nexogenous
            for t in 1:Tu
                M_ = zeros(Kp)
                for tau in 1:t
                    M_ += F^(t-tau) * Gamma[:, exo] * exogenous[tau, exo]
                end
                Q_[t, :, exo] = M_[1:K]
            end
        end
    end

    # Check: data ≈ deterministic + sum of shocks + exogenous
    yhat = B_ + dropdims(sum(E_, dims=3), dims=3)
    if exogenous_
        yhat += dropdims(sum(Q_, dims=3), dims=3)
    end
    data = result.var.data[p+1:end, :]
    data_check = data[1:Tu, :]
    maxdisc = maximum(abs.(data_check .- yhat))
    if maxdisc > tol
        @warn "Historical decomposition discrepancy: $maxdisc"
    end

    # Combine: shocks + exogenous (if any) + deterministic
    # Matches MATLAB ordering: E_, Q_, B_
    decomp = E_
    if exogenous_
        decomp = cat(decomp, Q_, dims=3)
    end
    decomp = cat(decomp, reshape(B_, Tu, K, 1), dims=3)

    return HistDecompResult(decomp, ierror)
end
