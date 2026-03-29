#=
types.jl — Core type hierarchy for VectorAutoregressions.jl
=#

# ─── Prior Specifications ───────────────────────────────────────────────────────

abstract type AbstractPrior end

"""
    FlatPrior()

Jeffrey / uninformative prior.  Posterior ∝ |Σ|^{-(n+1)/2}.
"""
struct FlatPrior <: AbstractPrior end

"""
    MinnesotaPrior(; tau, decay, lambda, mu, omega)

Minnesota prior implemented via dummy observations.

# Fields
- `tau::Float64`:    overall tightness (default 3.0)
- `decay::Float64`:  lag‑decay shrinkage (default 0.5)
- `lambda::Float64`: sum‑of‑coefficients weight (default 5.0)
- `mu::Float64`:     co‑persistence weight (default 2.0)
- `omega::Float64`:  shock‑variance weight (default 2.0)
"""
Base.@kwdef struct MinnesotaPrior <: AbstractPrior
    tau::Float64 = 3.0
    decay::Float64 = 0.5
    lambda::Float64 = 5.0
    mu::Float64 = 2.0
    omega::Float64 = 2.0
end

"""
    ConjugatePrior(; Phi_mean, Phi_cov, Sigma_scale, Sigma_df)

Multivariate‑Normal / Inverse‑Wishart conjugate prior.
"""
Base.@kwdef struct ConjugatePrior <: AbstractPrior
    Phi_mean::Matrix{Float64}
    Phi_cov::Matrix{Float64}
    Sigma_scale::Matrix{Float64}
    Sigma_df::Int
end

# ─── Identification Schemes ─────────────────────────────────────────────────────

abstract type AbstractIdentification end

struct CholeskyIdentification <: AbstractIdentification end

Base.@kwdef struct SignRestriction <: AbstractIdentification
    restrictions::Vector{String}
    max_rotations::Int = 30000
end

Base.@kwdef struct NarrativeSignRestriction <: AbstractIdentification
    signs::Vector{String}
    narrative::Vector{String}
    max_rotations::Int = 30000
end

Base.@kwdef struct ZeroSignRestriction <: AbstractIdentification
    restrictions::Vector{String}
    var_pos::Vector{Int} = Int[]
end

struct ProxyIdentification <: AbstractIdentification
    instrument::Matrix{Float64}
    proxy_end::Int
end
ProxyIdentification(Z::Matrix{Float64}) = ProxyIdentification(Z, 0)
ProxyIdentification(z::Vector{Float64}) = ProxyIdentification(reshape(z, :, 1), 0)

struct LongRunIdentification <: AbstractIdentification end

struct HeteroskedIdentification <: AbstractIdentification
    regimes::Vector{Int}
end

# ─── VAR Estimation Results ─────────────────────────────────────────────────────

"""
    VAREstimate

Stores OLS‑estimated reduced‑form VAR results.
"""
struct VAREstimate
    data::Matrix{Float64}          # original data T₀ × K
    Y::Matrix{Float64}             # T × K dependent variable (after pre‑sample)
    X::Matrix{Float64}             # T × (Kp+nx) regressors
    Phi::Matrix{Float64}           # (Kp+nx) × K  coefficient matrix  [lags | const | trend | exog]
    Sigma::Matrix{Float64}         # K × K  residual covariance
    residuals::Matrix{Float64}     # T × K  OLS residuals
    XXi::Matrix{Float64}           # (X'X)^{-1}
    nobs::Int                      # effective sample size T
    nvar::Int                      # K
    nlags::Int                     # p
    has_constant::Bool
    has_trend::Bool
    nexogenous::Int
end

# ─── Info Criteria ──────────────────────────────────────────────────────────────
struct InfoCriteria
    aic::Float64
    bic::Float64
    hqic::Float64
end

# ─── BVAR Results ───────────────────────────────────────────────────────────────

"""
    BVARResult

Container for posterior draws from a Bayesian VAR.
"""
struct BVARResult
    var::VAREstimate
    prior::AbstractPrior
    identification::AbstractIdentification
    Phi_draws::Array{Float64, 3}         # (Kp+nx) × K × ndraws
    Sigma_draws::Array{Float64, 3}       # K × K × ndraws
    ir_draws::Array{Float64, 4}          # K × hor × K × ndraws  (variable, horizon, shock, draw)
    irlr_draws::Array{Float64, 4}        # long‑run IRFs
    irsign_draws::Array{Float64, 4}      # sign‑restriction IRFs
    irnarrsign_draws::Array{Float64, 4}  # narrative+sign IRFs
    irzerosign_draws::Array{Float64, 4}  # zero+sign IRFs
    irproxy_draws::Array{Float64, 4}     # proxy IRFs
    e_draws::Array{Float64, 3}           # T × K × ndraws  residuals
    Omega_draws::Array{Float64, 3}       # K × K × ndraws  rotation matrices
    forecasts_no_shocks::Array{Float64, 3}   # fhor × K × ndraws
    forecasts_with_shocks::Array{Float64, 3} # fhor × K × ndraws
    forecasts_conditional::Array{Float64, 3} # fhor × K × ndraws
    marginal_likelihood::Float64
    info_criteria::InfoCriteria
    ndraws::Int
    nlags::Int
    nvar::Int
    hor::Int
    fhor::Int
end

# ─── IRF Results ────────────────────────────────────────────────────────────────

struct IRFResult
    irf::Array{Float64, 3}       # K × (hor+1) × K   median / point
    lower::Array{Float64, 3}     # K × (hor+1) × K
    upper::Array{Float64, 3}     # K × (hor+1) × K
    horizon::Int
    conf_level::Float64
end

# ─── FEVD Results ───────────────────────────────────────────────────────────────

struct FEVDResult
    decomposition::Matrix{Float64}    # K × K  (variable × shock)  at a given horizon
    horizon::Int
end

struct FEVDPosteriorResult
    median::Array{Float64, 3}    # K × K × hor
    lower::Array{Float64, 3}
    upper::Array{Float64, 3}
    conf_level::Float64
end

# ─── Forecast Results ───────────────────────────────────────────────────────────

struct ForecastResult
    point_no_shocks::Matrix{Float64}    # fhor × K
    point_with_shocks::Matrix{Float64}  # fhor × K
    conditional::Union{Nothing, Matrix{Float64}}
end

# ─── Historical Decomposition ───────────────────────────────────────────────────

struct HistDecompResult
    decomposition::Array{Float64, 3}  # T × K × (K+nextra)  last = deterministic
    structural_shocks::Matrix{Float64}  # T × K
end

# ─── Connectedness ──────────────────────────────────────────────────────────────

struct ConnectednessResult
    index::Float64                  # overall Diebold‑Yilmaz index
    from_all_to_unit::Vector{Float64}
    from_unit_to_all::Vector{Float64}
    net::Vector{Float64}
    theta::Matrix{Float64}
end

# ─── Local Projection Results ───────────────────────────────────────────────────

struct LPResult
    irf::Matrix{Float64}           # (hor+1) × K²
    lower::Matrix{Float64}
    upper::Matrix{Float64}
    std::Matrix{Float64}
    horizon::Int
    conf_level::Float64
end

# ─── Panel VAR ──────────────────────────────────────────────────────────────────

struct PanelVARResult
    Phi::Matrix{Float64}
    Sigma::Matrix{Float64}
    residuals::Matrix{Float64}
    unit_results::Vector{VAREstimate}
    method::Symbol  # :pooled, :unit, :exchangeable
end

# ─── FAVAR ──────────────────────────────────────────────────────────────────────

struct FAVARResult
    factors::Matrix{Float64}
    loadings::Matrix{Float64}
    var::VAREstimate
    Phi_draws::Array{Float64, 3}
    Sigma_draws::Array{Float64, 3}
    ir_draws::Array{Float64, 4}
    nfactors::Int
    ndraws::Int
end
