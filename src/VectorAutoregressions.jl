"""
    VectorAutoregressions.jl

A comprehensive Julia package for Vector Autoregressive (VAR) models,
Bayesian VARs, Factor-Augmented VARs, Local Projections, and time-series analysis.

Ported from the MATLAB BVAR_ toolbox by Filippo Ferroni and Fabio Canova.
Original Julia implementation by Luca Brugnolini.
"""
module VectorAutoregressions

using LinearAlgebra
using Statistics
using Random
using Distributions: TDist, quantile
using SpecialFunctions: loggamma

# ─── Source files ────────────────────────────────────────────────────────────────

include("types.jl")
include("utils.jl")
include("estimation.jl")
include("priors.jl")
include("irf.jl")
include("identification.jl")
include("bayesian.jl")
include("marginal_likelihood.jl")
include("fevd.jl")
include("historical_decomp.jl")
include("forecasting.jl")
include("connectedness.jl")
include("local_projections.jl")
include("kalman.jl")
include("panel.jl")
include("favar_new.jl")
include("filters.jl")
include("plotting.jl")

# ─── Exports ─────────────────────────────────────────────────────────────────────

# Types
export AbstractPrior, FlatPrior, MinnesotaPrior, ConjugatePrior
export AbstractIdentification, CholeskyIdentification
export SignRestriction, NarrativeSignRestriction, ZeroSignRestriction
export ProxyIdentification, LongRunIdentification, HeteroskedIdentification
export VAREstimate, BVARResult, IRFResult
export FEVDResult, FEVDPosteriorResult
export ForecastResult, HistDecompResult, ConnectednessResult
export LPResult, PanelVARResult, FAVARResult, InfoCriteria

# Estimation
export var_estimate, var_lagorder, information_criteria, rfvar3

# Bayesian
export bvar, classical_var

# Priors
export get_prior_moments, build_dummy_observations, compute_prior_posterior

# IRF
export compute_irf, compute_irf_longrun, compute_irf_proxy, compute_irf_heterosked
export wild_bootstrap_irf_proxy

# Identification
export irf_sign_restriction, irf_narrative_sign, irf_zero_sign
export parse_sign_restriction, parse_zero_sign_restrictions
export check_sign_restrictions, check_narrative_restrictions
export max_horizon_sign, findQs, findP

# FEVD & Historical Decomposition
export compute_fevd, fevd_posterior
export historical_decomposition

# Forecasting
export forecast_unconditional, forecast_conditional

# Connectedness
export compute_connectedness, connectedness_posterior

# Marginal Likelihood
export compute_marginal_likelihood, optimize_hyperparameters

# Local Projections
export lp_irf, lp_lagorder

# Kalman Filter
export kalman_filter

# Panel VAR
export panel_var

# FAVAR
export principal_components, rescale_favar, favar

# Filters
export hp_filter, bk_filter, cf_filter, hamilton_filter

# Utilities
export lagmatrix, companion_form, check_stability
export rand_inverse_wishart, vech, ivech
export commutation_matrix, duplication_matrix, elimination_matrix
export var2ma, var2ss, ols_svd, generate_rotation_matrix, matrictint

end # module
