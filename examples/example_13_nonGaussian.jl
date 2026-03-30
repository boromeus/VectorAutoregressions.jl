# Example 13: Non-Gaussian BVAR — Robust Bayesian Inference
# Translation of MATLAB example_13_nonGaussian_bvar.m
# Reference: Andrade, Ferroni, Melosi (2023)
#
# 1) Standard Gaussian BVAR
# 2) Robust BVAR with fat tails (kurtosis)
# 3) Robust BVAR with skewness and kurtosis
# 4) Compare posterior distributions

using VectorAutoregressions
using DelimitedFiles, Statistics, Random, LinearAlgebra

# ─── Load Data ─────────────────────────────────────────────────────────────────
data_dir = joinpath(@__DIR__, "data")
raw, headers = readdlm(joinpath(data_dir, "DataGK.csv"), ','; header=true)
colnames = vec(String.(headers))
col(name) = findfirst(==(name), colnames)

logip  = Float64.(raw[:, col("logip")])
logcpi = Float64.(raw[:, col("logcpi")])
gs1    = Float64.(raw[:, col("gs1")])
ebp    = Float64.(raw[:, col("ebp")])

y = hcat(logip, logcpi, gs1, ebp)
lags = 12
K_draws = parse(Int, get(ENV, "BVAR_K", "5000"))
varnames = ["IP", "CPI", "1 year rate", "EBP"]
println("Data loaded: $(size(y, 1)) obs × $(size(y, 2)) vars")

# ─── 1) Standard Gaussian BVAR ──────────────────────────────────────────────
println("\n=== 1) Standard Gaussian BVAR ===")
bvar0 = bvar(y, lags; prior=FlatPrior(), K=K_draws, hor=24, verbose=false)

# Compute OLS residual statistics
var_ols = var_estimate(y, lags)
resid = var_ols.residuals
T_eff = size(resid, 1)

println("OLS Residual Statistics:")
for (j, vn) in enumerate(varnames)
    r = resid[:, j]
    sk = mean(((r .- mean(r)) ./ std(r)).^3)
    ku = mean(((r .- mean(r)) ./ std(r)).^4)
    println("  $vn: skewness=$(round(sk, digits=3)), kurtosis=$(round(ku, digits=3))")
end
println("  (Normal: skewness=0, kurtosis=3)")

# ─── 2) Robust BVAR — Kurtosis Only ─────────────────────────────────────────
println("\n=== 2) Robust BVAR (kurtosis) ===")
bvar1 = bvar(y, lags; prior=FlatPrior(), K=K_draws, hor=24,
             robust_bayes=1, verbose=false)

# ─── 3) Robust BVAR — Skewness + Kurtosis ───────────────────────────────────
println("\n=== 3) Robust BVAR (skewness + kurtosis) ===")
bvar2 = bvar(y, lags; prior=FlatPrior(), K=K_draws, hor=24,
             robust_bayes=2, verbose=false)

# ─── 4) Compare Cholesky IRFs ───────────────────────────────────────────────
println("\n=== Cholesky IRF Comparison (EBP shock) ===")
indx_sho = 4  # shock to EBP
println("Median IRF at h=12:")
println("  Variable         Gaussian   Kurtosis   Skew+Kurt")
for (j, vn) in enumerate(varnames)
    m0 = round(median(bvar0.ir_draws[j, 12, indx_sho, :]), digits=6)
    m1 = round(median(bvar1.ir_draws[j, 12, indx_sho, :]), digits=6)
    m2 = round(median(bvar2.ir_draws[j, 12, indx_sho, :]), digits=6)
    println("  $(rpad(vn, 18)) $m0   $m1   $m2")
end

println("\n68% credible bands at h=12:")
println("  Variable         Gaussian                  Kurtosis")
for (j, vn) in enumerate(varnames)
    lo0 = round(quantile(bvar0.ir_draws[j, 12, indx_sho, :], 0.16), digits=6)
    hi0 = round(quantile(bvar0.ir_draws[j, 12, indx_sho, :], 0.84), digits=6)
    lo1 = round(quantile(bvar1.ir_draws[j, 12, indx_sho, :], 0.16), digits=6)
    hi1 = round(quantile(bvar1.ir_draws[j, 12, indx_sho, :], 0.84), digits=6)
    println("  $(rpad(vn, 18)) [$lo0, $hi0]  [$lo1, $hi1]")
end

# ─── 5) Compare Posterior of Sigma ───────────────────────────────────────────
println("\n=== Posterior of Sigma (Cholesky lower triangle) ===")
# Extract vech of lower Cholesky factor
ny = size(y, 2)
n_vech = ny * (ny + 1) ÷ 2

sig0 = zeros(n_vech, bvar0.ndraws)
sig1 = zeros(n_vech, bvar1.ndraws)
for k in 1:bvar0.ndraws
    L0 = cholesky(Hermitian(bvar0.Sigma_draws[:, :, k])).L
    sig0[:, k] = vech(L0)
end
for k in 1:bvar1.ndraws
    L1 = cholesky(Hermitian(bvar1.Sigma_draws[:, :, k])).L
    sig1[:, k] = vech(L1)
end

println("Mean of vech(chol(Sigma)):")
println("  Gaussian: $(round.(mean(sig0, dims=2)[:, 1], digits=6))")
println("  Kurtosis: $(round.(mean(sig1, dims=2)[:, 1], digits=6))")

# Check if distributions differ significantly
println("\nKS-like difference (max |median difference|):")
for i in 1:n_vech
    diff_med = abs(median(sig0[i, :]) - median(sig1[i, :]))
    if diff_med > 0.001
        println("  Element $i: $(round(diff_med, digits=6))")
    end
end

println("\nNote: For the same identification (Cholesky), IRF differences")
println("are small. Only the posterior of Sigma is affected by the")
println("distributional assumption on errors.")

println("\nExample 13 complete.")
