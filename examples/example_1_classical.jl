# Example 1: Classical VAR — Estimation, IRFs, FEVD, Bootstrap
# Translation of MATLAB example_1_classical.m
# Reference: Gertler and Karadi (2015) data set
#
# 1) Optimal lag selection
# 2) Flat-prior BVAR with Cholesky IRFs
# 3) Forecast Error Variance Decomposition
# 4) Classical VAR with bootstrap confidence intervals

using VectorAutoregressions
using DelimitedFiles, Statistics, Random

# ─── Load Data ─────────────────────────────────────────────────────────────────
data_dir = joinpath(@__DIR__, "data")
raw = readdlm(joinpath(data_dir, "DataGK.csv"), ','; header=true)[1]
headers = readdlm(joinpath(data_dir, "DataGK.csv"), ','; header=true)[2]

# Map column names to indices
colnames = vec(String.(headers))
col(name) = findfirst(==(name), colnames)

logip  = Float64.(raw[:, col("logip")])
logcpi = Float64.(raw[:, col("logcpi")])
gs1    = Float64.(raw[:, col("gs1")])
ebp    = Float64.(raw[:, col("ebp")])

y = hcat(logip, logcpi, gs1, ebp)
println("Data loaded: $(size(y, 1)) observations, $(size(y, 2)) variables")

# ─── 1) Optimal lag length ─────────────────────────────────────────────────────
println("\n=== Lag Selection ===")
lag_range = parse(Int, get(ENV, "BVAR_LAG_MIN", "5")):parse(Int, get(ENV, "BVAR_LAG_MAX", "24"))
for nlags in lag_range
    bvar_ic = bvar(y, nlags; prior=FlatPrior(), K=1, hor=1, fhor=1, verbose=false)
    ic = bvar_ic.info_criteria
    println("Lags=$nlags  AIC=$(round(ic.aic, digits=2))  " *
            "BIC=$(round(ic.bic, digits=2))  " *
            "HQIC=$(round(ic.hqic, digits=2))")
end

# ─── 2) Flat-prior BVAR with Cholesky decomposition ───────────────────────────
lags = 12
hor  = 48
K_draws = parse(Int, get(ENV, "BVAR_K", "5000"))
println("\n=== Flat-prior BVAR (lags=$lags, hor=$hor, K=$K_draws) ===")
bvar1 = bvar(y, lags; prior=FlatPrior(), K=K_draws, hor=hor, verbose=true)

# Define the IRF of interest
# Variable order for plotting: gs1(3), logcpi(2), logip(1), ebp(4)
indx_var = [3, 2, 1, 4]
indx_sho = indx_var
varnames = ["1 year rate", "CPI", "IP", "EBP"]
shocknames = ["eR", "eCPI", "eIP", "eEBP"]

# Print median IRFs at horizon 12 for each shock
println("\nMedian IRF at horizon 12:")
for (si, sname) in enumerate(shocknames)
    println("  Shock: $sname")
    for (vi, vname) in enumerate(varnames)
        med = median(bvar1.ir_draws[indx_var[vi], 12, indx_sho[si], :])
        println("    $vname: $(round(med, digits=6))")
    end
end

# ─── 3) Forecast Error Variance Decomposition ─────────────────────────────────
println("\n=== FEVD ===")

# Using the mean of the posterior distribution
Phi_mean   = dropdims(mean(bvar1.Phi_draws, dims=3), dims=3)
Sigma_mean = dropdims(mean(bvar1.Sigma_draws, dims=3), dims=3)
Phi_med    = dropdims(median(bvar1.Phi_draws, dims=3), dims=3)
Sigma_med  = dropdims(median(bvar1.Sigma_draws, dims=3), dims=3)

# Index of the shock of interest (gs1 = variable 3)
indx_sho_fevd = 3

println("% Forecast Error Variance Decomposition")
println("% Percentage of 1/2/4 years ahead volatility")
println("% Explained by a Monetary Policy shock")
println("    logip     logcpi    gs1        ebp")

ny = size(y, 2)
for (label, Phi_use, Sigma_use) in [("Mean", Phi_mean, Sigma_mean),
                                      ("Median", Phi_med, Sigma_med)]
    println(" $label")
    for hh in [12, 24, 48]
        FEVD = compute_fevd(Phi_use[1:ny*lags, :], Sigma_use, hh)
        row = FEVD.decomposition[:, indx_sho_fevd]
        println("  h=$hh: $(round.(row', digits=2))")
    end
end

# Computing the distribution of the FEVD
println("\n=== FEVD Posterior Distribution (12-month horizon) ===")
fevd_post = fevd_posterior(bvar1; conf_level=0.95, horizons=1:12)
println("    logip     logcpi    gs1        ebp")
println("Upper HPD:")
println("  $(round.(fevd_post.upper[:, indx_sho_fevd, end]', digits=2))")
println("Median:")
println("  $(round.(fevd_post.median[:, indx_sho_fevd, end]', digits=2))")
println("Lower HPD:")
println("  $(round.(fevd_post.lower[:, indx_sho_fevd, end]', digits=2))")

# ─── 4) Classical bootstrap VAR ───────────────────────────────────────────────
println("\n=== Classical VAR with Bootstrap ===")
ir_point, ir_boot, var_ols = classical_var(y, lags;
    nboot=K_draws, hor=hor, constant=true, rng=Random.MersenneTwister(42))

# Compare BVAR vs bootstrap at horizon 12 for MP shock
println("\nIRF at h=12, MP shock (gs1=3):")
println("  Method       logip     logcpi    gs1        ebp")
bvar_med = [median(bvar1.ir_draws[v, 12, 3, :]) for v in 1:4]
boot_med = [median(ir_boot[v, 12, 3, :]) for v in 1:4]
println("  BVAR median: $(round.(bvar_med', digits=6))")
println("  Boot median: $(round.(boot_med', digits=6))")
println("  OLS point:   $(round.(ir_point[:, 12, 3]', digits=6))")

println("\nExample 1 complete.")
