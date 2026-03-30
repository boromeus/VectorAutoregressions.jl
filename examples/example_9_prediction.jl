# Example 9: Out-of-Sample Prediction
# Translation of MATLAB example_9_prediction.m
# Reference: Euro Area macro data
#
# 1) Unconditional forecasts with flat prior
# 2) Unconditional forecasts with default Minnesota prior
# 3) Unconditional forecasts with optimal Minnesota prior
# 4) Conditional forecasts on Euribor path
# 5) Conditional forecasts using only MP shocks (Cholesky)

using VectorAutoregressions
using DelimitedFiles, Statistics, Random, LinearAlgebra

# ─── Load Data ─────────────────────────────────────────────────────────────────
data_dir = joinpath(@__DIR__, "data")
raw, headers = readdlm(joinpath(data_dir, "Data.csv"), ','; header=true)
colnames = vec(String.(headers))
col(name) = findfirst(==(name), colnames)

T_time     = Float64.(raw[:, col("T")])
IPI        = Float64.(raw[:, col("IPI")])
HICP       = Float64.(raw[:, col("HICP")])
CORE       = Float64.(raw[:, col("CORE")])
Euribor1Y  = Float64.(raw[:, col("Euribor1Y")])
M3         = Float64.(raw[:, col("M3")])
EXRATE     = Float64.(raw[:, col("EXRATE")])

yactual = hcat(IPI, HICP, CORE, Euribor1Y, M3, EXRATE)

# Estimation sample: up to a cutoff (simulating real-time)
# Find the index closest to the last in-sample period
in_sample_end = size(yactual, 1) - 12  # leave 12 months for evaluation
y = yactual[1:in_sample_end, :]
println("In-sample: $(in_sample_end) obs, Out-of-sample: $(size(yactual,1) - in_sample_end) obs")

lags = 6
fhor = 12
K_draws = parse(Int, get(ENV, "BVAR_K", "5000"))
varnames = ["IPI", "HICP", "CORE", "Euribor1Y", "M3", "EXRATE"]

# ─── 1) Unconditional Forecasts — Flat Prior ────────────────────────────────
println("\n=== 1) Flat Prior Forecast ===")
bvar1 = bvar(y, lags; prior=FlatPrior(), K=K_draws, fhor=fhor, verbose=false)
frcst1_mean = dropdims(mean(bvar1.forecasts_no_shocks, dims=3), dims=3)
println("12-month ahead forecast (flat prior):")
for (j, vn) in enumerate(varnames)
    println("  $vn: $(round(frcst1_mean[fhor, j], digits=4))")
end

# ─── 2) Unconditional Forecasts — Minnesota Prior ───────────────────────────
println("\n=== 2) Minnesota Prior Forecast ===")
bvar2 = bvar(y, lags; prior=MinnesotaPrior(), K=K_draws, fhor=fhor, verbose=false)
frcst2_mean = dropdims(mean(bvar2.forecasts_no_shocks, dims=3), dims=3)
println("12-month ahead forecast (Minnesota):")
for (j, vn) in enumerate(varnames)
    println("  $vn: $(round(frcst2_mean[fhor, j], digits=4))")
end

# ─── 3) Unconditional Forecasts — Optimal Minnesota ─────────────────────────
println("\n=== 3) Optimal Minnesota Forecast ===")
best_prior, best_ml = optimize_hyperparameters_optim(y, lags;
    hyperpara=[3.0, 0.5, 5.0, 2.0, 2.0],
    index_est=collect(1:4),
    lb=[0.0, 0.0, 0.0, 0.0],
    ub=[10.0, 10.0, 10.0, 10.0],
    method=:nelder_mead)
println("Optimal hyperparameters: tau=$(round(best_prior.tau, digits=3)), " *
        "decay=$(round(best_prior.decay, digits=3)), " *
        "lambda=$(round(best_prior.lambda, digits=3)), " *
        "mu=$(round(best_prior.mu, digits=3))")

bvar3 = bvar(y, lags; prior=best_prior, K=K_draws, fhor=fhor, verbose=false)
frcst3_mean = dropdims(mean(bvar3.forecasts_no_shocks, dims=3), dims=3)
println("12-month ahead forecast (optimal Minnesota):")
for (j, vn) in enumerate(varnames)
    println("  $vn: $(round(frcst3_mean[fhor, j], digits=4))")
end

# ─── 4) Conditional Forecast on Euribor Path ────────────────────────────────
println("\n=== 4) Conditional Forecast on Euribor ===")
# Condition on the actual future path of Euribor1Y
endo_index = [4]  # Euribor1Y
endo_path = reshape(yactual[in_sample_end+1:in_sample_end+fhor, 4], :, 1)

# Use mean posterior parameters for illustration
Phi_mean = dropdims(mean(bvar3.Phi_draws, dims=3), dims=3)
Sigma_mean = dropdims(mean(bvar3.Sigma_draws, dims=3), dims=3)

forecast_initval = y[end-lags+1:end, :]
forecast_xdata = ones(fhor, 1)

cond_frcst, _ = forecast_conditional(
    endo_path, endo_index,
    forecast_initval, forecast_xdata,
    Phi_mean, Sigma_mean, fhor, lags)

println("Conditional forecast (Euribor path imposed):")
println("  h    Euribor(cond)  Euribor(actual)  IPI(cond)")
for h in [1, 3, 6, 12]
    println("  $h    $(round(cond_frcst[h, 4], digits=4))  " *
            "$(round(yactual[in_sample_end+h, 4], digits=4))  " *
            "$(round(cond_frcst[h, 1], digits=4))")
end

# Verify condition is met
cond_err = maximum(abs.(cond_frcst[:, 4] .- endo_path[:, 1]))
println("  Max conditioning error: $(round(cond_err, digits=8))")

# ─── 5) Conditional Forecast Using Only MP Shocks ───────────────────────────
println("\n=== 5) Conditional Forecast (MP shocks only) ===")
exo_index = [4]  # only shock 4 (Cholesky: Euribor) used for conditioning
cond_frcst2, shocks = forecast_conditional_exo(
    endo_path, endo_index, exo_index,
    forecast_initval, forecast_xdata,
    Phi_mean, Sigma_mean, fhor, lags)

println("Conditional forecast (only MP shock moves):")
println("  h    Euribor(cond)  Euribor(actual)  IPI(cond)")
for h in [1, 3, 6, 12]
    println("  $h    $(round(cond_frcst2[h, 4], digits=4))  " *
            "$(round(yactual[in_sample_end+h, 4], digits=4))  " *
            "$(round(cond_frcst2[h, 1], digits=4))")
end

# ─── Compare forecasts ──────────────────────────────────────────────────────
println("\n=== Forecast Comparison at h=12 ===")
actual_12 = yactual[in_sample_end+12, :]
println("  Variable    Flat     Minn     OptMinn  CondFrcst  Actual")
for (j, vn) in enumerate(varnames)
    f1 = round(frcst1_mean[12, j], digits=3)
    f2 = round(frcst2_mean[12, j], digits=3)
    f3 = round(frcst3_mean[12, j], digits=3)
    fc = round(cond_frcst[12, j], digits=3)
    act = round(actual_12[j], digits=3)
    println("  $(rpad(vn, 12)) $f1    $f2    $f3    $fc    $act")
end

println("\nExample 9 complete.")
