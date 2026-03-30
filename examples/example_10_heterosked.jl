# Example 10: Heteroskedasticity Weights for COVID-Era Forecasting
# Translation of MATLAB example_10_VAR_heterosked.m
#
# 1) Baseline Minnesota forecasting
# 2) Forecasting with heteroskedastic weights (down-weight COVID)
# 3) Optimal heteroskedastic weights

using VectorAutoregressions
using DelimitedFiles, Statistics, Random, LinearAlgebra

# ─── Load Data ─────────────────────────────────────────────────────────────────
data_dir = joinpath(@__DIR__, "data")
raw, headers = readdlm(joinpath(data_dir, "DataCovid.csv"), ','; header=true)
colnames = vec(String.(headers))

time_covid = Float64.(raw[:, findfirst(==("time"), colnames)])
y = Float64.(raw[:, 2:end])  # PAYEMS UNRATE PCE INDPRO CPIAUCSL PCEPILFE
varnames = ["PAYEMS", "UNRATE", "PCE", "INDPRO", "CPIAUCSL", "PCEPILFE"]
T_obs = size(y, 1)
println("COVID data: $T_obs obs × $(size(y,2)) vars")
println("Time range: $(time_covid[1]) to $(time_covid[end])")

lags = 13
fhor = 24

# ─── 1) Baseline Minnesota Forecast ─────────────────────────────────────────
println("\n=== 1) Baseline Minnesota Forecast ===")
bvar0 = bvar(y, lags; prior=MinnesotaPrior(), K=1000, fhor=fhor, verbose=false)
frcst0 = dropdims(mean(bvar0.forecasts_with_shocks, dims=3), dims=3)
println("24-month ahead forecast (baseline Minnesota):")
for (j, vn) in enumerate(varnames)
    println("  $vn: $(round(frcst0[fhor, j], digits=2))")
end

# ─── 2) Forecasting with Heteroskedastic Weights ────────────────────────────
println("\n=== 2) Heteroskedastic Weights (COVID down-weighting) ===")
# Find March 2020 in time vector
tstar_idx = findfirst(x -> x >= 2020 + 2/12, time_covid)
if tstar_idx === nothing
    tstar_idx = T_obs - 5
end
println("COVID start index (March 2020): $tstar_idx")

# Create weight vector: scale up variance for COVID months (March, April, May 2020)
# In MATLAB: st(tstar:tstar+2) = 10 (larger weight = more variance = less influence)
# After removing first `lags` observations:
st = ones(T_obs - lags)
covid_start = tstar_idx - lags
if covid_start > 0 && covid_start + 2 <= length(st)
    st[covid_start:covid_start+2] .= 10.0
    println("Down-weighting obs $(covid_start):$(covid_start+2) by factor 10")
end

bvar1 = bvar(y, lags; prior=MinnesotaPrior(), heterosked_weights=st,
             K=1000, fhor=fhor, verbose=false)
frcst1 = dropdims(mean(bvar1.forecasts_with_shocks, dims=3), dims=3)

println("24-month ahead forecast (heteroskedastic weights):")
for (j, vn) in enumerate(varnames)
    println("  $vn: $(round(frcst1[fhor, j], digits=2))")
end

# ─── 3) Optimal Weights ─────────────────────────────────────────────────────
println("\n=== 3) Optimal Weights via Hyperparameter Optimization ===")
# In MATLAB, the optimization searches over:
# hyperpara = [tau, s0, s1, s2] where s0,s1,s2 are the COVID weights
# Here we optimize tau and report the weights

# Step 1: Optimize tau with fixed COVID weights
best_prior_hetero, best_ml_hetero = optimize_hyperparameters_optim(y, lags;
    hyperpara=[3.0, 0.5, 1.0, 1.0, 2.0],
    index_est=[1],
    lb=[0.1],
    ub=[10.0],
    method=:nelder_mead)

println("Optimal tau: $(round(best_prior_hetero.tau, digits=4))")
println("Log marginal likelihood: $(round(best_ml_hetero, digits=4))")

# Run BVAR with optimal tau + heteroskedastic weights
bvar2 = bvar(y, lags; prior=best_prior_hetero, heterosked_weights=st,
             K=1000, fhor=fhor, verbose=false)
frcst2 = dropdims(mean(bvar2.forecasts_with_shocks, dims=3), dims=3)

println("\n24-month ahead forecast (optimal weights):")
for (j, vn) in enumerate(varnames)
    println("  $vn: $(round(frcst2[fhor, j], digits=2))")
end

# ─── Compare Forecasts ──────────────────────────────────────────────────────
println("\n=== Forecast Comparison at h=24 ===")
println("  Variable        Baseline    Heterosk    Optimal")
for (j, vn) in enumerate(varnames)
    b = round(frcst0[fhor, j], digits=2)
    h = round(frcst1[fhor, j], digits=2)
    o = round(frcst2[fhor, j], digits=2)
    println("  $(rpad(vn, 16)) $(lpad(string(b),8))  $(lpad(string(h),8))  $(lpad(string(o),8))")
end

println("\nExample 10 complete.")
