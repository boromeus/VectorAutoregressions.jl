# Example 4: Mixed-Frequency BVAR
# Translation of MATLAB example_4_mfvar.m
# Reference: Euro Area quarterly GDP + monthly indicators
#
# 1) Estimate a Monthly-Quarterly mixed-frequency VAR
# 2) Trailing nowcast example

using VectorAutoregressions
using DelimitedFiles, Statistics, Random, LinearAlgebra

# ─── Load Data ─────────────────────────────────────────────────────────────────
data_dir = joinpath(@__DIR__, "data")
raw, headers = readdlm(joinpath(data_dir, "DataMF.csv"), ','; header=true)
colnames = vec(String.(headers))
col(name) = findfirst(==(name), colnames)

# Variables: GDP (quarterly), IPI, HICP, CORE, Euribor1Y, UNRATE (monthly)
# Handle missing values (empty strings → NaN)
function safe_float(x)
    if x isa Number
        return Float64(x)
    end
    s = strip(string(x))
    return isempty(s) || s == "NaN" ? NaN : parse(Float64, s)
end

GDP       = safe_float.(raw[:, col("GDP")])
IPI       = safe_float.(raw[:, col("IPI")])
HICP      = safe_float.(raw[:, col("HICP")])
CORE      = safe_float.(raw[:, col("CORE")])
Euribor1Y = safe_float.(raw[:, col("Euribor1Y")])
UNRATE    = safe_float.(raw[:, col("UNRATE")])

y = hcat(GDP, IPI, HICP, CORE, Euribor1Y, UNRATE)
lags = 6
K_draws = parse(Int, get(ENV, "BVAR_K", "1000"))
println("Data loaded: $(size(y, 1)) obs × $(size(y, 2)) vars")
println("GDP has $(sum(isnan.(GDP))) NaN entries (quarterly → monthly NaNs)")

# ─── 1) Mixed-Frequency VAR Estimation ─────────────────────────────────────────
println("\n=== Mixed-Frequency VAR ===")

# Step 1: Linear interpolation for initial BVAR estimation
y_init = copy(y)
for j in 1:size(y, 2)
    yj = y_init[:, j]
    nan_idx = findall(isnan, yj)
    obs_idx = findall(!isnan, yj)
    if !isempty(nan_idx) && length(obs_idx) >= 2
        for ni in nan_idx
            below = findlast(x -> x < ni, obs_idx)
            above = findfirst(x -> x > ni, obs_idx)
            if below !== nothing && above !== nothing
                t0, t1 = obs_idx[below], obs_idx[above]
                y_init[ni, j] = yj[t0] + (yj[t1] - yj[t0]) * (ni - t0) / (t1 - t0)
            elseif below !== nothing
                y_init[ni, j] = yj[obs_idx[below]]
            elseif above !== nothing
                y_init[ni, j] = yj[obs_idx[above]]
            end
        end
    end
end

# Step 2: Estimate BVAR on interpolated data
bvarmf = bvar(y_init, lags; prior=MinnesotaPrior(), K=K_draws, hor=24, verbose=false)
println("BVAR estimated: $(bvarmf.ndraws) draws")

# Step 3: Use Kalman filter to get smoothed states
Phi_mean = dropdims(mean(bvarmf.Phi_draws, dims=3), dims=3)
Sigma_mean = dropdims(mean(bvarmf.Sigma_draws, dims=3), dims=3)

kf_result = kalman_filter(Phi_mean, Sigma_mean, y;
                           initial_cond=:diffuse)

println("Kalman smoother completed")
ny = size(y, 2)
gdp_smoothed = kf_result.smoothed[:, 1]
println("Smoothed monthly GDP sample: first 5 values = $(round.(gdp_smoothed[1:5], digits=4))")

# Compare smoothed vs observed GDP
obs_gdp_idx = findall(!isnan, GDP)
if !isempty(obs_gdp_idx)
    gdp_err = GDP[obs_gdp_idx] .- gdp_smoothed[obs_gdp_idx]
    println("Mean abs error on observed quarters: $(round(mean(abs.(gdp_err)), digits=6))")
end

# ─── 2) Nowcasting ─────────────────────────────────────────────────────────────
println("\n=== Nowcasting ===")
last_complete = size(y, 1) - 3
forecast_initval = y_init[last_complete-lags+1:last_complete, :]
forecast_xdata = ones(3, 1)
frcst_no, frcst_with = forecast_unconditional(
    forecast_initval, forecast_xdata,
    Phi_mean, Sigma_mean, 3, lags)

println("Nowcast results (GDP, next 3 months):")
println("  No-shock forecast: $(round.(frcst_no[:, 1], digits=4))")
if !isnan(GDP[end])
    println("  Actual GDP(end):   $(round(GDP[end], digits=4))")
end

println("\nExample 4 complete.")
