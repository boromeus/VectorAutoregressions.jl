# Example 7: Local Projections
# Translation of MATLAB example_7_LP.m
# Reference: Gertler and Karadi (2015)
#
# 1) Classical LP with Cholesky identification
# 2) Classical LP with IV (proxy)
# 3) Bayesian LP
# 4) Bayesian LP with Optimal Shrinkage

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
T_obs = size(y, 1)

# Load instrument
fd_raw, fd_headers = readdlm(joinpath(data_dir, "factor_data.csv"), ','; header=true)
fd_cols = vec(String.(fd_headers))
ff4_idx = findfirst(x -> occursin("ff4", x), fd_cols)
instrument_raw = Float64.(fd_raw[:, ff4_idx])

# Build full-length proxy (NaN-padded to match y length)
proxy_full = fill(NaN, T_obs, 1)
proxy_full[T_obs - length(instrument_raw) + 1:end, 1] = instrument_raw

K = size(y, 2)
indx_var = [3, 2, 1, 4]
indx_sho = 3  # gs1 shock
varnames = ["1 year rate", "CPI", "IP", "EBP"]

# ─── 1) Classical LP with Cholesky ────────────────────────────────────────────
println("=== 1) Classical LP — Cholesky ===")
lags = 12
hor = 48
lp1 = lp_irf(y, lags, hor; identification=:cholesky, conf_level=0.90)

println("LP IRF at h=12, MP shock (gs1=3):")
for (i, vname) in enumerate(varnames)
    v = indx_var[i]
    col_idx = (indx_sho - 1) * K + v
    med = round(lp1.irf[13, col_idx], digits=6)
    println("  $vname: $med")
end

# ─── 2) Classical LP with IV (proxy) ─────────────────────────────────────────
println("\n=== 2) Classical LP — IV ===")
lp2 = lp_irf(y, lags, hor; identification=:proxy, proxy=proxy_full,
             conf_level=0.90)

println("LP-IV IRF at h=12, MP shock:")
for (i, vname) in enumerate(varnames)
    v = indx_var[i]
    if size(lp2.irf, 2) >= v
        med = round(lp2.irf[13, v], digits=6)
        println("  $vname: $med")
    end
end

# ─── 3) Bayesian LP ──────────────────────────────────────────────────────────
println("\n=== 3) Bayesian LP ===")
K_draws = parse(Int, get(ENV, "BVAR_K", "1000"))
blp = lp_bayesian(y, lags, hor; prior=MinnesotaPrior(), K=K_draws,
                  conf_level=0.90)

println("BLP IRF at h=12, Cholesky MP shock:")
for (i, vname) in enumerate(varnames)
    v = indx_var[i]
    med = round(blp.irf[v, 13, indx_sho], digits=6)
    println("  $vname: $med")
end

# ─── 4) Bayesian LP with Optimal Shrinkage ─────────────────────────────────
println("\n=== 4) Bayesian LP with Optimal Shrinkage ===")
best_ml = -Inf
best_tau = 1.0
for tau_try in [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
    try
        ml = lp_marginal_likelihood(y, lags, hor,
            MinnesotaPrior(tau=tau_try))
        if ml > best_ml
            global best_ml = ml
            global best_tau = tau_try
        end
        println("  tau=$(rpad(tau_try, 6)) → logML=$(round(ml, digits=2))")
    catch e
        println("  tau=$(rpad(tau_try, 6)) → failed: $e")
    end
end
println("Optimal tau: $(best_tau), logML: $(round(best_ml, digits=2))")

blp_opt = lp_bayesian(y, lags, hor; prior=MinnesotaPrior(tau=best_tau),
                      K=K_draws, conf_level=0.90)
println("\nOptimal BLP IRF at h=12, Cholesky MP shock:")
for (i, vname) in enumerate(varnames)
    v = indx_var[i]
    med = round(blp_opt.irf[v, 13, indx_sho], digits=6)
    println("  $vname: $med")
end

println("\nExample 7 complete.")
