# Example 2: Minnesota Prior — Hyperparameter Optimization
# Translation of MATLAB example_2_minn.m
# Reference: Euro Area macro data
#
# 1) Default Minnesota prior
# 2) Partial hyperparameter optimization (cherry-picked + estimated)
# 3) Sequential optimization of Minnesota hyperparameters
# 4) Conjugate prior with presample

using VectorAutoregressions
using DelimitedFiles, Statistics, Random, LinearAlgebra

# ─── Load Data ─────────────────────────────────────────────────────────────────
data_dir = joinpath(@__DIR__, "data")
raw, headers = readdlm(joinpath(data_dir, "Data.csv"), ','; header=true)
colnames = vec(String.(headers))
col(name) = findfirst(==(name), colnames)

IPI       = Float64.(raw[:, col("IPI")])
HICP      = Float64.(raw[:, col("HICP")])
CORE      = Float64.(raw[:, col("CORE")])
Euribor1Y = Float64.(raw[:, col("Euribor1Y")])
M3        = Float64.(raw[:, col("M3")])
EXRATE    = Float64.(raw[:, col("EXRATE")])

y = hcat(IPI, HICP, CORE, Euribor1Y, M3, EXRATE)
println("Data loaded: $(size(y, 1)) obs × $(size(y, 2)) vars")

# ─── Case 1: Default Minnesota Prior ─────────────────────────────────────────
println("\n=== Case 1: Default Minnesota Prior + Optimization ===")
lags = 6
K_draws = parse(Int, get(ENV, "BVAR_K", "5000"))
# Optimize all 5 hyperparameters
best_prior0, best_ml0 = optimize_hyperparameters_optim(y, lags;
    hyperpara=[3.0, 0.5, 5.0, 2.0, 2.0],
    index_est=collect(1:5),
    lb=[0.1, 0.1, 0.1, 0.1, 0.1],
    ub=[10.0, 10.0, 10.0, 10.0, 10.0],
    method=:nelder_mead)
println("Optimal hyperparameters (all 5):")
println("  tau=$(round(best_prior0.tau, digits=4)), " *
        "decay=$(round(best_prior0.decay, digits=4)), " *
        "lambda=$(round(best_prior0.lambda, digits=4)), " *
        "mu=$(round(best_prior0.mu, digits=4)), " *
        "omega=$(round(best_prior0.omega, digits=4))")
println("  Log marginal likelihood: $(round(best_ml0, digits=4))")

BVAR0 = bvar(y, lags; prior=best_prior0, K=K_draws, hor=24, verbose=false)

# ─── Case 2: Mixed Cherry-Picked/Estimated ───────────────────────────────────
println("\n=== Case 2: Partial Optimization (lambda, mu only) ===")
# Fix tau=10, optimize lambda and mu only
best_prior1, best_ml1 = optimize_hyperparameters_optim(y, lags;
    hyperpara=[10.0, 0.5, 5.0, 2.0, 2.0],
    index_est=[3, 4],
    lb=[0.0, 0.0],
    ub=[20.0, 20.0],
    method=:nelder_mead)
println("Optimal (lambda, mu): " *
        "lambda=$(round(best_prior1.lambda, digits=4)), " *
        "mu=$(round(best_prior1.mu, digits=4))")
println("  Log marginal likelihood: $(round(best_ml1, digits=4))")

BVAR1 = bvar(y, lags; prior=best_prior1, K=K_draws, hor=24, verbose=false)

# ─── Case 3: Sequential Optimization ─────────────────────────────────────────
println("\n=== Case 3: Sequential Optimization ===")
# Step 3.1: optimize tau
hp = [3.0, 0.5, 5.0, 2.0, 2.0]
best_p31, ml31 = optimize_hyperparameters_optim(y, lags;
    hyperpara=hp, index_est=[1], lb=[0.8], ub=[10.0], method=:nelder_mead)
println("Step 3.1: tau=$(round(best_p31.tau, digits=4))")

# Step 3.2: optimize tau, decay, lambda
hp[1] = best_p31.tau
best_p32, ml32 = optimize_hyperparameters_optim(y, lags;
    hyperpara=[best_p31.tau, 0.5, 5.0, 2.0, 2.0],
    index_est=[1, 2, 3], lb=[0.1, 0.1, 0.1], ub=[10.0, 10.0, 10.0],
    method=:nelder_mead)
println("Step 3.2: tau=$(round(best_p32.tau, digits=4)), " *
        "decay=$(round(best_p32.decay, digits=4)), " *
        "lambda=$(round(best_p32.lambda, digits=4))")

# Step 3.3: optimize tau, decay, lambda, mu
best_p33, ml33 = optimize_hyperparameters_optim(y, lags;
    hyperpara=[best_p32.tau, best_p32.decay, best_p32.lambda, 2.0, 2.0],
    index_est=[1, 2, 3, 4], lb=[0.1, 0.1, 0.1, 0.1], ub=[10.0, 10.0, 10.0, 10.0],
    method=:nelder_mead)
println("Step 3.3: tau=$(round(best_p33.tau, digits=4)), " *
        "decay=$(round(best_p33.decay, digits=4)), " *
        "lambda=$(round(best_p33.lambda, digits=4)), " *
        "mu=$(round(best_p33.mu, digits=4))")

BVAR2 = bvar(y, lags; prior=best_p33, K=K_draws, hor=24, verbose=false)

# ─── Compare IRFs ────────────────────────────────────────────────────────────
# Monetary policy shock (Euribor1Y = variable 4)
indx_sho = 4
indx_var = [4, 1, 2, 3]  # rate, IP, HICP, CORE
varnames = ["1 year rate", "IP", "HICP", "CORE INF"]

println("\n=== IRF Comparison at h=12, MP shock ===")
println("    Variable    BVAR0(opt-all)  BVAR1(partial)  BVAR2(sequential)")
for (i, vname) in enumerate(varnames)
    m0 = round(median(BVAR0.ir_draws[indx_var[i], 12, indx_sho, :]), digits=6)
    m1 = round(median(BVAR1.ir_draws[indx_var[i], 12, indx_sho, :]), digits=6)
    m2 = round(median(BVAR2.ir_draws[indx_var[i], 12, indx_sho, :]), digits=6)
    println("    $(rpad(vname, 16)) $m0    $m1    $m2")
end

# ─── Case 4: Conjugate Prior with Presample ──────────────────────────────────
println("\n=== Case 4: Conjugate Prior from Presample ===")
presample = 50  # ~8 years monthly
bvar_pre = bvar(y[1:presample, :], lags; prior=FlatPrior(), K=1000, verbose=false)

# Use presample posterior to form conjugate prior
Phi_pre_mean = dropdims(mean(bvar_pre.Phi_draws, dims=3), dims=3)
Phi_pre_var  = dropdims(mean(var(bvar_pre.Phi_draws, dims=3), dims=2), dims=2)
Sigma_pre_mean = dropdims(mean(bvar_pre.Sigma_draws, dims=3), dims=3)
nk_pre = size(bvar_pre.Phi_draws, 1)

conj_prior = ConjugatePrior(
    Phi_mean=Phi_pre_mean,
    Phi_cov=Diagonal(vec(Phi_pre_var)) |> Matrix,
    Sigma_scale=Sigma_pre_mean,
    Sigma_df=nk_pre - 2
)

bvar_conj = bvar(y[presample+1:end, :], lags;
    prior=conj_prior, K=1000, hor=24, verbose=false)
println("Conjugate prior BVAR: $(bvar_conj.ndraws) draws, " *
        "$(bvar_conj.nvar) vars, $(bvar_conj.nlags) lags")

# IRF comparison
println("\nMedian IRF at h=12, Euribor shock (conjugate prior):")
for (i, vname) in enumerate(varnames)
    med = round(median(bvar_conj.ir_draws[indx_var[i], 12, indx_sho, :]), digits=6)
    println("  $vname: $med")
end

println("\nExample 2 complete.")
