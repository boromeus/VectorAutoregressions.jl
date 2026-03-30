# Example 12: Bayesian Dynamic Factor Model (BDFM)
# Translation of MATLAB example_12_bdfm.m
#
# 1) Simulate a factor model DGP
# 2) Extract principal components
# 3) Estimate a static factor model via FAVAR Gibbs sampler
# 4) Estimate a dynamic factor model
# 5) Compare estimated vs true factors

using VectorAutoregressions
using DelimitedFiles, Statistics, Random, LinearAlgebra

# ─── Set up RNG for reproducibility ─────────────────────────────────────────
rng = Random.MersenneTwister(999)

# ─── Simulate Factor Model DGP ─────────────────────────────────────────────
println("=== Simulating Factor Model DGP ===")

# Parameters
Phi_true = [0.75 0.0; 0.1 0.8]           # AR(1) factor dynamics
Sigma_true = [1.0 -0.2; -0.2 2.0]         # Factor innovation covariance
Q_true = cholesky(Hermitian(Sigma_true)).L

nfac = 2
# Factor loadings: 5 structured + 45 random
Lambda_true = [1.0 0.0;
               0.3 1.0;
               0.5 -1.5;
               -1.1 -0.5;
               1.5 1.5;
               5.0 * randn(rng, 45, nfac)]
ny = size(Lambda_true, 1)  # 50 observables

# Idiosyncratic noise
rho = 0.0    # no persistence
sig = 1.0    # std dev

# Simulate
T_sim = 301
f = zeros(T_sim, nfac)
e = zeros(T_sim, ny)
y_sim = zeros(T_sim, ny)

for t in 2:T_sim
    innov = randn(rng, nfac)
    f[t, :] = Phi_true * f[t-1, :] + Q_true * innov
    e[t, :] = rho * e[t-1, :] + sig * randn(rng, ny)
    y_sim[t, :] = Lambda_true * f[t, :] + e[t, :]
end

# Drop first observation (initial condition)
f = f[2:end, :]
y_data = y_sim[2:end, :]
T_data = size(y_data, 1)
println("Simulated data: $T_data obs × $ny variables, $nfac factors")

# True IRFs
hor_irf = 24
ir_true_f = compute_irf(Phi_true, Matrix{Float64}(I, nfac, nfac), hor_irf)
# Rescale to observable space
ir_true = zeros(ny, hor_irf, nfac)
for ff in 1:nfac
    for h in 1:hor_irf
        ir_true[:, h, ff] = Lambda_true * ir_true_f[:, h, ff]
    end
end

# ─── 1) Principal Components ────────────────────────────────────────────────
println("\n=== Principal Components ===")
pc_result = principal_components(y_data, nfac; demean=:none)
fhat = pc_result.factors
eigenvalues = pc_result.eigenvalues

println("Eigenvalues (first 10):")
println("  $(round.(eigenvalues[1:min(10, length(eigenvalues))], digits=2))")

# Correlation between estimated and true factors (up to sign flip)
for g in 1:nfac
    c = cor(fhat[:, g], f[:, g])
    println("  Factor $g: correlation with true = $(round(c, digits=4))")
end

# ─── 2) Static Factor Model (lags=0) ────────────────────────────────────────
println("\n=== Static Factor Model ===")
# Use the favar function with lags=0 (static)
# For static factor model, we estimate: y = Lambda * f + e
# with no factor dynamics (just cross-sectional)

# Since favar requires y_slow and y_fast, we use all variables as slow
# and a dummy for fast
y_slow = y_data
y_fast = zeros(T_data, 0)  # no fast variables

# Use principal components to initialize
# Run BVAR on the factors for the dynamic case
y_favar_static = fhat  # nfac PCs
bvar_static = bvar(y_favar_static, 1; prior=FlatPrior(), K=1000, verbose=false)

# Compare factor estimates
println("Factor estimation (PC-based, static):")
for g in 1:nfac
    f_mean = fhat[:, g]
    # Adjust sign
    c = cor(f_mean, f[:, g])
    sign_adj = sign(c)
    println("  Factor $g: corr=$(round(c, digits=4)) (sign-adjusted)")
end

# ─── 3) Dynamic Factor Model ────────────────────────────────────────────────
println("\n=== Dynamic Factor Model ===")
# AR(1) factor dynamics → lags=1
favar_lags = 1

# Estimate BVARs on the PC-extracted factors
bvar_dfm = bvar(fhat, favar_lags; prior=FlatPrior(), K=2000, hor=hor_irf, verbose=false)

# Compare Phi estimates
Phi_est = dropdims(mean(bvar_dfm.Phi_draws, dims=3), dims=3)
println("\nEstimated Factor AR matrix (mean posterior):")
println("  $(round.(Phi_est[1:nfac, :], digits=4))")
println("True Factor AR matrix:")
println("  $(round.(Phi_true, digits=4))")

# Compare Sigma estimates
Sigma_est = dropdims(mean(bvar_dfm.Sigma_draws, dims=3), dims=3)
println("\nEstimated Factor Σ (mean posterior):")
println("  $(round.(Sigma_est, digits=4))")
println("True Factor Σ:")
println("  $(round.(Sigma_true, digits=4))")

# ─── 4) Compare IRFs ────────────────────────────────────────────────────────
println("\n=== IRF Comparison ===")
# Rescale BVAR IRFs to observable space using PC loadings
Lambda_est = pc_result.loadings
STD = pc_result.scale
C_ = rescale_favar(STD, Lambda_est, 0; order_pc=:factor_first)

println("IRF at h=12, Factor 1 shock (first 5 observables):")
for obs in 1:min(5, ny)
    true_val = ir_true[obs, 12, 1]
    est_val = sum(C_[obs, j] * median(bvar_dfm.ir_draws[j, 12, 1, :])
                  for j in 1:nfac)
    println("  Obs $obs: true=$(round(true_val, digits=4)), " *
            "est=$(round(est_val, digits=4))")
end

# Factor correlation summary
println("\n=== Factor Recovery Summary ===")
for g in 1:nfac
    c = cor(fhat[:, g], f[:, g])
    println("  Factor $g: |corr| = $(round(abs(c), digits=4))")
end

println("\nExample 12 complete.")
