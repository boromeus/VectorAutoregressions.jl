# Example 5: Factor-Augmented VAR (FAVAR)
# Translation of MATLAB example_5_favar.m
# Reference: Bernanke, Boivin, Eliasz (2005)
#
# 1) Extract principal components from slow-moving variables
# 2) Estimate BVAR on factors + policy variable (Cholesky)
# 3) Rescale IRFs to original variable space
# 4) Sign restrictions on uncompressed variables

using VectorAutoregressions
using DelimitedFiles, Statistics, Random, LinearAlgebra

# ─── Load Data ─────────────────────────────────────────────────────────────────
data_dir = joinpath(@__DIR__, "data")

# y1 = policy variable (TBILL3M), y2 = slow-moving variables
y1_raw = readdlm(joinpath(data_dir, "DataFAVAR_y1.csv"), ','; header=true)[1]
y2_raw = readdlm(joinpath(data_dir, "DataFAVAR_y2.csv"), ','; header=true)[1]
vn_raw = readdlm(joinpath(data_dir, "DataFAVAR_varnames.csv"), ','; header=true)[1]

y1 = Float64.(y1_raw)
y2 = Float64.(y2_raw)
varnames_y2 = vec(String.(vn_raw))

println("FAVAR data: y1=$(size(y1)), y2=$(size(y2))")
println("  Slow-moving variables: $(length(varnames_y2))")

# ─── 1) Principal Components ─────────────────────────────────────────────────
nfac = 3  # number of factors
pc_result = principal_components(y2, nfac; demean=:standardize)
fhat = pc_result.factors
Lambda = pc_result.loadings
STD = pc_result.scale

println("\n=== Principal Components ===")
println("Factors: $(size(fhat))")
println("Eigenvalues (first 5): $(round.(pc_result.eigenvalues[1:5], digits=2))")

# ─── 2) FAVAR: BVAR on [factors, y1] with Cholesky ──────────────────────────
# Factors first, then policy variable
y_favar = hcat(fhat, y1)
lags = 2
K_draws = parse(Int, get(ENV, "BVAR_K", "5000"))
println("\n=== FAVAR Estimation (lags=$lags) ===")
fabvar = bvar(y_favar, lags; prior=FlatPrior(), K=K_draws, hor=24, verbose=false)

# ─── 3) Rescale IRFs to original variable space ─────────────────────────────
# PC are ordered first (factor_first)
ny1 = size(y1, 2)
C_ = rescale_favar(STD, Lambda, ny1; order_pc=:factor_first)

# MP shock index (after nfac factors)
indx_sho = nfac + 1

# Construct rescaled IRF for each draw
n_orig = size(C_, 1)
hor = fabvar.hor
irX_draws = zeros(n_orig, hor, 1, fabvar.ndraws)
for k in 1:fabvar.ndraws
    irX_draws[:, :, 1, k] = C_ * fabvar.ir_draws[:, :, indx_sho, k]
end

# Identify variables of interest
gdp_idx = findfirst(==("GDPC96"), varnames_y2)
core_idx = findfirst(==("JCXFE"), varnames_y2)

if gdp_idx !== nothing && core_idx !== nothing
    println("\n=== Rescaled IRF to MP shock ===")
    println("Median IRF at h=12:")
    println("  GDP:      $(round(median(irX_draws[gdp_idx, 12, 1, :]), digits=6))")
    println("  CORE PCE: $(round(median(irX_draws[core_idx, 12, 1, :]), digits=6))")
    println("  TBILL3M:  $(round(median(fabvar.ir_draws[nfac+1, 12, indx_sho, :]), digits=6))")
else
    println("Warning: Could not find GDPC96 or JCXFE in variable names")
    println("  Available: $(varnames_y2[1:min(5, length(varnames_y2))])")
end

# ─── 4) Sign Restrictions on Uncompressed Variables ──────────────────────────
println("\n=== Sign Restrictions on Uncompressed Variables ===")
# In the MATLAB BVAR_ toolbox, sign restrictions on the uncompressed space
# are applied by checking C * irf against the sign constraints.
# The Julia implementation currently applies sign restrictions on the factor space;
# a full implementation of the uncompressed sign restriction requires extending
# irf_sign_restriction to accept a rescaling matrix (C_).
# Here we demonstrate the approach manually.
gdpdefl_idx = findfirst(==("GDPCTPI"), varnames_y2)

if gdp_idx !== nothing && gdpdefl_idx !== nothing
    nfac_plus_y1 = size(y_favar, 2)
    accepted = 0
    irXsign_draws = zeros(n_orig, hor, 1, fabvar.ndraws)

    for k in 1:fabvar.ndraws
        Phi_k = fabvar.Phi_draws[:, :, k]
        Sigma_k = fabvar.Sigma_draws[:, :, k]

        for rot_try in 1:5000
            Omega = VectorAutoregressions.generate_rotation_matrix(nfac_plus_y1)
            ir_fac = compute_irf(Phi_k[1:nfac_plus_y1*lags, :], Sigma_k, hor; Omega=Omega)
            # Rescale to original space
            irX = C_ * ir_fac[:, :, 1]

            # Check sign restrictions: GDP(+), GDP deflator(−) in h=1:3
            if all(irX[gdp_idx, 1:3] .> 0) && all(irX[gdpdefl_idx, 1:3] .< 0)
                irXsign_draws[:, :, 1, k] = irX
                global accepted += 1
                break
            end
        end
    end

    println("Accepted sign-restricted draws: $(accepted) / $(fabvar.ndraws)")

    if accepted > 0
        nz = [k for k in 1:fabvar.ndraws if !all(irXsign_draws[:, :, 1, k] .== 0)]
        println("Median IRF to AS shock at h=12:")
        println("  GDP:       $(round(median(irXsign_draws[gdp_idx, 12, 1, nz]), digits=6))")
        println("  GDP Defl:  $(round(median(irXsign_draws[gdpdefl_idx, 12, 1, nz]), digits=6))")
        jcxfe_idx = findfirst(==("JCXFE"), varnames_y2)
        if jcxfe_idx !== nothing
            println("  CORE PCE:  $(round(median(irXsign_draws[jcxfe_idx, 12, 1, nz]), digits=6))")
        end
    end
else
    println("Skipping sign restrictions: variable indices not found")
end

println("\nExample 5 complete.")
