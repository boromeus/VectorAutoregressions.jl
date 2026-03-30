# Example 8: Panel VARs
# Translation of MATLAB example_8_panels.m
# Reference: Bank lending/deposit rate panel data
#
# 1) Unit-by-unit estimation and response averaging
# 2) Pooled estimation
# 3) Partial pooling (Bayesian shrinkage)
# 4) Exchangeable prior (use subset to form prior for remaining)

using VectorAutoregressions
using DelimitedFiles, Statistics, Random, LinearAlgebra

# ─── Load Data ─────────────────────────────────────────────────────────────────
data_dir = joinpath(@__DIR__, "data")
lr_raw = readdlm(joinpath(data_dir, "DataBanks_LendingRate.csv"), ',')
dr_raw = readdlm(joinpath(data_dir, "DataBanks_DepositRate.csv"), ',')

LendingRate = Float64.(lr_raw)
DepositRate = Float64.(dr_raw)
T_obs, NBanks = size(LendingRate)
println("Panel data: $T_obs months × $NBanks banks")

lags = 4
hor = 24
varnames = ["Lending Rate", "Deposit Rate"]

# ─── 1) Unit-by-unit estimation ─────────────────────────────────────────────
println("\n=== 1) Unit-by-Unit Estimation ===")
irfs_unit = zeros(2, hor, 2, NBanks)
for i in 1:NBanks
    yi = hcat(LendingRate[:, i], DepositRate[:, i])
    bvar_i = bvar(yi, lags; prior=FlatPrior(), K=1, hor=hor, verbose=false)
    # Use OLS point IRF (K=1 draw)
    irfs_unit[:, :, :, i] = bvar_i.ir_draws[:, :, :, 1]
end

# Average and median across banks
irf_mean = dropdims(mean(irfs_unit, dims=4), dims=4)
irf_med = dropdims(median(irfs_unit, dims=4), dims=4)

println("Mean IRF at h=12 across $NBanks banks:")
for (vi, vname) in enumerate(varnames)
    for (si, sname) in enumerate(varnames)
        println("  $vname ← $sname shock: $(round(irf_mean[vi, 12, si], digits=4))")
    end
end

# Cross-sectional dispersion
println("\nCross-sectional std of IRF at h=12:")
for (vi, vname) in enumerate(varnames)
    for (si, sname) in enumerate(varnames)
        sd = round(std(irfs_unit[vi, 12, si, :]), digits=4)
        println("  $vname ← $sname shock: $sd")
    end
end

# ─── 2) Pooled Estimation ───────────────────────────────────────────────────
println("\n=== 2) Pooled Estimation ===")
# Demean each bank and stack into a 3D array
panels = Vector{Matrix{Float64}}()
for i in 1:NBanks
    lr_dm = LendingRate[:, i] .- mean(LendingRate[:, i])
    dr_dm = DepositRate[:, i] .- mean(DepositRate[:, i])
    push!(panels, hcat(lr_dm, dr_dm))
end

pool_result = panel_var(panels, 2; method=:pooled, constant=true)
println("Pooled VAR: Phi size=$(size(pool_result.Phi)), Sigma=$(round.(pool_result.Sigma, digits=4))")

# Compute pooled IRF
ir_pooled = compute_irf(pool_result.Phi[1:2*2, :], pool_result.Sigma, hor)
println("\nPooled IRF at h=12:")
for (vi, vname) in enumerate(varnames)
    for (si, sname) in enumerate(varnames)
        println("  $vname ← $sname shock: $(round(ir_pooled[vi, 12, si], digits=4))")
    end
end

# ─── 3) Partial Pooling (Bayesian Shrinkage) ────────────────────────────────
println("\n=== 3) Partial Pooling ===")
Nv = 2
lags_pp = 2
k_pp = lags_pp * Nv + 1  # intercept
barBet = zeros(k_pp, Nv)
barBet[1, 1] = 1.0        # random walk for lending rate
barBet[lags_pp + 1, 2] = 1.0  # random walk for deposit rate
gam = 0.1                 # shrinkage parameter

conj_pp = ConjugatePrior(
    Phi_mean=barBet,
    Phi_cov=gam * Matrix{Float64}(I, k_pp, k_pp),
    Sigma_scale=Matrix{Float64}(I, Nv, Nv),
    Sigma_df=Nv + 2
)

# Estimate for last bank as illustration
i_last = NBanks
yi_last = hcat(LendingRate[:, i_last] .- mean(LendingRate[:, i_last]),
               DepositRate[:, i_last] .- mean(DepositRate[:, i_last]))
bvar_pp = bvar(yi_last, lags_pp; prior=conj_pp, K=1000, hor=hor, verbose=false)

println("Partial pooling, Bank #$i_last:")
println("  Median IRF at h=12:")
for (vi, vname) in enumerate(varnames)
    for (si, sname) in enumerate(varnames)
        med = round(median(bvar_pp.ir_draws[vi, 12, si, :]), digits=4)
        println("    $vname ← $sname: $med")
    end
end

# ─── 4) Exchangeable Prior ──────────────────────────────────────────────────
println("\n=== 4) Exchangeable Prior ===")
# Use first 20 banks to form the prior for the remaining 30
N1 = 20
# Stack first N1 banks
lr_stacked = vcat([LendingRate[:, i] .- mean(LendingRate[:, i]) for i in 1:N1]...)
dr_stacked = vcat([DepositRate[:, i] .- mean(DepositRate[:, i]) for i in 1:N1]...)
yp = hcat(lr_stacked, dr_stacked)

bvar_exch_pre = bvar(yp, lags_pp; prior=FlatPrior(), K=1000, verbose=false)
Phi_exch_mean = dropdims(mean(bvar_exch_pre.Phi_draws, dims=3), dims=3)

gam_exch = 0.1
conj_exch = ConjugatePrior(
    Phi_mean=Phi_exch_mean,
    Phi_cov=gam_exch * Matrix{Float64}(I, size(Phi_exch_mean, 1), size(Phi_exch_mean, 1)),
    Sigma_scale=Matrix{Float64}(I, Nv, Nv),
    Sigma_df=Nv + 2
)

# Estimate for last bank with exchangeable prior
yi_exch = hcat(LendingRate[:, NBanks], DepositRate[:, NBanks])
bvar_exch = bvar(yi_exch, lags_pp; prior=conj_exch, K=1000, hor=hor, verbose=false)

# Collect OLS IRFs for banks 21-50
irfs_exch = zeros(2, hor, 2, NBanks - N1)
for i in N1+1:NBanks
    yi = hcat(LendingRate[:, i], DepositRate[:, i])
    bv = bvar(yi, lags_pp; prior=FlatPrior(), K=1, hor=hor, verbose=false)
    irfs_exch[:, :, :, i - N1] = bv.ir_draws[:, :, :, 1]
end

println("Exchangeable prior, Bank #$NBanks:")
println("  Median IRF at h=12:")
for (vi, vname) in enumerate(varnames)
    for (si, sname) in enumerate(varnames)
        med = round(median(bvar_exch.ir_draws[vi, 12, si, :]), digits=4)
        println("    $vname ← $sname: $med")
    end
end

println("\nCross-sectional IRF (banks 21-50), mean at h=12:")
for (vi, vname) in enumerate(varnames)
    for (si, sname) in enumerate(varnames)
        m = round(mean(irfs_exch[vi, 12, si, :]), digits=4)
        println("  $vname ← $sname: $m")
    end
end

println("\nExample 8 complete.")
