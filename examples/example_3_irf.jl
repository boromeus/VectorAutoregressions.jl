# Example 3: Impulse Responses — Multiple Identification Schemes
# Translation of MATLAB example_3_irf.m
# Reference: Gertler and Karadi (2015)
#
# 1) Cholesky identification
# 2) Sign restrictions
# 3) Narrative + sign restrictions
# 4) Zero + sign restrictions
# 5) Long-run restrictions (technology shock)
# 6) Proxy / IV identification
# Extra: FEVD and Historical Decomposition

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
K_draws = parse(Int, get(ENV, "BVAR_K", "1000"))
K_slow  = max(100, K_draws ÷ 5)  # fewer draws for expensive identification

# Variable ordering: 1=logip, 2=logcpi, 3=gs1, 4=ebp
# Plot order: gs1(3), logcpi(2), logip(1), ebp(4)
indx_var = [3, 2, 1, 4]
varnames = ["1 year rate", "CPI", "IP", "EBP"]

# ─── 1) Cholesky ──────────────────────────────────────────────────────────────
println("=== 1) Cholesky Identification ===")
bvar1 = bvar(y, lags; prior=FlatPrior(), K=K_draws, hor=24, verbose=false)

indx_sho = 3  # shock to gs1
println("Median IRF at h=12, MP shock (Cholesky):")
for (i, vname) in enumerate(varnames)
    med = round(median(bvar1.ir_draws[indx_var[i], 12, indx_sho, :]), digits=6)
    println("  $vname: $med")
end

# ─── 2) Sign Restrictions ────────────────────────────────────────────────────
println("\n=== 2) Sign Restrictions ===")
# gs1 up in periods 1–3, CPI down in periods 1–3
sign_id = SignRestriction(
    restrictions=["y(3,1:3,1)>0", "y(2,1:3,1)<0"],
    max_rotations=30000
)
bvar2 = bvar(y, lags; prior=FlatPrior(), identification=sign_id,
             K=K_draws, hor=24, verbose=false)

indx_sho_sign = 1  # sign-restricted shocks are re-ordered
println("Median IRF at h=12, MP tightening (sign restr.):")
for (i, vname) in enumerate(varnames)
    med = round(median(bvar2.irsign_draws[indx_var[i], 12, indx_sho_sign, :]), digits=6)
    println("  $vname: $med")
end

# Verify sign restrictions hold
gs1_h1to3 = bvar2.irsign_draws[3, 1:3, 1, :]  # variable 3, horizons 1-3, shock 1
cpi_h1to3 = bvar2.irsign_draws[2, 1:3, 1, :]  # variable 2, horizons 1-3, shock 1
gs1_pos = all(minimum(gs1_h1to3, dims=1) .>= 0)
cpi_neg = all(maximum(cpi_h1to3, dims=1) .<= 0)
println("  Sign restrictions satisfied: gs1>0=$(gs1_pos), cpi<0=$(cpi_neg)")

# ─── 3) Narrative + Sign Restrictions ────────────────────────────────────────
println("\n=== 3) Narrative + Sign Restrictions ===")
# Sign: gs1 up h1-3, CPI down h1-3
# Narrative: Volcker tightening episodes
# Sample starts 1979m7, first innovation is 1980m7 (after 12 lags)
# 1980m9-11 → indices 3:5, 1981m5 → index 11
narrsign_id = NarrativeSignRestriction(
    signs=["y(3,1:3,1)>0", "y(2,1:3,1)<0"],
    narrative=["v(3:5,1)>0", "v(11,1)>0"],
    max_rotations=30000
)
bvar3 = bvar(y, lags; prior=FlatPrior(), identification=narrsign_id,
             K=K_slow, hor=24, verbose=false)

println("Median IRF at h=12, MP tightening (narrative+sign):")
for (i, vname) in enumerate(varnames)
    med = round(median(bvar3.irnarrsign_draws[indx_var[i], 12, 1, :]), digits=6)
    println("  $vname: $med")
end

# ─── 4) Zero + Sign Restrictions ─────────────────────────────────────────────
println("\n=== 4) Zero + Sign Restrictions ===")
# AD shock: IP(+), CPI(+), gs1(+)
# AS shock: IP(+), CPI(−)
# MP shock: no contemporaneous IP/CPI response [zeros], gs1(+), ebp(+)
zerosign_id = ZeroSignRestriction(
    restrictions=[
        "y(1,1)=1", "y(2,1)=1", "y(3,1)=1",       # AD: IP+, CPI+, gs1+
        "y(1,2)=1", "y(2,2)=-1",                     # AS: IP+, CPI−
        "ys(1,3)=0", "ys(2,3)=0",                     # MP: zero on IP, CPI
        "y(3,3)=1", "y(4,3)=1"                        # MP: gs1+, ebp+
    ]
)
bvar4 = bvar(y, lags; prior=FlatPrior(), identification=zerosign_id,
             K=K_slow, hor=24, verbose=false)

println("Median IRF at h=12, 3 shocks (zero+sign):")
for shock in 1:3
    snames = ["AD", "AS", "MP"]
    println("  $(snames[shock]) shock:")
    for (i, vname) in enumerate(varnames)
        med = round(median(bvar4.irzerosign_draws[indx_var[i], 12, shock, :]), digits=6)
        println("    $vname: $med")
    end
end

# ─── 5) Long-Run Restrictions ────────────────────────────────────────────────
println("\n=== 5) Long-Run Restrictions (Technology Shock) ===")
# Use differenced IP for long-run identification (Blanchard–Quah)
y_lr = hcat(diff(logip), logcpi[2:end], gs1[2:end], ebp[2:end])
lr_id = LongRunIdentification()
bvar5 = bvar(y_lr, lags; prior=FlatPrior(), identification=lr_id,
             K=K_draws, hor=24, verbose=false)

varnames_lr = ["D(IP)", "CPI", "1 year rate", "EBP"]
println("Median IRF at h=12, Technology shock (long-run):")
for (i, vname) in enumerate(varnames_lr)
    med = round(median(bvar5.irlr_draws[i, 12, 1, :]), digits=6)
    println("  $vname: $med")
end

# Cumulative IRF for IP (since it's in differences)
cum_ip = cumsum(bvar5.irlr_draws[1, :, 1, :], dims=1)
println("  IP (cumulative, h=12): $(round(median(cum_ip[12, :]), digits=6))")

# ─── 6) Proxy / IV Identification ────────────────────────────────────────────
println("\n=== 6) Proxy / IV Identification ===")
# Load instrument (ff4_tc from factor_data)
fd_raw, fd_headers = readdlm(joinpath(data_dir, "factor_data.csv"), ','; header=true)
fd_cols = vec(String.(fd_headers))
ff4_idx = findfirst(x -> occursin("ff4", x), fd_cols)
instrument = Float64.(fd_raw[:, ff4_idx])

# Ordering: gs1 first for IV identification (policy variable first)
y_iv = hcat(gs1, logip, logcpi, ebp)
proxy_id = ProxyIdentification(instrument)
bvar6 = bvar(y_iv, lags; prior=FlatPrior(), identification=proxy_id,
             K=K_draws, hor=24, verbose=false)

varnames_iv = ["1 year rate", "IP", "CPI", "EBP"]
println("Median IRF at h=12, MP shock (proxy/IV):")
for (i, vname) in enumerate(varnames_iv)
    med = round(median(bvar6.irproxy_draws[i, 12, 1, :]), digits=6)
    println("  $vname: $med")
end

# ─── Extra 1: FEVD with Cholesky ─────────────────────────────────────────────
println("\n=== Extra: FEVD (Cholesky, 2-year horizon) ===")
Phi_mean = dropdims(mean(bvar1.Phi_draws, dims=3), dims=3)
Sigma_mean = dropdims(mean(bvar1.Sigma_draws, dims=3), dims=3)
ny = size(y, 2)
FEVD24 = compute_fevd(Phi_mean[1:ny*lags, :], Sigma_mean, 24)
println("% of volatility explained by MP (gs1) shock at h=24:")
println("  logip     logcpi    gs1        ebp")
println("  $(round.(FEVD24.decomposition[:, 3]', digits=2))")

# ─── Extra 2: Historical Decomposition (zero+sign) ──────────────────────────
println("\n=== Extra: Historical Decomposition ===")
hd = historical_decomposition(bvar4)
T_hd = size(hd.decomposition, 1)
# Verify decomposition sums to (approximately) the observed data
y_recon = dropdims(sum(hd.decomposition, dims=3), dims=3)
y_actual = y[lags+1:lags+T_hd, :]
max_diff = maximum(abs.(y_recon .- y_actual))
println("Max absolute reconstruction error: $(round(max_diff, digits=10))")
println("  (should be ~0 if decomposition is correct)")

println("\nExample 3 complete.")
