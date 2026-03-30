# Example 11: Connectedness in the Cryptocurrency Market
# Translation of MATLAB example_11_connectedness.m
#
# 1) Estimate VAR with Ridge/Lasso/ElasticNet regularization
# 2) Compute Diebold-Yilmaz connectedness measures
# 3) Rolling window estimation

using VectorAutoregressions
using DelimitedFiles, Statistics, Random, LinearAlgebra

# ─── Load Data ─────────────────────────────────────────────────────────────────
data_dir = joinpath(@__DIR__, "data")
price_raw, price_headers = readdlm(joinpath(data_dir, "crypto_price.csv"), ','; header=true)
price_cols = vec(String.(price_headers))

# Select a subset of cryptocurrencies (medium cross-section)
sel1_names = ["BinanceCoin", "BitcoinCash", "Bitcoin", "Cardano", "Dash",
              "EOS", "Ethereum", "EthereumClassic", "IOTA", "Litecoin",
              "Monero", "NEM", "Neo", "Qtum", "Stellar", "TRON", "VeChain", "XRP", "Zcash"]

# Map names to column indices
sel1_idx = Int[]
for name in sel1_names
    idx = findfirst(==(name), price_cols)
    if idx !== nothing
        push!(sel1_idx, idx)
    end
end
cryptonames = price_cols[sel1_idx]
println("Selected $(length(sel1_idx)) cryptocurrencies")

# Extract log prices
price_data = Float64.(price_raw[:, sel1_idx])
# Remove any rows with zeros or negatives
valid_rows = all(price_data .> 0, dims=2) |> vec
y = log.(price_data[valid_rows, :])
T_obs = size(y, 1)
println("Data: $T_obs obs × $(size(y, 2)) cryptos")

lags = 3
nethor = 10

# ─── 1) Ridge Estimation + Connectedness ─────────────────────────────────────
println("\n=== Ridge VAR ===")
var_ridge = var_estimate(y, lags; regularization=:ridge, lambda=0.25)
conn_ridge = compute_connectedness(var_ridge.Phi, var_ridge.Sigma, nethor)
println("Ridge Connectedness Index: $(round(conn_ridge.index, digits=2))")

# ─── 2) Lasso Estimation + Connectedness ─────────────────────────────────────
println("\n=== Lasso VAR ===")
var_lasso = var_estimate(y, lags; regularization=:lasso, lambda=0.25)
conn_lasso = compute_connectedness(var_lasso.Phi, var_lasso.Sigma, nethor)
println("Lasso Connectedness Index: $(round(conn_lasso.index, digits=2))")

# ─── 3) Elastic Net + Connectedness ──────────────────────────────────────────
println("\n=== Elastic Net VAR ===")
var_enet = var_estimate(y, lags; regularization=:elastic_net, lambda=0.25, alpha=0.5)
conn_enet = compute_connectedness(var_enet.Phi, var_enet.Sigma, nethor)
println("ElasticNet Connectedness Index: $(round(conn_enet.index, digits=2))")

# ─── 4) Minnesota Prior BVAR + Connectedness ────────────────────────────────
println("\n=== Minnesota BVAR ===")
# Use default Minnesota prior (skip optimization for speed with large K)
best_prior = MinnesotaPrior(tau=1.0, decay=0.5, lambda=5.0, mu=2.0)
K_draws = parse(Int, get(ENV, "BVAR_K", "200"))
bvar1 = bvar(y, lags; prior=best_prior, K=K_draws, hor=nethor, verbose=false)

# Compute connectedness for each draw
conn_indices = zeros(bvar1.ndraws)
for k in 1:bvar1.ndraws
    Phi_k = bvar1.Phi_draws[:, :, k]
    Sigma_k = bvar1.Sigma_draws[:, :, k]
    c = compute_connectedness(Phi_k, Sigma_k, nethor)
    conn_indices[k] = c.index
end
println("BVAR Minnesota Connectedness Index (mean): $(round(mean(conn_indices), digits=2))")

# ─── Summary Table ──────────────────────────────────────────────────────────
println("\n=== Overall Connectedness Index ===")
println("  BVAR-Minn   Ridge   Lasso   ElasticNet")
println("  $(round(mean(conn_indices), digits=2))     " *
        "$(round(conn_ridge.index, digits=2))   " *
        "$(round(conn_lasso.index, digits=2))   " *
        "$(round(conn_enet.index, digits=2))")

# Spillover (From Unit to All)
println("\n=== From Unit to All (Spillover) ===")
println("  $(rpad("Crypto", 20)) Ridge    Lasso    ENet")
for (i, cn) in enumerate(cryptonames)
    println("  $(rpad(cn, 20)) " *
            "$(round(conn_ridge.from_unit_to_all[i], digits=2))  " *
            "$(round(conn_lasso.from_unit_to_all[i], digits=2))  " *
            "$(round(conn_enet.from_unit_to_all[i], digits=2))")
end

# Vulnerability (From All to Unit)
println("\n=== From All to Unit (Vulnerability) ===")
println("  $(rpad("Crypto", 20)) Ridge    Lasso    ENet")
for (i, cn) in enumerate(cryptonames)
    println("  $(rpad(cn, 20)) " *
            "$(round(conn_ridge.from_all_to_unit[i], digits=2))  " *
            "$(round(conn_lasso.from_all_to_unit[i], digits=2))  " *
            "$(round(conn_enet.from_all_to_unit[i], digits=2))")
end

# ─── 5) Rolling Window ──────────────────────────────────────────────────────
println("\n=== Rolling Window Estimation ===")
W = 200
n_windows = T_obs - W - lags
println("Window size: $W, Number of windows: $n_windows")

n_roll = parse(Int, get(ENV, "BVAR_ROLL", "10"))
rolling_idx = zeros(min(n_windows, n_roll), 3)  # store first n_roll for speed
for dd in 1:min(n_windows, n_roll)
    span = dd:W+dd
    v_r = var_estimate(y[span, :], lags; regularization=:ridge, lambda=0.5)
    v_l = var_estimate(y[span, :], lags; regularization=:lasso, lambda=0.5)
    v_e = var_estimate(y[span, :], lags; regularization=:elastic_net, lambda=0.5, alpha=0.5)

    rolling_idx[dd, 1] = compute_connectedness(v_r.Phi, v_r.Sigma, nethor).index
    rolling_idx[dd, 2] = compute_connectedness(v_l.Phi, v_l.Sigma, nethor).index
    rolling_idx[dd, 3] = compute_connectedness(v_e.Phi, v_e.Sigma, nethor).index

    if dd % 10 == 0
        println("  Window $dd/$n_windows")
    end
end

println("\nRolling connectedness (first $n_roll windows):")
println("  Window   Ridge    Lasso    ENet")
for dd in [1, min(5, n_roll), min(10, n_roll)]
    if dd <= size(rolling_idx, 1)
        println("  $(rpad(string(dd), 8)) " *
                "$(round(rolling_idx[dd, 1], digits=2))  " *
                "$(round(rolling_idx[dd, 2], digits=2))  " *
                "$(round(rolling_idx[dd, 3], digits=2))")
    end
end

println("\nExample 11 complete.")
