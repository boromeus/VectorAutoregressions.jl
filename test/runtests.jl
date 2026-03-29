using VectorAutoregressions
using Test
using DelimitedFiles: readdlm
using LinearAlgebra
using Random
using Statistics

path = joinpath(dirname(pathof(VectorAutoregressions)), "..")

# ─── Helper: generate simple VAR data ──────────────────────────────────────────

function generate_var_data(T::Int, K::Int, p::Int; rng=Random.MersenneTwister(42))
    Phi_true = zeros(K * p, K)
    # Simple diagonal AR(1) coefficients
    for i in 1:K
        Phi_true[i, i] = 0.5
    end
    Sigma_true = Matrix{Float64}(I, K, K) * 0.5
    Sigma_true[1, 2] = 0.1
    Sigma_true[2, 1] = 0.1
    if K > 2
        Sigma_true[1, 3] = 0.05
        Sigma_true[3, 1] = 0.05
    end

    L = cholesky(Hermitian(Sigma_true)).L
    y = zeros(T + 100, K)  # burn-in
    for t in p+1:T+100
        for lag in 1:p
            y[t, :] += Phi_true[(lag-1)*K+1:lag*K, :]' * y[t-lag, :]
        end
        y[t, :] += L * randn(rng, K)
    end
    return y[101:end, :], Phi_true, Sigma_true
end

# ──────────────────────────────────────────────────────────────────────────────
# Test Suite
# ──────────────────────────────────────────────────────────────────────────────

@testset "VectorAutoregressions.jl" begin

    # ─── Utilities ──────────────────────────────────────────────────────────
    @testset "Utilities" begin
        @testset "lagmatrix" begin
            y = [1.0 2.0; 3.0 4.0; 5.0 6.0; 7.0 8.0]
            X = lagmatrix(y, 1)
            @test size(X) == (3, 2)
            @test X[1, :] == [1.0, 2.0]

            X2 = lagmatrix(y, 1; constant=true)
            @test size(X2) == (3, 3)
            @test X2[:, 3] == ones(3)

            X3 = lagmatrix(y, 2; constant=true, trend=true)
            @test size(X3) == (2, 6)  # 2*2 lags + const + trend
        end

        @testset "companion_form" begin
            K, p = 2, 2
            Phi = [0.5 0.1; 0.0 0.4; 0.1 0.0; 0.0 0.1]
            F = companion_form(Phi, K, p)
            @test size(F) == (4, 4)
            @test F[3:4, 1:2] == I(2)
        end

        @testset "check_stability" begin
            Phi_stable = [0.3 0.0; 0.0 0.3]
            @test check_stability(Phi_stable, 2, 1) == true
            Phi_unstable = [1.1 0.0; 0.0 1.1]
            @test check_stability(Phi_unstable, 2, 1) == false
        end

        @testset "rand_inverse_wishart" begin
            rng = Random.MersenneTwister(123)
            S = Matrix{Float64}(I, 3, 3)
            W = rand_inverse_wishart(10, S; rng=rng)
            @test size(W) == (3, 3)
            @test issymmetric(round.(W; digits=10))
            @test all(eigvals(Hermitian(W)) .> 0)
        end

        @testset "vech_ivech" begin
            A = [1.0 2.0 3.0; 2.0 4.0 5.0; 3.0 5.0 6.0]
            v = vech(A)
            @test length(v) == 6
            A2 = ivech(v, 3)
            @test A2 == A
        end

        @testset "var2ma" begin
            Phi = [0.5 0.0; 0.0 0.3]
            Psi = var2ma(Phi, 3)
            @test size(Psi) == (2, 2, 3)
            @test Psi[:, :, 1] ≈ I(2)
        end

        @testset "ols_svd" begin
            X = randn(Random.MersenneTwister(1), 100, 3)
            beta_true = [1.0, 2.0, 3.0]
            y = X * beta_true + 0.01 * randn(Random.MersenneTwister(2), 100, 1)
            Phi, resid, xxi = ols_svd(y, X)
            @test vec(Phi) ≈ beta_true atol = 0.1
        end

        @testset "generate_rotation_matrix" begin
            rng = Random.MersenneTwister(42)
            Q = generate_rotation_matrix(3; rng=rng)
            @test size(Q) == (3, 3)
            @test Q' * Q ≈ I(3) atol = 1e-10
        end

        @testset "matrix_operations" begin
            K = commutation_matrix(2, 3)
            @test size(K) == (6, 6)
            D = duplication_matrix(3)
            @test size(D) == (9, 6)
            L = elimination_matrix(3)
            @test size(L) == (6, 9)
        end
    end

    # ─── VAR Estimation ────────────────────────────────────────────────────
    @testset "VAR Estimation" begin
        y, Phi_true, Sigma_true = generate_var_data(500, 3, 1)

        @testset "var_estimate" begin
            v = var_estimate(y, 1; constant=true)
            @test v.nvar == 3
            @test v.nlags == 1
            @test v.nobs == 499
            @test size(v.Phi) == (4, 3)  # 3 lags + constant
            @test size(v.Sigma) == (3, 3)
            @test size(v.residuals) == (499, 3)
            @test issymmetric(round.(v.Sigma; digits=10))

            # AR coefficients should be close to true values
            @test v.Phi[1:3, :] ≈ Phi_true atol = 0.15
        end

        @testset "var_lagorder" begin
            lag = var_lagorder(y, 8; ic="bic", verbose=false)
            @test lag isa Int
            @test 1 <= lag <= 8
        end

        @testset "information_criteria" begin
            v = var_estimate(y, 1; constant=true)
            ic = information_criteria(v)
            @test ic isa InfoCriteria
            @test isfinite(ic.aic)
            @test isfinite(ic.bic)
            @test isfinite(ic.hqic)
        end
    end

    # ─── IRF ───────────────────────────────────────────────────────────────
    @testset "IRF" begin
        y, _, _ = generate_var_data(500, 3, 1)
        v = var_estimate(y, 1; constant=true)

        @testset "compute_irf (Cholesky)" begin
            ir = compute_irf(v.Phi, v.Sigma, 12)
            @test size(ir) == (3, 12, 3)
            # Impact: lower triangular (Cholesky)
            @test ir[1, 1, 1] > 0  # own shock positive
        end

        @testset "compute_irf_longrun" begin
            ir, Q = compute_irf_longrun(v.Phi, v.Sigma, 12, 1)
            @test size(ir) == (3, 12, 3)
        end
    end

    # ─── Priors ────────────────────────────────────────────────────────────
    @testset "Priors" begin
        y, _, _ = generate_var_data(200, 3, 1)

        @testset "get_prior_moments" begin
            mu, sig, delta = get_prior_moments(y, 1)
            @test length(mu) == 3
            @test length(sig) == 3
            @test all(sig .> 0)
            @test length(delta) == 3
        end

        @testset "build_dummy_observations" begin
            mu, sig, delta = get_prior_moments(y, 1)
            prior = MinnesotaPrior()
            Yd, Xd, _ = build_dummy_observations(prior, 3, 1, sig, delta, vec(mu))
            @test size(Yd, 2) == 3
            @test size(Xd, 2) == 1  # exogenous part only (constant)
        end
    end

    # ─── BVAR ──────────────────────────────────────────────────────────────
    @testset "BVAR" begin
        y, _, _ = generate_var_data(200, 2, 1)

        @testset "bvar flat prior" begin
            result = bvar(y, 1; prior=FlatPrior(), K=50, hor=6, fhor=4,
                          verbose=false)
            @test result.ndraws == 50
            @test result.nvar == 2
            @test size(result.Phi_draws) == (3, 2, 50)  # [K*p+const, K, ndraws]
            @test size(result.Sigma_draws) == (2, 2, 50)
            @test size(result.ir_draws, 2) == 6  # hor
        end

        @testset "bvar Minnesota prior" begin
            result = bvar(y, 1; prior=MinnesotaPrior(), K=50, hor=6, fhor=4,
                          verbose=false)
            @test result.ndraws == 50
            @test result.prior isa MinnesotaPrior
        end
    end

    # ─── FEVD ──────────────────────────────────────────────────────────────
    @testset "FEVD" begin
        y, _, _ = generate_var_data(200, 2, 1)
        v = var_estimate(y, 1; constant=true)

        @testset "compute_fevd" begin
            fevd = compute_fevd(v.Phi, v.Sigma, 12)
            @test size(fevd.decomposition) == (2, 2)
            # Rows should sum to 100 (percentages)
            for i in 1:2
                @test sum(fevd.decomposition[i, :]) ≈ 100.0 atol = 1.0
            end
        end
    end

    # ─── Forecasting ──────────────────────────────────────────────────────
    @testset "Forecasting" begin
        y, _, _ = generate_var_data(200, 2, 1)
        v = var_estimate(y, 1; constant=true)

        @testset "forecast_unconditional" begin
            initval = y[end:end, :]
            xdata = ones(8, 1)  # constant
            fno, fwith = forecast_unconditional(initval, xdata,
                                                 v.Phi, v.Sigma, 8, 1)
            @test size(fno) == (8, 2)
            @test size(fwith) == (8, 2)
        end
    end

    # ─── Connectedness ────────────────────────────────────────────────────
    @testset "Connectedness" begin
        y, _, _ = generate_var_data(200, 3, 1)
        v = var_estimate(y, 1; constant=true)

        c = compute_connectedness(v.Phi, v.Sigma, 12)
        @test c isa ConnectednessResult
        @test isfinite(c.index)
        @test length(c.from_all_to_unit) == 3
        @test length(c.net) == 3
    end

    # ─── Historical Decomposition ─────────────────────────────────────────
    @testset "Historical Decomposition" begin
        y, _, _ = generate_var_data(200, 2, 1)
        result = bvar(y, 1; prior=FlatPrior(), K=20, hor=6, fhor=4,
                       verbose=false)

        hd = historical_decomposition(result)
        @test hd isa HistDecompResult
        @test size(hd.structural_shocks, 2) == 2
    end

    # ─── Marginal Likelihood ──────────────────────────────────────────────
    @testset "Marginal Likelihood" begin
        y, _, _ = generate_var_data(200, 2, 1)

        ml = compute_marginal_likelihood(y, 1, MinnesotaPrior())
        @test isfinite(ml)
    end

    # ─── Filters ──────────────────────────────────────────────────────────
    @testset "Filters" begin
        rng = Random.MersenneTwister(42)
        x = cumsum(randn(rng, 200))

        @testset "HP filter" begin
            r = hp_filter(x, 1600)
            @test length(r.trend) == 200
            @test length(r.cycle) == 200
            @test r.trend .+ r.cycle ≈ x atol = 1e-10
        end

        @testset "Hamilton filter" begin
            r = hamilton_filter(x, 8, 4)
            @test length(r.trend) == 200
            # First d+h-1 entries should be NaN
            @test isnan(r.cycle[1])
            @test !isnan(r.cycle[end])
        end

        @testset "BK filter" begin
            fX = bk_filter(x, 6, 32)
            @test length(fX) == 200
        end
    end

    # ─── Local Projections ────────────────────────────────────────────────
    @testset "Local Projections" begin
        y, _, _ = generate_var_data(300, 2, 1)

        @testset "lp_irf" begin
            r = lp_irf(y, 4, 12; conf_level=0.90)
            @test r isa LPResult
            @test size(r.irf) == (13, 4)  # (H+1) × K²
            @test r.horizon == 12
        end

        @testset "lp_lagorder" begin
            lags = lp_lagorder(y, 8, 12, "bic")
            @test length(lags) == 12
            @test all(1 .<= lags .<= 8)
        end
    end

    # ─── Panel VAR ────────────────────────────────────────────────────────
    @testset "Panel VAR" begin
        panels = [generate_var_data(100, 2, 1)[1] for _ in 1:3]

        @testset "pooled" begin
            r = panel_var(panels, 1; method=:pooled)
            @test r isa PanelVARResult
            @test r.method == :pooled
            @test size(r.Phi, 2) == 2
        end

        @testset "unit-by-unit" begin
            r = panel_var(panels, 1; method=:unit)
            @test r isa PanelVARResult
            @test r.method == :unit
            @test length(r.unit_results) == 3
        end
    end

    # ─── Kalman Filter ────────────────────────────────────────────────────
    @testset "Kalman Filter" begin
        y, _, _ = generate_var_data(100, 2, 1)
        v = var_estimate(y, 1; constant=true)

        result = kalman_filter(v.Phi, v.Sigma, y)
        @test isfinite(result.logL)
        @test size(result.states) == (100, 2)
    end

    # ─── Principal Components ─────────────────────────────────────────────
    @testset "Principal Components" begin
        rng = Random.MersenneTwister(99)
        X = randn(rng, 100, 10)
        pc = principal_components(X, 2)
        @test size(pc.factors) == (100, 2)
        @test size(pc.loadings) == (10, 2)
        @test length(pc.eigenvalues) == 100
    end

end  # top-level testset
