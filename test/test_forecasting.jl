using VectorAutoregressions
using Test
using LinearAlgebra
using Random
using Statistics

include("test_helpers.jl")

@testset "Forecasting" begin

    y2, _, _ = generate_var_data(200, 2, 1)
    v2 = var_estimate(y2, 1; constant = true)

    # ── Unconditional forecasting ───────────────────────────────────────────
    @testset "forecast_unconditional — output shapes" begin
        initval = y2[end:end, :]
        xdata = ones(8, 1)
        fno, fwith = forecast_unconditional(initval, xdata,
            v2.Phi, v2.Sigma, 8, 1)
        @test size(fno) == (8, 2)
        @test size(fwith) == (8, 2)
    end

    @testset "forecast_unconditional — no-shock is deterministic" begin
        initval = y2[end:end, :]
        xdata = ones(8, 1)
        rng1 = Random.MersenneTwister(1)
        rng2 = Random.MersenneTwister(999)
        fno1, _ = forecast_unconditional(initval, xdata,
            v2.Phi, v2.Sigma, 8, 1; rng = rng1)
        fno2, _ = forecast_unconditional(initval, xdata,
            v2.Phi, v2.Sigma, 8, 1; rng = rng2)
        @test fno1 ≈ fno2 atol = 1e-12
    end

    @testset "forecast_unconditional — with-shock reproducibility" begin
        initval = y2[end:end, :]
        xdata = ones(8, 1)
        rng1 = Random.MersenneTwister(42)
        rng2 = Random.MersenneTwister(42)
        _, fw1 = forecast_unconditional(initval, xdata,
            v2.Phi, v2.Sigma, 8, 1; rng = rng1)
        _, fw2 = forecast_unconditional(initval, xdata,
            v2.Phi, v2.Sigma, 8, 1; rng = rng2)
        @test fw1 ≈ fw2 atol = 1e-12
    end

    @testset "forecast_unconditional — with-shock varies across seeds" begin
        initval = y2[end:end, :]
        xdata = ones(8, 1)
        _, fw1 = forecast_unconditional(initval, xdata,
            v2.Phi, v2.Sigma, 8, 1; rng = Random.MersenneTwister(1))
        _, fw2 = forecast_unconditional(initval, xdata,
            v2.Phi, v2.Sigma, 8, 1; rng = Random.MersenneTwister(2))
        @test !(fw1 ≈ fw2)
    end

    @testset "forecast_unconditional — no-shock forecasts are finite" begin
        initval = y2[end:end, :]
        xdata = ones(8, 1)
        fno, fwith = forecast_unconditional(initval, xdata,
            v2.Phi, v2.Sigma, 8, 1; rng = Random.MersenneTwister(42))
        @test all(isfinite.(fno))
        @test all(isfinite.(fwith))
    end

    @testset "forecast_unconditional — flat-prior BVAR ≈ OLS forecast" begin
        result = bvar(y2, 1; prior = FlatPrior(), K = 200, hor = 6, fhor = 8,
            verbose = false)
        # Mean of posterior no-shock forecasts
        bvar_fno_mean = mean(result.forecasts_no_shocks, dims = 3)[:, :, 1]

        # OLS point forecast
        initval = y2[end:end, :]
        xdata = ones(8, 1)
        ols_fno, _ = forecast_unconditional(initval, xdata,
            v2.Phi, v2.Sigma, 8, 1)

        @test bvar_fno_mean ≈ ols_fno atol = 0.5
    end

    @testset "forecast_unconditional — multi-lag (p=2, 3 vars)" begin
        y3, _, _ = generate_var_data(300, 3, 2)
        v3 = var_estimate(y3, 2; constant = true)
        initval = y3[end - 1:end, :]
        xdata = ones(6, 1)
        fno, fwith = forecast_unconditional(initval, xdata,
            v3.Phi, v3.Sigma, 6, 2; rng = Random.MersenneTwister(42))
        @test size(fno) == (6, 3)
        @test size(fwith) == (6, 3)
        @test all(isfinite.(fno))
    end

    # ── Conditional forecasting ─────────────────────────────────────────────
    @testset "forecast_conditional — hits target path exactly" begin
        initval = y2[end:end, :]
        xdata = ones(8, 1)
        # Condition variable 1 to follow a specific path
        endo_index = [1]
        endo_path = zeros(8, 1)
        for t in 1:8
            endo_path[t, 1] = 0.5 * sin(t * π / 4)
        end

        cond_f, shocks = forecast_conditional(endo_path, endo_index,
            initval, xdata, v2.Phi, v2.Sigma, 8, 1;
            rng = Random.MersenneTwister(42))

        @test size(cond_f) == (8, 2)
        @test size(shocks) == (8, 2)
        # Conditioned variable must match the target
        @test cond_f[:, 1] ≈ endo_path[:, 1] atol = 1e-6
    end

    @testset "forecast_conditional — condition on second variable" begin
        initval = y2[end:end, :]
        xdata = ones(6, 1)
        endo_index = [2]
        endo_path = fill(0.3, 6, 1)

        cond_f, shocks = forecast_conditional(endo_path, endo_index,
            initval, xdata, v2.Phi, v2.Sigma, 6, 1;
            rng = Random.MersenneTwister(77))

        @test cond_f[:, 2] ≈ endo_path[:, 1] atol = 1e-6
        @test size(shocks) == (6, 2)
    end

    @testset "forecast_conditional — condition on all variables" begin
        initval = y2[end:end, :]
        xdata = ones(4, 1)
        endo_index = [1, 2]
        endo_path = [0.1 0.2; 0.3 0.4; 0.5 0.6; 0.7 0.8]

        cond_f, shocks = forecast_conditional(endo_path, endo_index,
            initval, xdata, v2.Phi, v2.Sigma, 4, 1;
            rng = Random.MersenneTwister(42))

        @test cond_f ≈ endo_path atol = 1e-6
    end

    @testset "forecast_conditional — shocks are finite" begin
        initval = y2[end:end, :]
        xdata = ones(6, 1)
        endo_index = [1]
        endo_path = zeros(6, 1)

        cond_f, shocks = forecast_conditional(endo_path, endo_index,
            initval, xdata, v2.Phi, v2.Sigma, 6, 1;
            rng = Random.MersenneTwister(42))
        @test all(isfinite.(cond_f))
        @test all(isfinite.(shocks))
    end

    @testset "forecast_conditional — reproducibility" begin
        initval = y2[end:end, :]
        xdata = ones(6, 1)
        endo_index = [1]
        endo_path = fill(0.0, 6, 1)

        cf1, _ = forecast_conditional(endo_path, endo_index,
            initval, xdata, v2.Phi, v2.Sigma, 6, 1;
            rng = Random.MersenneTwister(42))
        cf2, _ = forecast_conditional(endo_path, endo_index,
            initval, xdata, v2.Phi, v2.Sigma, 6, 1;
            rng = Random.MersenneTwister(42))

        # Conditioned variable is deterministic
        @test cf1[:, 1] ≈ cf2[:, 1] atol = 1e-12
        # Free variable is also deterministic given same RNG
        @test cf1[:, 2] ≈ cf2[:, 2] atol = 1e-12
    end

    # ── Conditional Exo-Index Forecasting (B3) ──────────────────────────────
    @testset "forecast_conditional_exo — hits target path" begin
        initval = y2[end:end, :]
        xdata = ones(8, 1)
        endo_index = [1]
        exo_index = [1]
        endo_path = zeros(8, 1)
        for t in 1:8
            endo_path[t, 1] = 0.5 * sin(t * π / 4)
        end

        cond_f, shocks = forecast_conditional_exo(endo_path, endo_index, exo_index,
            initval, xdata, v2.Phi, v2.Sigma, 8, 1;
            rng = Random.MersenneTwister(42))

        @test size(cond_f) == (8, 2)
        @test cond_f[:, 1] ≈ endo_path[:, 1] atol = 1e-6
    end

    @testset "forecast_conditional_exo — square system check" begin
        initval = y2[end:end, :]
        xdata = ones(4, 1)
        # Mismatched lengths should error
        @test_throws ArgumentError forecast_conditional_exo(
            zeros(4, 1), [1], [1, 2], initval, xdata, v2.Phi, v2.Sigma, 4, 1)
    end

    @testset "forecast_conditional_exo — condition 2 vars with 2 shocks" begin
        initval = y2[end:end, :]
        xdata = ones(4, 1)
        endo_index = [1, 2]
        exo_index = [1, 2]
        endo_path = [0.1 0.2; 0.3 0.4; 0.5 0.6; 0.7 0.8]

        cond_f, shocks = forecast_conditional_exo(endo_path, endo_index, exo_index,
            initval, xdata, v2.Phi, v2.Sigma, 4, 1;
            rng = Random.MersenneTwister(42))

        @test cond_f[:, 1] ≈ endo_path[:, 1] atol = 1e-6
        @test cond_f[:, 2] ≈ endo_path[:, 2] atol = 1e-6
    end

    @testset "forecast_conditional_exo — non-identity Omega" begin
        initval = y2[end:end, :]
        xdata = ones(6, 1)
        endo_index = [1]
        exo_index = [1]
        endo_path = fill(0.3, 6, 1)

        # Random orthogonal rotation
        rng_rot = Random.MersenneTwister(99)
        A = randn(rng_rot, 2, 2)
        Q, _ = qr(A)
        Omega = Matrix(Q)

        cond_f, _ = forecast_conditional_exo(endo_path, endo_index, exo_index,
            initval, xdata, v2.Phi, v2.Sigma, 6, 1;
            Omega=Omega, rng = Random.MersenneTwister(77))

        @test cond_f[:, 1] ≈ endo_path[:, 1] atol = 1e-6
    end

    @testset "forecast_conditional_exo — only exo_index shocks solved" begin
        initval = y2[end:end, :]
        xdata = ones(6, 1)
        endo_index = [1]
        exo_index = [1]
        endo_path = fill(0.0, 6, 1)

        rng1 = Random.MersenneTwister(42)
        _, shocks = forecast_conditional_exo(endo_path, endo_index, exo_index,
            initval, xdata, v2.Phi, v2.Sigma, 6, 1; rng=rng1)

        # Non-exo shocks (column 2) should be the random draws
        rng2 = Random.MersenneTwister(42)
        EPSi_ref = randn(rng2, 6, 2)
        # Column 2 should match original random draw
        @test shocks[:, 2] ≈ EPSi_ref[:, 2] atol = 1e-12
    end

    @testset "forecast_conditional_exo — reproducibility" begin
        initval = y2[end:end, :]
        xdata = ones(6, 1)
        endo_index = [2]
        exo_index = [1]
        endo_path = fill(0.5, 6, 1)

        cf1, _ = forecast_conditional_exo(endo_path, endo_index, exo_index,
            initval, xdata, v2.Phi, v2.Sigma, 6, 1;
            rng = Random.MersenneTwister(42))
        cf2, _ = forecast_conditional_exo(endo_path, endo_index, exo_index,
            initval, xdata, v2.Phi, v2.Sigma, 6, 1;
            rng = Random.MersenneTwister(42))

        @test cf1 ≈ cf2 atol = 1e-12
    end
end
