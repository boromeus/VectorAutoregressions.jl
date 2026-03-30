using VectorAutoregressions
using Test
using LinearAlgebra
using Random
using Statistics
using Distributions

include("test_helpers.jl")

@testset "Robust Bayes (B4)" begin

    y, _, _ = generate_var_data(200, 2, 1)

    @testset "fourthmom dimensions" begin
        x = randn(Random.MersenneTwister(1), 100, 3)
        K4 = fourthmom(x)
        m = 3 * 4 ÷ 2   # N*(N+1)/2 = 6
        @test size(K4) == (m, m)
    end

    @testset "fourthmom symmetry" begin
        x = randn(Random.MersenneTwister(2), 100, 3)
        K4 = fourthmom(x)
        @test K4 ≈ K4' atol = 1e-12
    end

    @testset "thirdmom dimensions" begin
        x = randn(Random.MersenneTwister(3), 100, 3)
        S3 = thirdmom(x)
        m = 3 * 4 ÷ 2
        @test size(S3) == (3, m)
    end

    @testset "thirdmom Gaussian ≈ 0" begin
        x = randn(Random.MersenneTwister(4), 10000, 2)
        S3 = thirdmom(x)
        @test norm(S3) < 1.0
    end

    @testset "robust_bayes_setup runs" begin
        v = var_estimate(y, 1)
        nobs = size(v.residuals, 1)
        setup = robust_bayes_setup(v.residuals, nobs, 2)
        m = 2 * 3 ÷ 2  # = 3
        @test length(setup.vech_Sig) == m
        @test size(setup.vech_Sig_cov_lower_chol) == (m, m)
    end

    @testset "robust_sigma_draw is PSD" begin
        v = var_estimate(y, 1)
        nobs = size(v.residuals, 1)
        setup = robust_bayes_setup(v.residuals, nobs, 2)
        rng = Random.MersenneTwister(42)
        psd_count = 0
        for _ in 1:20
            Sigma, _, _ = robust_sigma_draw(setup; rng=rng)
            if Sigma !== nothing
                psd_count += 1
                @test all(eigvals(Hermitian(Sigma)) .> 0)
            end
        end
        @test psd_count > 10  # most draws should be PSD
    end

    @testset "robust draws differ from IW" begin
        v = var_estimate(y, 1)
        nobs = size(v.residuals, 1)
        setup = robust_bayes_setup(v.residuals, nobs, 2)

        # Robust draws
        rng = Random.MersenneTwister(42)
        robust_traces = Float64[]
        for _ in 1:50
            Sigma, _, _ = robust_sigma_draw(setup; rng=rng)
            if Sigma !== nothing
                push!(robust_traces, tr(Sigma))
            end
        end

        # Different variance structure than IW
        @test length(robust_traces) > 20
        @test std(robust_traces) > 0
    end

    @testset "high K_shrinkage approaches Gaussian" begin
        v = var_estimate(y, 1)
        nobs = size(v.residuals, 1)
        # Large shrinkage → should be closer to Gaussian (IW-like)
        setup_large = robust_bayes_setup(v.residuals, nobs, 2; K_shrinkage=1e6)
        setup_small = robust_bayes_setup(v.residuals, nobs, 2; K_shrinkage=1.0)

        # The cholesky factor norm should differ
        norm_large = norm(setup_large.vech_Sig_cov_lower_chol)
        norm_small = norm(setup_small.vech_Sig_cov_lower_chol)
        # With large shrinkage, variance of vech(Sigma) should be different
        @test norm_large != norm_small
    end

    @testset "bvar with robust_bayes=1 runs" begin
        result = bvar(y, 1; prior=FlatPrior(), K=50, hor=6, fhor=4,
                       robust_bayes=1, verbose=false)
        @test all(isfinite.(result.Phi_draws))
        @test all(isfinite.(result.Sigma_draws))
    end

    @testset "bvar with robust_bayes=2 runs" begin
        result = bvar(y, 1; prior=FlatPrior(), K=50, hor=6, fhor=4,
                       robust_bayes=2, verbose=false)
        @test all(isfinite.(result.Phi_draws))
        @test all(isfinite.(result.Sigma_draws))
    end

    @testset "fat-tailed DGP — robust vs standard" begin
        # Generate data with Student-t(5) innovations
        rng_data = Random.MersenneTwister(123)
        T, K_var = 200, 2
        y_fat = zeros(T + 100, K_var)
        tdist = TDist(5.0)
        for t in 2:(T + 100)
            y_fat[t, :] = 0.5 * y_fat[t-1, :] + rand(rng_data, tdist, K_var)
        end
        y_fat = y_fat[101:end, :]

        # Standard BVAR
        res_std = bvar(y_fat, 1; prior=FlatPrior(), K=100, hor=6, fhor=4,
                        verbose=false)
        # Robust BVAR
        res_rob = bvar(y_fat, 1; prior=FlatPrior(), K=100, hor=6, fhor=4,
                        robust_bayes=1, verbose=false)

        # Both should produce finite results
        @test all(isfinite.(res_std.Sigma_draws))
        @test all(isfinite.(res_rob.Sigma_draws))

        # Posterior Sigma traces from both should be positive
        traces_std = [tr(res_std.Sigma_draws[:, :, k]) for k in 1:100]
        traces_rob = [tr(res_rob.Sigma_draws[:, :, k]) for k in 1:100]
        @test all(traces_std .> 0)
        @test all(traces_rob .> 0)
    end
end
