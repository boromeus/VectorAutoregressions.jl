using VectorAutoregressions
using Test
using LinearAlgebra
using Random
using Statistics

include("test_helpers.jl")

@testset "Heteroskedasticity Weights (B2)" begin

    y, _, _ = generate_var_data(200, 2, 1)
    T, K = size(y)
    p = 1

    # ── Unit Tests ──────────────────────────────────────────────────────────

    @testset "rfvar3 uniform weights = no weights" begin
        xdata = ones(T, 1)
        B1, u1, xxi1, _, _ = rfvar3(y, p, xdata, [T, T], 0.0, 0.0)
        ww_uniform = ones(T - p)
        B2, u2, xxi2, _, _ = rfvar3(y, p, xdata, [T, T], 0.0, 0.0; ww=ww_uniform)
        @test B1 ≈ B2 atol = 1e-10
        @test u1 ≈ u2 atol = 1e-10
        @test xxi1 ≈ xxi2 atol = 1e-10
    end

    @testset "rfvar3 weights change coefficients" begin
        xdata = ones(T, 1)
        B1, _, _, _, _ = rfvar3(y, p, xdata, [T, T], 0.0, 0.0)
        # Create non-uniform weights (emphasise early observations)
        ww = [exp(-0.01 * t) for t in 1:(T - p)]
        B2, _, _, _, _ = rfvar3(y, p, xdata, [T, T], 0.0, 0.0; ww=ww)
        @test !(B1 ≈ B2)
    end

    @testset "GLS equivalence — manual rescaling vs ww" begin
        xdata = ones(T, 1)
        ww = 1.0 .+ 0.5 * randn(Random.MersenneTwister(99), T - p).^2

        # Method 1: use rfvar3 with ww
        B1, u1, xxi1, _, _ = rfvar3(y, p, xdata, [T, T], 0.0, 0.0; ww=ww)

        # Method 2: manually rescale then run OLS
        Y = y[p+1:end, :]
        X = hcat(y[p:end-1, :], ones(T - p))
        Y_w = Y ./ ww
        X_w = X ./ ww
        B2 = X_w \ Y_w
        @test B1 ≈ B2 atol = 1e-10
    end

    @testset "compute_prior_posterior Flat with uniform ww = without ww" begin
        pr1, po1, B1, u1, _, _, _ = compute_prior_posterior(y, p, FlatPrior())
        ww_uniform = ones(T - p)
        pr2, po2, B2, u2, _, _, _ = compute_prior_posterior(y, p, FlatPrior(); ww=ww_uniform)
        @test B1 ≈ B2 atol = 1e-10
        @test po1.S ≈ po2.S atol = 1e-10
    end

    @testset "compute_prior_posterior Minnesota with uniform ww = without ww" begin
        mp = MinnesotaPrior()
        pr1, po1, B1, _, _, _, _ = compute_prior_posterior(y, p, mp)
        ww_uniform = ones(T - p)
        pr2, po2, B2, _, _, _, _ = compute_prior_posterior(y, p, mp; ww=ww_uniform)
        @test B1 ≈ B2 atol = 1e-10
        @test po1.S ≈ po2.S atol = 1e-8
    end

    # ── Integration Tests ───────────────────────────────────────────────────

    @testset "bvar with heterosked_weights runs" begin
        ww = ones(T - p)
        result = bvar(y, p; prior=FlatPrior(), K=50, hor=6, fhor=4,
                       heterosked_weights=ww, verbose=false)
        @test all(isfinite.(result.Phi_draws))
        @test all(isfinite.(result.Sigma_draws))
    end

    @testset "bvar without weights = bvar with uniform weights" begin
        rng1 = Random.MersenneTwister(42)
        result1 = bvar(y, p; prior=FlatPrior(), K=30, hor=6, fhor=4,
                        verbose=false, rng=rng1)
        rng2 = Random.MersenneTwister(42)
        ww_uniform = ones(T - p)
        result2 = bvar(y, p; prior=FlatPrior(), K=30, hor=6, fhor=4,
                        heterosked_weights=ww_uniform, verbose=false, rng=rng2)
        @test result1.Phi_draws ≈ result2.Phi_draws atol = 1e-8
    end

    @testset "COVID scenario — weights reduce posterior Sigma" begin
        # Generate data with a volatility jump
        rng_data = Random.MersenneTwister(123)
        y_covid = copy(y)
        # Big shocks at rows 100-105
        for t in 100:105
            y_covid[t, :] .+= 5.0 * randn(rng_data, K)
        end

        # No weights
        res_nw = bvar(y_covid, p; prior=FlatPrior(), K=100, hor=6, fhor=4,
                       verbose=false)
        # Weights: downweight the large-volatility period
        ww_covid = ones(size(y_covid, 1) - p)
        for t in (100 - p):(105 - p)
            if 1 <= t <= length(ww_covid)
                ww_covid[t] = 5.0  # inflate ww → divides obs by 5 → downweights
            end
        end
        res_ww = bvar(y_covid, p; prior=FlatPrior(), K=100, hor=6, fhor=4,
                       heterosked_weights=ww_covid, verbose=false)

        # Posterior median of trace(Sigma) should be smaller with weights
        trace_nw = [tr(res_nw.Sigma_draws[:, :, k]) for k in 1:100]
        trace_ww = [tr(res_ww.Sigma_draws[:, :, k]) for k in 1:100]
        @test median(trace_ww) < median(trace_nw)
    end

    @testset "bvar Minnesota with heterosked_weights runs" begin
        ww = ones(T - 2 * p)  # Minnesota uses more presample
        result = bvar(y, p; prior=MinnesotaPrior(), K=50, hor=6, fhor=4,
                       heterosked_weights=ww, verbose=false)
        @test all(isfinite.(result.Phi_draws))
    end
end
