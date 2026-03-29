using VectorAutoregressions
using Test
using LinearAlgebra
using Random
using Statistics

include("test_helpers.jl")

@testset "FEVD" begin

    # ── Baseline setup ──────────────────────────────────────────────────────
    y2, _, _ = generate_var_data(200, 2, 1)
    v2 = var_estimate(y2, 1; constant = true)

    y3, _, _ = generate_var_data(200, 3, 1)
    v3 = var_estimate(y3, 1; constant = true)

    @testset "compute_fevd — row sums = 100 at multiple horizons" begin
        for h in [1, 6, 12, 24]
            fevd = compute_fevd(v2.Phi, v2.Sigma, h)
            @test size(fevd.decomposition) == (2, 2)
            @test fevd.horizon == h
            for i in 1:2
                @test sum(fevd.decomposition[i, :]) ≈ 100.0 atol = 1e-8
            end
        end
    end

    @testset "compute_fevd — 3-variable row sums = 100" begin
        for h in [1, 12]
            fevd = compute_fevd(v3.Phi, v3.Sigma, h)
            @test size(fevd.decomposition) == (3, 3)
            for i in 1:3
                @test sum(fevd.decomposition[i, :]) ≈ 100.0 atol = 1e-8
            end
        end
    end

    @testset "compute_fevd — impact (h=1) own-shock dominance" begin
        # For diagonal AR(1) DGP, own-shock should be the largest at impact
        fevd = compute_fevd(v2.Phi, v2.Sigma, 1)
        for i in 1:2
            @test fevd.decomposition[i, i] == maximum(fevd.decomposition[i, :])
        end
    end

    @testset "compute_fevd — all entries non-negative" begin
        fevd = compute_fevd(v3.Phi, v3.Sigma, 12)
        @test all(fevd.decomposition .>= -1e-10)
    end

    @testset "compute_fevd — univariate (K=1)" begin
        y1 = randn(Random.MersenneTwister(99), 200, 1)
        v1 = var_estimate(y1 .+ cumsum(0.3 * randn(Random.MersenneTwister(100), 200, 1), dims = 1), 1; constant = true)
        fevd = compute_fevd(v1.Phi, v1.Sigma, 12)
        @test size(fevd.decomposition) == (1, 1)
        @test fevd.decomposition[1, 1] ≈ 100.0 atol = 1e-8
    end

    @testset "compute_fevd — multi-lag (p=2)" begin
        y3p2, _, _ = generate_var_data(300, 3, 2)
        v3p2 = var_estimate(y3p2, 2; constant = true)
        for h in [1, 6, 12]
            fevd = compute_fevd(v3p2.Phi, v3p2.Sigma, h)
            @test size(fevd.decomposition) == (3, 3)
            for i in 1:3
                @test sum(fevd.decomposition[i, :]) ≈ 100.0 atol = 1e-8
            end
        end
    end

    @testset "compute_fevd — custom Omega rotation" begin
        rng = Random.MersenneTwister(42)
        Q = generate_rotation_matrix(2; rng = rng)
        fevd = compute_fevd(v2.Phi, v2.Sigma, 12; Omega = Q)
        @test size(fevd.decomposition) == (2, 2)
        for i in 1:2
            @test sum(fevd.decomposition[i, :]) ≈ 100.0 atol = 1e-8
        end
        @test all(fevd.decomposition .>= -1e-10)
    end

    @testset "compute_fevd — identity Omega = default" begin
        fevd_default = compute_fevd(v2.Phi, v2.Sigma, 12)
        fevd_identity = compute_fevd(v2.Phi, v2.Sigma, 12;
            Omega = Matrix{Float64}(I, 2, 2))
        @test fevd_default.decomposition ≈ fevd_identity.decomposition atol = 1e-12
    end

    @testset "fevd_posterior — type and dimensions" begin
        result = bvar(y2, 1; prior = FlatPrior(), K = 50, hor = 6, fhor = 4,
            verbose = false)
        fp = fevd_posterior(result; horizons = [1, 3, 6])
        @test fp isa FEVDPosteriorResult
        @test size(fp.median) == (2, 2, 3)
        @test size(fp.lower) == (2, 2, 3)
        @test size(fp.upper) == (2, 2, 3)
        @test fp.conf_level == 0.68
    end

    @testset "fevd_posterior — band ordering lower ≤ median ≤ upper" begin
        result = bvar(y2, 1; prior = FlatPrior(), K = 100, hor = 6, fhor = 4,
            verbose = false)
        fp = fevd_posterior(result; horizons = [1, 3, 6], conf_level = 0.90)
        @test all(fp.lower .<= fp.median .+ 1e-10)
        @test all(fp.median .<= fp.upper .+ 1e-10)
    end

    @testset "fevd_posterior — median rows ≈ 100" begin
        result = bvar(y2, 1; prior = FlatPrior(), K = 100, hor = 6, fhor = 4,
            verbose = false)
        fp = fevd_posterior(result; horizons = [1, 6])
        for hi in 1:2
            for i in 1:2
                @test sum(fp.median[i, :, hi]) ≈ 100.0 atol = 5.0
            end
        end
    end
end
