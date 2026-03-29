using VectorAutoregressions
using Test
using LinearAlgebra
using Random
using Statistics

include("test_helpers.jl")

@testset "Connectedness" begin

    y3, _, _ = generate_var_data(200, 3, 1)
    v3 = var_estimate(y3, 1; constant = true)

    @testset "compute_connectedness — type and dimensions" begin
        c = compute_connectedness(v3.Phi, v3.Sigma, 12)
        @test c isa ConnectednessResult
        @test isfinite(c.index)
        @test length(c.from_all_to_unit) == 3
        @test length(c.from_unit_to_all) == 3
        @test length(c.net) == 3
        @test size(c.theta) == (3, 3)
    end

    @testset "compute_connectedness — index in [0, 100]" begin
        c = compute_connectedness(v3.Phi, v3.Sigma, 12)
        @test 0.0 <= c.index <= 100.0
    end

    @testset "compute_connectedness — net spillovers sum to zero" begin
        c = compute_connectedness(v3.Phi, v3.Sigma, 12)
        @test sum(c.net) ≈ 0.0 atol = 1e-10
    end

    @testset "compute_connectedness — normalized rows sum to 100%" begin
        c = compute_connectedness(v3.Phi, v3.Sigma, 12)
        # Reconstruct normalized theta from raw theta
        Theta = c.theta ./ sum(c.theta, dims = 2)
        for i in 1:3
            @test sum(Theta[i, :]) ≈ 1.0 atol = 1e-12
        end
    end

    @testset "compute_connectedness — directional measures non-negative" begin
        c = compute_connectedness(v3.Phi, v3.Sigma, 12)
        @test all(c.from_all_to_unit .>= -1e-10)
        @test all(c.from_unit_to_all .>= -1e-10)
    end

    @testset "compute_connectedness — multiple horizons give valid results" begin
        for h in [1, 6, 12, 24]
            c = compute_connectedness(v3.Phi, v3.Sigma, h)
            @test 0.0 <= c.index <= 100.0
            @test sum(c.net) ≈ 0.0 atol = 1e-10
        end
    end

    @testset "compute_connectedness — different horizons produce different results" begin
        c1 = compute_connectedness(v3.Phi, v3.Sigma, 1)
        c12 = compute_connectedness(v3.Phi, v3.Sigma, 12)
        @test !(c1.index ≈ c12.index)
    end

    @testset "compute_connectedness — 2-variable system" begin
        y2, _, _ = generate_var_data(200, 2, 1)
        v2 = var_estimate(y2, 1; constant = true)
        c = compute_connectedness(v2.Phi, v2.Sigma, 12)
        @test 0.0 <= c.index <= 100.0
        @test sum(c.net) ≈ 0.0 atol = 1e-10
        @test length(c.from_all_to_unit) == 2
    end

    @testset "compute_connectedness — with custom Omega (Cholesky)" begin
        A = cholesky(Hermitian(v3.Sigma)).L
        c = compute_connectedness(v3.Phi, v3.Sigma, 12; Omega = A)
        @test 0.0 <= c.index <= 100.0
        @test sum(c.net) ≈ 0.0 atol = 1e-10
    end

    @testset "connectedness_posterior — band ordering" begin
        y2, _, _ = generate_var_data(200, 2, 1)
        result = bvar(y2, 1; prior = FlatPrior(), K = 100, hor = 6, fhor = 4,
            verbose = false)
        cp = connectedness_posterior(result; horizon = 6, conf_level = 0.90)
        @test cp.lower <= cp.median + 1e-10
        @test cp.median <= cp.upper + 1e-10
    end

    @testset "connectedness_posterior — all draws in [0, 100]" begin
        y2, _, _ = generate_var_data(200, 2, 1)
        result = bvar(y2, 1; prior = FlatPrior(), K = 100, hor = 6, fhor = 4,
            verbose = false)
        cp = connectedness_posterior(result; horizon = 6)
        @test all(0.0 .<= cp.draws .<= 100.0)
    end

    @testset "connectedness_posterior — median is finite" begin
        y2, _, _ = generate_var_data(200, 2, 1)
        result = bvar(y2, 1; prior = FlatPrior(), K = 50, hor = 6, fhor = 4,
            verbose = false)
        cp = connectedness_posterior(result; horizon = 6)
        @test isfinite(cp.median)
        @test isfinite(cp.lower)
        @test isfinite(cp.upper)
    end
end
