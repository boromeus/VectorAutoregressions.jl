using VectorAutoregressions
using Test
using LinearAlgebra
using Random
using Statistics

include("test_helpers.jl")

@testset "FAVAR BBE Rotation" begin

    @testset "bbe_rotation orthogonality" begin
        rng = Random.MersenneTwister(42)
        T = 200
        y_slow = randn(rng, T, 2)
        y_fast = randn(rng, T, 10) + y_slow * randn(rng, 2, 10) * 0.5

        pc = principal_components(y_fast, 3)
        F_raw = pc.factors

        F_rot = bbe_rotation(F_raw, y_slow)

        @test size(F_rot) == (T, 3)

        # Rotated factors should be orthogonal to policy variables
        corr_mat = y_slow' * F_rot / T
        @test maximum(abs.(corr_mat)) < 0.15  # should be approximately zero
    end

    @testset "bbe_rotation return dimensions" begin
        rng = Random.MersenneTwister(42)
        T = 100
        F_raw = randn(rng, T, 4)
        y_pol = randn(rng, T, 2)

        F_rot = bbe_rotation(F_raw, y_pol)
        @test size(F_rot) == (T, 4)
    end

    @testset "favar with rotation=:none" begin
        rng = Random.MersenneTwister(42)
        T = 120
        y_slow = randn(rng, T, 1)
        y_fast = randn(rng, T, 8)

        r = favar(y_slow, y_fast, 2, 1; K=20, hor=4, burnin=50, skip=1,
                  rotation=:none, rng=rng)
        @test r isa FAVARResult
        @test r.nfactors == 2
        @test r.ndraws == 20
        @test size(r.factors) == (T, 2)
    end

    @testset "favar with rotation=:bbe" begin
        rng = Random.MersenneTwister(42)
        T = 120
        y_slow = randn(rng, T, 1)
        y_fast = randn(rng, T, 8)

        r = favar(y_slow, y_fast, 2, 1; K=20, hor=4, burnin=50, skip=1,
                  rotation=:bbe, rng=rng)
        @test r isa FAVARResult
        @test r.nfactors == 2
        @test r.ndraws == 20
    end

    @testset "favar posterior draw dimensions" begin
        rng = Random.MersenneTwister(42)
        T = 120
        y_slow = randn(rng, T, 1)
        y_fast = randn(rng, T, 8)
        nfac = 2
        p = 1
        ny_var = 1 + nfac  # n_slow + n_factors
        nk = ny_var * p + 1  # + constant

        r = favar(y_slow, y_fast, nfac, p; K=20, hor=4, burnin=50,
                  skip=1, rng=rng)
        @test size(r.Phi_draws) == (nk, ny_var, 20)
        @test size(r.Sigma_draws) == (ny_var, ny_var, 20)
        @test size(r.ir_draws) == (ny_var, 5, ny_var, 20)  # hor+1=5
    end

    @testset "favar loadings updated" begin
        rng = Random.MersenneTwister(42)
        T = 120
        y_slow = randn(rng, T, 1)
        y_fast = randn(rng, T, 8)

        r = favar(y_slow, y_fast, 2, 1; K=20, hor=4, burnin=100,
                  skip=1, rng=rng)
        # Loading matrix should have correct dimensions: n_fast × nfac
        @test size(r.loadings) == (8, 2)
    end
end
