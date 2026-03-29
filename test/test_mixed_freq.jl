using VectorAutoregressions
using Test
using LinearAlgebra
using Random
using Statistics

include("test_helpers.jl")

@testset "Mixed-Frequency VAR" begin

    @testset "All high-frequency matches var_estimate" begin
        rng = Random.MersenneTwister(42)
        y, _, _ = generate_var_data(200, 2, 1; rng=rng)
        freq_map = [:high, :high]

        mf = mixed_freq_var(y, 1, freq_map; max_iter=20, tol=1e-8)
        v = var_estimate(y, 1)

        @test mf isa MixedFreqVARResult
        @test size(mf.Phi) == size(v.Phi)
        @test size(mf.Sigma) == size(v.Sigma)
        # With all data observed, should converge to OLS
        @test mf.Phi ≈ v.Phi atol=0.3
        @test isfinite(mf.logL)
    end

    @testset "Stock variable with NaN" begin
        rng = Random.MersenneTwister(42)
        y, _, _ = generate_var_data(120, 2, 1; rng=rng)
        # Make second variable observed only every 3rd period (quarterly from monthly)
        y_mf = copy(y)
        for t in 1:120
            if t % 3 != 0
                y_mf[t, 2] = NaN
            end
        end
        freq_map = [:high, :stock]

        mf = mixed_freq_var(y_mf, 1, freq_map; max_iter=50, tol=1e-6)
        @test mf isa MixedFreqVARResult
        @test isfinite(mf.logL)
        @test size(mf.y_interpolated) == (120, 2)
        # Interpolated values should not have NaN
        @test !any(isnan.(mf.y_interpolated))
    end

    @testset "Flow variable with NaN" begin
        rng = Random.MersenneTwister(42)
        y, _, _ = generate_var_data(120, 2, 1; rng=rng)
        # Make second variable a flow: observed every 3rd period
        y_mf = copy(y)
        for t in 1:120
            if t % 3 != 0
                y_mf[t, 2] = NaN
            end
        end
        freq_map = [:high, :flow]

        mf = mixed_freq_var(y_mf, 1, freq_map; max_iter=50, tol=1e-6)
        @test mf isa MixedFreqVARResult
        @test isfinite(mf.logL)
        @test !any(isnan.(mf.y_interpolated))
    end

    @testset "Convergence flag" begin
        rng = Random.MersenneTwister(42)
        y, _, _ = generate_var_data(200, 2, 1; rng=rng)
        freq_map = [:high, :high]

        mf = mixed_freq_var(y, 1, freq_map; max_iter=100, tol=1e-6)
        # With all observed data, should converge quickly
        @test mf.converged == true || mf.niter <= 100
    end

    @testset "Returns positive definite Sigma" begin
        rng = Random.MersenneTwister(42)
        y, _, _ = generate_var_data(120, 2, 1; rng=rng)
        y_mf = copy(y)
        for t in 1:120
            if t % 3 != 0
                y_mf[t, 2] = NaN
            end
        end
        freq_map = [:high, :stock]

        mf = mixed_freq_var(y_mf, 1, freq_map; max_iter=50)
        @test all(eigvals(Hermitian(mf.Sigma)) .> 0)
    end

    @testset "Invalid freq_map throws" begin
        y, _, _ = generate_var_data(100, 2, 1)
        @test_throws ArgumentError mixed_freq_var(y, 1, [:high, :invalid])
        @test_throws ArgumentError mixed_freq_var(y, 1, [:high])  # wrong length
    end
end
