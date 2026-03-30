using VectorAutoregressions
using Test
using LinearAlgebra
using Random

include("test_helpers.jl")

@testset "Exogenous Block (B6)" begin

    # Generate y (endogenous, 2 vars) and z (exogenous, 1 var)
    rng_data = Random.MersenneTwister(42)
    T = 200
    z = cumsum(randn(rng_data, T, 1), dims=1) * 0.1  # exogenous random walk
    y, _, _ = generate_var_data(T, 2, 1)
    # Add some z-dependence to y
    y[2:end, 1] .+= 0.3 * z[1:end-1, 1]

    @testset "bvar with exogenous_block runs" begin
        result = bvar(y, 1; prior=FlatPrior(), K=50, hor=6, fhor=4,
                       exogenous_block=z, verbose=false)
        @test all(isfinite.(result.Phi_draws))
        @test all(isfinite.(result.Sigma_draws))
    end

    @testset "combined dimensions" begin
        result = bvar(y, 1; prior=FlatPrior(), K=50, hor=6, fhor=4,
                       exogenous_block=z, verbose=false)
        ny_total = 2 + 1  # 2 endogenous + 1 exogenous
        @test result.nvar == ny_total
        @test size(result.Sigma_draws, 1) == ny_total
        @test size(result.Sigma_draws, 2) == ny_total
    end

    @testset "block structure — Phi zeros" begin
        result = bvar(y, 1; prior=FlatPrior(), K=50, hor=6, fhor=4,
                       exogenous_block=z, verbose=false)
        ny, nz_var = 2, 1
        # In combined Phi: y-lag block is rows 1:ny*p, columns ny+1:end should be zero
        # (z does not depend on y-lags)
        for k in 1:50
            Phi_k = result.Phi_draws[:, :, k]
            zPhiy = Phi_k[1:ny*1, ny+1:end]  # y-lags → z equation
            @test all(zPhiy .== 0.0)
        end
    end

    @testset "block-diagonal Sigma" begin
        result = bvar(y, 1; prior=FlatPrior(), K=50, hor=6, fhor=4,
                       exogenous_block=z, verbose=false)
        ny, nz_var = 2, 1
        # Cross-block covariance should be zero
        for k in 1:50
            S = result.Sigma_draws[:, :, k]
            @test all(S[1:ny, ny+1:end] .== 0.0)
            @test all(S[ny+1:end, 1:ny] .== 0.0)
        end
    end

    @testset "IRFs — z→y should be zero at impact" begin
        result = bvar(y, 1; prior=FlatPrior(), K=50, hor=6, fhor=4,
                       exogenous_block=z, verbose=false)
        ny = 2
        # IRF of y variables to z shocks should be zero at horizon 0
        # ir_draws[variable, horizon, shock, draw]
        # z shock is shock ny+1, y variables are 1:ny
        for k in 1:50
            for v in 1:ny
                # Impact of z-shock on y at horizon 1 should be zero
                # because Sigma is block-diagonal
                @test result.ir_draws[v, 1, ny+1, k] ≈ 0.0 atol=1e-10
            end
        end
    end

    @testset "IRFs — y→z propagates through lags" begin
        result = bvar(y, 1; prior=FlatPrior(), K=50, hor=6, fhor=4,
                       exogenous_block=z, verbose=false)
        ny = 2
        # z-equation has z-lags coefficients from y-equation
        # IRF of z to y-shock at later horizons may be non-zero
        # because z-lags appear in y-equation, and y shocks affect y which
        # feeds back to z through the z-lag coefficients in y-equation
        # At impact: z does not respond to y shocks (block-diagonal Sigma)
        for k in 1:50
            @test result.ir_draws[3, 1, 1, k] ≈ 0.0 atol=1e-10
        end
    end

    @testset "Minnesota prior with exogenous_block" begin
        result = bvar(y, 1; prior=MinnesotaPrior(), K=50, hor=6, fhor=4,
                       exogenous_block=z, verbose=false)
        @test all(isfinite.(result.Phi_draws))
        @test result.nvar == 3
    end
end
