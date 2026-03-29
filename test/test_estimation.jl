using VectorAutoregressions
using Test
using LinearAlgebra
using Random

include("test_helpers.jl")

@testset "VAR Estimation" begin
    y, Phi_true, Sigma_true = generate_var_data(500, 3, 1)

    @testset "var_estimate" begin
        v = var_estimate(y, 1; constant = true)
        @test v.nvar == 3
        @test v.nlags == 1
        @test v.nobs == 499
        @test size(v.Phi) == (4, 3)  # 3 lags + constant
        @test size(v.Sigma) == (3, 3)
        @test size(v.residuals) == (499, 3)
        @test issymmetric(round.(v.Sigma; digits = 10))

        # AR coefficients should be close to true values
        @test v.Phi[1:3, :] ≈ Phi_true atol = 0.15
    end

    @testset "var_lagorder" begin
        lag = var_lagorder(y, 8; ic = "bic", verbose = false)
        @test lag isa Int
        @test 1 <= lag <= 8
    end

    @testset "information_criteria" begin
        v = var_estimate(y, 1; constant = true)
        ic = information_criteria(v)
        @test ic isa InfoCriteria
        @test isfinite(ic.aic)
        @test isfinite(ic.bic)
        @test isfinite(ic.hqic)
    end
end
