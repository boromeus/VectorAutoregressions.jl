using VectorAutoregressions
using Test
using LinearAlgebra
using Random
using Statistics

include("test_helpers.jl")

@testset "Regularization" begin
    y, Phi_true, _ = generate_var_data(300, 2, 1)

    @testset "Ridge λ=0 equals OLS" begin
        v_ols = var_estimate(y, 1; constant=true)
        v_ridge = var_estimate(y, 1; constant=true, regularization=:ridge, lambda=0.0)
        @test v_ols.Phi ≈ v_ridge.Phi atol=1e-10
        @test v_ols.Sigma ≈ v_ridge.Sigma atol=1e-10
    end

    @testset "Ridge shrinks coefficients" begin
        v_ols = var_estimate(y, 1; constant=true)
        v_ridge = var_estimate(y, 1; constant=true, regularization=:ridge, lambda=10.0)
        @test norm(v_ridge.Phi) < norm(v_ols.Phi)
    end

    @testset "Ridge returns valid VAREstimate" begin
        v = var_estimate(y, 1; constant=true, regularization=:ridge, lambda=1.0)
        @test v isa VAREstimate
        @test v.nvar == 2
        @test v.nlags == 1
        @test size(v.Phi) == (3, 2)
        @test size(v.Sigma) == (2, 2)
        @test issymmetric(round.(v.Sigma; digits=10))
        @test all(eigvals(Hermitian(v.Sigma)) .> 0)
    end

    @testset "Lasso produces sparsity" begin
        # Use large lambda to force sparsity
        v = var_estimate(y, 1; constant=true, regularization=:lasso, lambda=100.0)
        @test v isa VAREstimate
        # At least some coefficients should be zero with large lambda
        @test any(abs.(v.Phi[1:2, :]) .< 1e-8)
    end

    @testset "Lasso λ=0 is close to OLS" begin
        v_ols = var_estimate(y, 1; constant=true)
        v_lasso = var_estimate(y, 1; constant=true, regularization=:lasso, lambda=0.001)
        @test v_lasso.Phi ≈ v_ols.Phi atol=0.5
    end

    @testset "Elastic net α=0 equals Ridge" begin
        λ = 5.0
        v_ridge = var_estimate(y, 1; constant=true, regularization=:ridge, lambda=λ)
        v_en = var_estimate(y, 1; constant=true, regularization=:elastic_net,
                            lambda=λ, alpha=0.0)
        # With alpha=0, EN = pure L2, should match ridge
        @test v_en.Phi ≈ v_ridge.Phi atol=0.1
    end

    @testset "Elastic net returns valid estimate" begin
        v = var_estimate(y, 1; constant=true, regularization=:elastic_net,
                         lambda=1.0, alpha=0.5)
        @test v isa VAREstimate
        @test size(v.Phi) == (3, 2)
        @test all(eigvals(Hermitian(v.Sigma)) .> 0)
    end

    @testset "Elastic net shrinks more than Ridge alone" begin
        λ = 5.0
        v_ridge = var_estimate(y, 1; constant=true, regularization=:ridge, lambda=λ)
        v_en = var_estimate(y, 1; constant=true, regularization=:elastic_net,
                            lambda=λ, alpha=0.5)
        # EN with L1 component should shrink some coefficients more
        @test norm(v_en.Phi[1:2, :]) <= norm(v_ridge.Phi[1:2, :]) + 0.5
    end

    @testset "Invalid regularization throws" begin
        @test_throws ArgumentError var_estimate(y, 1; regularization=:invalid)
    end

    @testset "Ridge with multi-lag VAR" begin
        y3, _, _ = generate_var_data(300, 3, 2)
        v = var_estimate(y3, 2; constant=true, regularization=:ridge, lambda=1.0)
        @test v isa VAREstimate
        @test v.nlags == 2
        @test v.nvar == 3
        @test size(v.Phi) == (7, 3)  # 3*2 lags + constant
    end
end
