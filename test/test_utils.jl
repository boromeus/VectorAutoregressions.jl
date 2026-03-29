using VectorAutoregressions
using Test
using LinearAlgebra
using Random

@testset "Utilities" begin
    @testset "lagmatrix" begin
        y = [1.0 2.0; 3.0 4.0; 5.0 6.0; 7.0 8.0]
        X = lagmatrix(y, 1)
        @test size(X) == (3, 2)
        @test X[1, :] == [1.0, 2.0]

        X2 = lagmatrix(y, 1; constant = true)
        @test size(X2) == (3, 3)
        @test X2[:, 3] == ones(3)

        X3 = lagmatrix(y, 2; constant = true, trend = true)
        @test size(X3) == (2, 6)  # 2*2 lags + const + trend
    end

    @testset "companion_form" begin
        K, p = 2, 2
        Phi = [0.5 0.1; 0.0 0.4; 0.1 0.0; 0.0 0.1]
        F = companion_form(Phi, K, p)
        @test size(F) == (4, 4)
        @test F[3:4, 1:2] == I(2)
    end

    @testset "check_stability" begin
        Phi_stable = [0.3 0.0; 0.0 0.3]
        @test check_stability(Phi_stable, 2, 1) == true
        Phi_unstable = [1.1 0.0; 0.0 1.1]
        @test check_stability(Phi_unstable, 2, 1) == false
    end

    @testset "rand_inverse_wishart" begin
        rng = Random.MersenneTwister(123)
        S = Matrix{Float64}(I, 3, 3)
        W = rand_inverse_wishart(10, S; rng = rng)
        @test size(W) == (3, 3)
        @test issymmetric(round.(W; digits = 10))
        @test all(eigvals(Hermitian(W)) .> 0)
    end

    @testset "vech_ivech" begin
        A = [1.0 2.0 3.0; 2.0 4.0 5.0; 3.0 5.0 6.0]
        v = vech(A)
        @test length(v) == 6
        A2 = ivech(v, 3)
        @test A2 == A
    end

    @testset "var2ma" begin
        Phi = [0.5 0.0; 0.0 0.3]
        Psi = var2ma(Phi, 3)
        @test size(Psi) == (2, 2, 3)
        @test Psi[:, :, 1] ≈ I(2)
    end

    @testset "ols_svd" begin
        X = randn(Random.MersenneTwister(1), 100, 3)
        beta_true = [1.0, 2.0, 3.0]
        y = X * beta_true + 0.01 * randn(Random.MersenneTwister(2), 100, 1)
        Phi, resid, xxi = ols_svd(y, X)
        @test vec(Phi) ≈ beta_true atol = 0.1
    end

    @testset "generate_rotation_matrix" begin
        rng = Random.MersenneTwister(42)
        Q = generate_rotation_matrix(3; rng = rng)
        @test size(Q) == (3, 3)
        @test Q' * Q ≈ I(3) atol = 1e-10
    end

    @testset "matrix_operations" begin
        K = commutation_matrix(2, 3)
        @test size(K) == (6, 6)
        D = duplication_matrix(3)
        @test size(D) == (9, 6)
        L = elimination_matrix(3)
        @test size(L) == (6, 9)
    end
end
