using VectorAutoregressions
using Test
using LinearAlgebra
using Random

include("test_helpers.jl")

@testset "BVAR" begin
    y, _, _ = generate_var_data(200, 2, 1)

    @testset "bvar flat prior" begin
        result = bvar(y, 1; prior = FlatPrior(), K = 50, hor = 6, fhor = 4,
            verbose = false)
        @test result.ndraws == 50
        @test result.nvar == 2
        @test size(result.Phi_draws) == (3, 2, 50)  # [K*p+const, K, ndraws]
        @test size(result.Sigma_draws) == (2, 2, 50)
        @test size(result.ir_draws, 2) == 6  # hor
    end

    @testset "bvar Minnesota prior" begin
        result = bvar(y, 1; prior = MinnesotaPrior(), K = 50, hor = 6, fhor = 4,
            verbose = false)
        @test result.ndraws == 50
        @test result.prior isa MinnesotaPrior
    end
end
