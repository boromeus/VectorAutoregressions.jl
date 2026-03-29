using VectorAutoregressions
using Test
using LinearAlgebra
using Random

include("test_helpers.jl")

@testset "Local Projections" begin
    y, _, _ = generate_var_data(300, 2, 1)

    @testset "lp_irf" begin
        r = lp_irf(y, 4, 12; conf_level = 0.90)
        @test r isa LPResult
        @test size(r.irf) == (13, 4)  # (H+1) × K²
        @test r.horizon == 12
    end

    @testset "lp_lagorder" begin
        lags = lp_lagorder(y, 8, 12, "bic")
        @test length(lags) == 12
        @test all(1 .<= lags .<= 8)
    end
end
