using VectorAutoregressions
using Test
using LinearAlgebra
using Random

include("test_helpers.jl")

@testset "FEVD" begin
    y, _, _ = generate_var_data(200, 2, 1)
    v = var_estimate(y, 1; constant = true)

    @testset "compute_fevd" begin
        fevd = compute_fevd(v.Phi, v.Sigma, 12)
        @test size(fevd.decomposition) == (2, 2)
        # Rows should sum to 100 (percentages)
        for i in 1:2
            @test sum(fevd.decomposition[i, :]) ≈ 100.0 atol = 1.0
        end
    end
end
