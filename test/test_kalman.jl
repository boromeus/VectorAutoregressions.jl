using VectorAutoregressions
using Test
using LinearAlgebra
using Random

include("test_helpers.jl")

@testset "Kalman Filter" begin
    y, _, _ = generate_var_data(100, 2, 1)
    v = var_estimate(y, 1; constant = true)

    result = kalman_filter(v.Phi, v.Sigma, y)
    @test isfinite(result.logL)
    @test size(result.states) == (100, 2)
end
