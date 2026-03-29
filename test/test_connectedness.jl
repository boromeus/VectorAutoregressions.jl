using VectorAutoregressions
using Test
using LinearAlgebra
using Random

include("test_helpers.jl")

@testset "Connectedness" begin
    y, _, _ = generate_var_data(200, 3, 1)
    v = var_estimate(y, 1; constant = true)

    c = compute_connectedness(v.Phi, v.Sigma, 12)
    @test c isa ConnectednessResult
    @test isfinite(c.index)
    @test length(c.from_all_to_unit) == 3
    @test length(c.net) == 3
end
