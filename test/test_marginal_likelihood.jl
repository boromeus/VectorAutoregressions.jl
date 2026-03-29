using VectorAutoregressions
using Test
using LinearAlgebra
using Random

include("test_helpers.jl")

@testset "Marginal Likelihood" begin
    y, _, _ = generate_var_data(200, 2, 1)

    ml = compute_marginal_likelihood(y, 1, MinnesotaPrior())
    @test isfinite(ml)
end
