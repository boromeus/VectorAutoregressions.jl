using VectorAutoregressions
using Test
using Random

@testset "Principal Components" begin
    rng = Random.MersenneTwister(99)
    X = randn(rng, 100, 10)
    pc = principal_components(X, 2)
    @test size(pc.factors) == (100, 2)
    @test size(pc.loadings) == (10, 2)
    @test length(pc.eigenvalues) == 100
end
