using VectorAutoregressions
using Test
using LinearAlgebra
using Random

include("test_helpers.jl")

@testset "Priors" begin
    y, _, _ = generate_var_data(200, 3, 1)

    @testset "get_prior_moments" begin
        mu, sig, delta = get_prior_moments(y, 1)
        @test length(mu) == 3
        @test length(sig) == 3
        @test all(sig .> 0)
        @test length(delta) == 3
    end

    @testset "build_dummy_observations" begin
        mu, sig, delta = get_prior_moments(y, 1)
        prior = MinnesotaPrior()
        Yd, Xd, _ = build_dummy_observations(prior, 3, 1, sig, delta, vec(mu))
        @test size(Yd, 2) == 3
        @test size(Xd, 2) == 1  # exogenous part only (constant)
    end
end
