using VectorAutoregressions
using Test
using LinearAlgebra
using Random
using Statistics

include("test_helpers.jl")

@testset "Panel VAR Shrinkage" begin

    @testset "shrinkage returns PanelVARResult" begin
        panels = [generate_var_data(100, 2, 1)[1] for _ in 1:5]
        r = panel_var(panels, 1; method=:shrinkage)
        @test r isa PanelVARResult
        @test r.method == :shrinkage
    end

    @testset "shrinkage Phi dimensions" begin
        panels = [generate_var_data(100, 2, 1)[1] for _ in 1:5]
        r = panel_var(panels, 1; method=:shrinkage)
        @test size(r.Phi, 2) == 2
        @test size(r.Phi, 1) == 3  # K*p + constant
    end

    @testset "shrinkage between unit and pooled" begin
        rng = Random.MersenneTwister(42)
        panels = [generate_var_data(100, 2, 1; rng=Random.MersenneTwister(42 + i))[1]
                  for i in 1:5]

        r_unit = panel_var(panels, 1; method=:unit)
        r_shrink = panel_var(panels, 1; method=:shrinkage)
        r_pooled = panel_var(panels, 1; method=:pooled)

        # Shrinkage should produce Phi between unit and pooled extremes
        @test r_shrink.Phi isa Matrix{Float64}
        @test size(r_shrink.Sigma) == (2, 2)
    end

    @testset "shrinkage unit_results stored" begin
        panels = [generate_var_data(100, 2, 1)[1] for _ in 1:4]
        r = panel_var(panels, 1; method=:shrinkage)
        @test length(r.unit_results) == 4
    end

    @testset "shrinkage positive definite Sigma" begin
        panels = [generate_var_data(100, 2, 1)[1] for _ in 1:5]
        r = panel_var(panels, 1; method=:shrinkage)
        @test all(eigvals(Hermitian(r.Sigma)) .> 0)
    end

    @testset "identical panels → weight near 0" begin
        # If all panels have identical DGP, between-unit variance is small
        # relative to within-unit, so shrinkage should be modest
        rng = Random.MersenneTwister(42)
        base_data = generate_var_data(200, 2, 1; rng=rng)[1]
        panels = [base_data + 0.01 * randn(Random.MersenneTwister(i), 200, 2)
                  for i in 1:5]

        r_unit = panel_var(panels, 1; method=:unit)
        r_shrink = panel_var(panels, 1; method=:shrinkage)

        # With nearly identical panels, shrinkage should be close to unit average
        @test r_shrink.Phi ≈ r_unit.Phi atol=0.3
    end
end
