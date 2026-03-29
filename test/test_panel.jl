using VectorAutoregressions
using Test
using LinearAlgebra
using Random

include("test_helpers.jl")

@testset "Panel VAR" begin
    panels = [generate_var_data(100, 2, 1)[1] for _ in 1:3]

    @testset "pooled" begin
        r = panel_var(panels, 1; method = :pooled)
        @test r isa PanelVARResult
        @test r.method == :pooled
        @test size(r.Phi, 2) == 2
    end

    @testset "unit-by-unit" begin
        r = panel_var(panels, 1; method = :unit)
        @test r isa PanelVARResult
        @test r.method == :unit
        @test length(r.unit_results) == 3
    end
end
