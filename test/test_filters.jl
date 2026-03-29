using VectorAutoregressions
using Test
using Random

@testset "Filters" begin
    rng = Random.MersenneTwister(42)
    x = cumsum(randn(rng, 200))

    @testset "HP filter" begin
        r = hp_filter(x, 1600)
        @test length(r.trend) == 200
        @test length(r.cycle) == 200
        @test r.trend .+ r.cycle ≈ x atol = 1e-10
    end

    @testset "Hamilton filter" begin
        r = hamilton_filter(x, 8, 4)
        @test length(r.trend) == 200
        # First d+h-1 entries should be NaN
        @test isnan(r.cycle[1])
        @test !isnan(r.cycle[end])
    end

    @testset "BK filter" begin
        fX = bk_filter(x, 6, 32)
        @test length(fX) == 200
    end
end
