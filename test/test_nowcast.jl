using VectorAutoregressions
using Test
using LinearAlgebra
using Random

include("test_helpers.jl")

@testset "Nowcasting (B5)" begin

    y, _, _ = generate_var_data(200, 2, 1)

    @testset "nowcast_bvar — full data" begin
        result = bvar(y, 1; prior=FlatPrior(), K=30, hor=6, fhor=4, verbose=false)
        # No NaN → nowcast should approximate the data
        data_panels = reshape(y, size(y, 1), size(y, 2), 1)
        nc = nowcast_bvar(data_panels, result; n_draws=5)
        @test size(nc.nowcast) == (200, 5, 1)
        @test all(isfinite.(nc.nowcast))
    end

    @testset "nowcast_bvar — progressive data release" begin
        result = bvar(y, 1; prior=FlatPrior(), K=30, hor=6, fhor=4, verbose=false)

        # Panel 1: lots of NaN (early release)
        panel1 = copy(y)
        panel1[180:200, 2] .= NaN  # variable 2 missing at end

        # Panel 2: fewer NaN (later release)
        panel2 = copy(y)
        panel2[195:200, 2] .= NaN

        # Panel 3: complete data
        panel3 = copy(y)

        data_panels = cat(panel1, panel2, panel3; dims=3)
        nc = nowcast_bvar(data_panels, result; n_draws=10)
        @test size(nc.nowcast) == (200, 10, 3)
        @test all(isfinite.(nc.nowcast))
    end

    @testset "nowcast_bvar — draw indices valid" begin
        result = bvar(y, 1; prior=FlatPrior(), K=30, hor=6, fhor=4, verbose=false)
        data_panels = reshape(y, size(y, 1), size(y, 2), 1)
        nc = nowcast_bvar(data_panels, result; n_draws=10)
        @test all(1 .<= nc.draw_indices .<= 30)
    end
end
