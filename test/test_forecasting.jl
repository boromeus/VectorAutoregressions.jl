using VectorAutoregressions
using Test
using LinearAlgebra
using Random

include("test_helpers.jl")

@testset "Forecasting" begin
    y, _, _ = generate_var_data(200, 2, 1)
    v = var_estimate(y, 1; constant = true)

    @testset "forecast_unconditional" begin
        initval = y[end:end, :]
        xdata = ones(8, 1)  # constant
        fno, fwith = forecast_unconditional(initval, xdata,
            v.Phi, v.Sigma, 8, 1)
        @test size(fno) == (8, 2)
        @test size(fwith) == (8, 2)
    end
end
