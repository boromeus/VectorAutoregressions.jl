using VectorAutoregressions
using Test
using LinearAlgebra
using Random

include("test_helpers.jl")

@testset "Historical Decomposition" begin
    y, _, _ = generate_var_data(200, 2, 1)
    result = bvar(y, 1; prior = FlatPrior(), K = 20, hor = 6, fhor = 4,
        verbose = false)

    hd = historical_decomposition(result)
    @test hd isa HistDecompResult
    @test size(hd.structural_shocks, 2) == 2
end
