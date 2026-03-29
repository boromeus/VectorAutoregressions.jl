using VectorAutoregressions
using Test
using LinearAlgebra
using Random

include("test_helpers.jl")

@testset "IRF" begin
    y, _, _ = generate_var_data(500, 3, 1)
    v = var_estimate(y, 1; constant = true)

    @testset "compute_irf (Cholesky)" begin
        ir = compute_irf(v.Phi, v.Sigma, 12)
        @test size(ir) == (3, 12, 3)
        # Impact: lower triangular (Cholesky)
        @test ir[1, 1, 1] > 0  # own shock positive
    end

    @testset "compute_irf_longrun" begin
        ir, Q = compute_irf_longrun(v.Phi, v.Sigma, 12, 1)
        @test size(ir) == (3, 12, 3)
    end
end
