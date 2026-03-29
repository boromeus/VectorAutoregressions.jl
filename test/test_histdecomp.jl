using VectorAutoregressions
using Test
using LinearAlgebra
using Random
using Statistics

include("test_helpers.jl")

@testset "Historical Decomposition" begin

    # ── Baseline: 2-variable, 1 lag, flat prior ─────────────────────────────
    y2, _, _ = generate_var_data(200, 2, 1)
    result2 = bvar(y2, 1; prior = FlatPrior(), K = 50, hor = 6, fhor = 4,
        verbose = false)

    @testset "basic — type and dimensions" begin
        hd = historical_decomposition(result2)
        @test hd isa HistDecompResult
        K = result2.nvar
        p = result2.nlags
        Tu = size(result2.e_draws, 1)
        # decomposition: T × K × (K + 1)  [K shocks + deterministic]
        @test size(hd.decomposition) == (Tu, K, K + 1)
        @test size(hd.structural_shocks) == (Tu, K)
    end

    @testset "key invariant — decomposition sums to observed data" begin
        hd = historical_decomposition(result2)
        K = result2.nvar
        p = result2.nlags
        Tu = size(result2.e_draws, 1)
        data = result2.var.data[p + 1:end, :]
        data_check = data[1:Tu, :]

        # Sum across all components (K shocks + deterministic)
        reconstructed = dropdims(sum(hd.decomposition, dims = 3), dims = 3)
        @test reconstructed ≈ data_check atol = 1e-6
    end

    @testset "draw = :mean (default)" begin
        hd = historical_decomposition(result2; draw = :mean)
        @test hd isa HistDecompResult
        Tu = size(result2.e_draws, 1)
        K = result2.nvar
        reconstructed = dropdims(sum(hd.decomposition, dims = 3), dims = 3)
        data = result2.var.data[result2.nlags + 1:end, :]
        @test reconstructed ≈ data[1:Tu, :] atol = 1e-6
    end

    @testset "draw = :median" begin
        hd = historical_decomposition(result2; draw = :median)
        @test hd isa HistDecompResult
        @test size(hd.structural_shocks, 2) == result2.nvar
    end

    @testset "draw = specific integer" begin
        hd = historical_decomposition(result2; draw = 1)
        @test hd isa HistDecompResult
        Tu = size(result2.e_draws, 1)
        K = result2.nvar
        reconstructed = dropdims(sum(hd.decomposition, dims = 3), dims = 3)
        data = result2.var.data[result2.nlags + 1:end, :]
        @test reconstructed ≈ data[1:Tu, :] atol = 1e-6
    end

    @testset "structural shocks are finite" begin
        hd = historical_decomposition(result2)
        @test all(isfinite.(hd.structural_shocks))
        @test all(isfinite.(hd.decomposition))
    end

    @testset "Minnesota prior" begin
        result_mn = bvar(y2, 1; prior = MinnesotaPrior(), K = 50, hor = 6,
            fhor = 4, verbose = false)
        hd = historical_decomposition(result_mn)
        @test hd isa HistDecompResult
        Tu = size(result_mn.e_draws, 1)
        K = result_mn.nvar
        reconstructed = dropdims(sum(hd.decomposition, dims = 3), dims = 3)
        data = result_mn.var.data[result_mn.nlags + 1:end, :]
        @test reconstructed ≈ data[1:Tu, :] atol = 1e-6
    end

    @testset "3-variable system" begin
        y3, _, _ = generate_var_data(200, 3, 1)
        result3 = bvar(y3, 1; prior = FlatPrior(), K = 30, hor = 6, fhor = 4,
            verbose = false)
        hd = historical_decomposition(result3)
        Tu = size(result3.e_draws, 1)
        K = result3.nvar

        @test size(hd.decomposition) == (Tu, K, K + 1)
        reconstructed = dropdims(sum(hd.decomposition, dims = 3), dims = 3)
        data = result3.var.data[result3.nlags + 1:end, :]
        @test reconstructed ≈ data[1:Tu, :] atol = 1e-6
    end
end
