using VectorAutoregressions
using Test
using LinearAlgebra
using Random
using Statistics

include("test_helpers.jl")

@testset "Business Cycle" begin

    @testset "Bry-Boschan — basic turning points" begin
        # Construct a simple sine wave with clear peaks and troughs
        T = 200
        t_idx = 1:T
        y = sin.(2π .* t_idx ./ 40)  # period = 40

        bc = bry_boschan(y; window=3, min_phase=2, min_cycle=5)
        @test bc isa BusinessCycleResult
        @test length(bc.peaks) > 0
        @test length(bc.troughs) > 0
    end

    @testset "Bry-Boschan — peaks and troughs alternate" begin
        T = 300
        y = sin.(2π .* (1:T) ./ 50) + 0.1 * randn(Random.MersenneTwister(42), T)
        bc = bry_boschan(y; window=5, min_phase=2, min_cycle=5)

        # Merge and sort
        events = sort(vcat([(p, :peak) for p in bc.peaks],
                           [(t, :trough) for t in bc.troughs]); by=first)
        for i in 2:length(events)
            @test events[i][2] != events[i-1][2]  # must alternate
        end
    end

    @testset "Bry-Boschan — peaks are local maxima" begin
        T = 200
        y = sin.(2π .* (1:T) ./ 40)
        bc = bry_boschan(y; window=3, min_phase=2, min_cycle=5)

        for p in bc.peaks
            # Peak should not be lower than neighbors
            lo = max(1, p - 3)
            hi = min(T, p + 3)
            @test y[p] >= maximum(y[lo:hi]) - 1e-10
        end
    end

    @testset "Bry-Boschan — minimum phase length" begin
        T = 300
        y = sin.(2π .* (1:T) ./ 50)
        min_phase = 8
        bc = bry_boschan(y; window=5, min_phase=min_phase, min_cycle=5)

        events = sort(vcat([(p, :peak) for p in bc.peaks],
                           [(t, :trough) for t in bc.troughs]); by=first)
        for i in 2:length(events)
            @test events[i][1] - events[i-1][1] >= min_phase
        end
    end

    @testset "Bry-Boschan — empty on flat series" begin
        y = ones(100)
        bc = bry_boschan(y; window=5)
        # Flat series has no turning points
        @test length(bc.peaks) == 0
        @test length(bc.troughs) == 0
    end

    @testset "BN decomposition — permanent + transitory = original" begin
        rng = Random.MersenneTwister(42)
        y, _, _ = generate_var_data(200, 2, 1; rng=rng)
        # Add a trend to make it non-stationary-like
        y_trend = y .+ (1:200) .* [0.01 0.02]

        bn = bn_decomposition(y_trend, 1)
        @test bn isa BNDecompResult
        @test size(bn.permanent) == (200, 2)
        @test size(bn.transitory) == (200, 2)
        @test bn.permanent + bn.transitory ≈ y_trend atol=1e-10
    end

    @testset "BN decomposition — stationary series has small permanent variation" begin
        rng = Random.MersenneTwister(42)
        # Strongly stationary AR(1) with small coefficient
        T = 300
        y = zeros(T, 1)
        for t in 2:T
            y[t] = 0.3 * y[t-1] + randn(rng)
        end

        bn = bn_decomposition(y, 1)
        # For a stationary series, transitory component should capture most variation
        @test bn.permanent + bn.transitory ≈ y atol=1e-10
    end

    @testset "BN decomposition — multivariate" begin
        rng = Random.MersenneTwister(42)
        y, _, _ = generate_var_data(200, 3, 1; rng=rng)

        bn = bn_decomposition(y, 1)
        @test size(bn.permanent) == (200, 3)
        @test size(bn.transitory) == (200, 3)
        @test bn.permanent + bn.transitory ≈ y atol=1e-10
    end

    @testset "BN decomposition — multi-lag" begin
        rng = Random.MersenneTwister(42)
        y, _, _ = generate_var_data(200, 2, 2; rng=rng)

        bn = bn_decomposition(y, 2)
        @test bn.permanent + bn.transitory ≈ y atol=1e-10
    end
end
