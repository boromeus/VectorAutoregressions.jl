using VectorAutoregressions
using Test
using LinearAlgebra
using Random
using Statistics

include("test_helpers.jl")

@testset "LP Extensions" begin

    @testset "LP-IV" begin
        rng = Random.MersenneTwister(42)
        y, _, _ = generate_var_data(300, 2, 1; rng=rng)
        # Create a proxy that's correlated with the first structural shock
        v = var_estimate(y, 4)
        proxy = v.residuals[:, 1:1] + 0.5 * randn(rng, size(v.residuals, 1), 1)
        # Pad proxy to match y length (need T total)
        proxy_full = vcat(zeros(4, 1), proxy)  # p=4 lag pre-sample

        @testset "LP-IV returns LPResult" begin
            r = lp_irf(y, 4, 8; identification=:iv, proxy=proxy_full,
                       conf_level=0.90)
            @test r isa LPResult
            @test size(r.irf) == (9, 4)  # (H+1) × K²
            @test r.horizon == 8
        end

        @testset "LP-IV has finite values" begin
            r = lp_irf(y, 4, 8; identification=:iv, proxy=proxy_full,
                       conf_level=0.90)
            @test all(isfinite.(r.irf))
            @test all(isfinite.(r.lower))
            @test all(isfinite.(r.upper))
        end

        @testset "LP-IV confidence bands contain point estimates" begin
            r = lp_irf(y, 4, 8; identification=:iv, proxy=proxy_full,
                       conf_level=0.90)
            for h in 1:9, k in 1:4
                @test r.lower[h, k] <= r.irf[h, k] + 1e-10
                @test r.irf[h, k] <= r.upper[h, k] + 1e-10
            end
        end
    end

    @testset "LP-Bayesian" begin
        rng = Random.MersenneTwister(42)
        y, _, _ = generate_var_data(200, 2, 1; rng=rng)

        @testset "LP-Bayesian returns correct type" begin
            r = lp_bayesian(y, 2, 8; prior=MinnesotaPrior(), K=50,
                            conf_level=0.90, rng=rng)
            @test r isa LPBayesianResult
            @test r.horizon == 8
            @test r.ndraws == 50
            @test r.conf_level == 0.90
        end

        @testset "LP-Bayesian correct dimensions" begin
            r = lp_bayesian(y, 2, 8; prior=MinnesotaPrior(), K=50,
                            rng=rng)
            @test size(r.irf) == (2, 9, 2)  # K × (H+1) × K
            @test size(r.lower) == (2, 9, 2)
            @test size(r.upper) == (2, 9, 2)
        end

        @testset "LP-Bayesian bands ordered" begin
            r = lp_bayesian(y, 2, 8; prior=MinnesotaPrior(), K=100,
                            conf_level=0.90, rng=rng)
            for k in 1:2, h in 1:9, j in 1:2
                @test r.lower[k, h, j] <= r.upper[k, h, j] + 1e-10
            end
        end

        @testset "LP-Bayesian with flat prior" begin
            r = lp_bayesian(y, 2, 6; prior=FlatPrior(), K=50, rng=rng)
            @test r isa LPBayesianResult
            @test all(isfinite.(r.irf))
        end

        @testset "LP-Bayesian Phi draws stored" begin
            r = lp_bayesian(y, 2, 4; prior=MinnesotaPrior(), K=30, rng=rng)
            nk = 2 * 2 + 1  # K*p + constant
            @test size(r.Phi_draws) == (nk, 2, 5, 30)  # nk × K × (H+1) × ndraws
            @test size(r.Sigma_draws) == (2, 2, 5, 30)
        end
    end

    @testset "LP marginal likelihood" begin
        rng = Random.MersenneTwister(42)
        y, _, _ = generate_var_data(200, 2, 1; rng=rng)

        ml = lp_marginal_likelihood(y, 2, 4, MinnesotaPrior())
        @test isfinite(ml)
    end
end
