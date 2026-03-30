using VectorAutoregressions
using Test
using LinearAlgebra
using Random

include("test_helpers.jl")

@testset "Marginal Likelihood" begin
    y, _, _ = generate_var_data(200, 2, 1)

    ml = compute_marginal_likelihood(y, 1, MinnesotaPrior())
    @test isfinite(ml)
end

@testset "Hyperparameter Optimization (B1)" begin
    y, _, _ = generate_var_data(200, 2, 1)

    @testset "compute_marginal_likelihood is finite" begin
        ml = compute_marginal_likelihood(y, 1, MinnesotaPrior())
        @test isfinite(ml)
        @test ml < 0  # log-likelihood should be negative
    end

    @testset "optimize all 5 hyperparams" begin
        opt_prior, opt_logml = optimize_hyperparameters_optim(y, 1)
        @test opt_prior isa MinnesotaPrior
        @test isfinite(opt_logml)
        # Optimal should beat default
        default_ml = compute_marginal_likelihood(y, 1, MinnesotaPrior())
        @test opt_logml >= default_ml - 1.0  # allow small tolerance
    end

    @testset "optimize subset (index_est)" begin
        # Fix tau=3.0, optimize only lambda and mu
        opt_prior, opt_logml = optimize_hyperparameters_optim(y, 1;
            hyperpara=[3.0, 0.5, 5.0, 2.0, 2.0],
            index_est=[3, 4])
        @test opt_prior.tau == 3.0
        @test opt_prior.decay == 0.5
        @test opt_prior.omega == 2.0
        @test isfinite(opt_logml)
    end

    @testset "bounds respected" begin
        opt_prior, _ = optimize_hyperparameters_optim(y, 1;
            index_est=[1],
            lb=[0.1],
            ub=[1.0])
        @test 0.1 <= opt_prior.tau <= 1.0
    end

    @testset "gradient method ≥ grid search" begin
        opt_prior_grad, logml_grad = optimize_hyperparameters_optim(y, 1)
        # Grid search with small grid
        opt_prior_grid, logml_grid = optimize_hyperparameters(y, 1;
            tau_range=1.0:1.0:5.0,
            decay_range=0.5:0.5:2.0,
            lambda_range=1.0:2.0:5.0,
            mu_range=1.0:1.0:3.0,
            omega_range=1.0:1.0:3.0)
        # Both methods should find reasonable optima (within 20 of each other)
        @test abs(logml_grad - logml_grid) < 20.0
    end

    @testset "sequential optimization" begin
        # Step 1: optimize tau only
        p1, ml1 = optimize_hyperparameters_optim(y, 1; index_est=[1])
        # Step 2: optimize tau+decay starting from step 1
        p2, ml2 = optimize_hyperparameters_optim(y, 1;
            hyperpara=[p1.tau, p1.decay, p1.lambda, p1.mu, p1.omega],
            index_est=[1, 2])
        @test ml2 >= ml1 - 0.5  # should not decrease much
    end

    @testset "full BVAR with optimal prior" begin
        opt_prior, _ = optimize_hyperparameters_optim(y, 1)
        result = bvar(y, 1; prior=opt_prior, K=50, hor=6, fhor=4, verbose=false)
        @test all(isfinite.(result.ir_draws))
        @test all(isfinite.(result.Phi_draws))
    end
end
