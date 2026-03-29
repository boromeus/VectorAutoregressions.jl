using VectorAutoregressions
using Test
using LinearAlgebra
using Random

include("test_helpers.jl")

@testset "BVAR Identification Integration" begin
    rng = Random.MersenneTwister(42)
    y, _, _ = generate_var_data(200, 2, 1; rng=rng)
    K_var = 2
    p_var = 1
    ndraws = 30
    hor_val = 8

    # ─── Long-Run ───────────────────────────────────────────────────────
    @testset "bvar LongRunIdentification" begin
        result = bvar(y, p_var;
                      prior=FlatPrior(),
                      identification=LongRunIdentification(),
                      K=ndraws, hor=hor_val, fhor=4,
                      verbose=false, rng=Random.MersenneTwister(10))
        @test result.ndraws == ndraws
        @test result.identification isa LongRunIdentification
        @test size(result.irlr_draws) == (K_var, hor_val, K_var, ndraws)
        # At least some long-run IRFs should be non-zero
        @test any(result.irlr_draws .!= 0)
        # Omega draws should be stored
        @test any(result.Omega_draws .!= 0)
    end

    # ─── Sign Restrictions ──────────────────────────────────────────────
    @testset "bvar SignRestriction" begin
        sr = SignRestriction(restrictions=["y(1,1,1)>0", "y(2,1,2)>0"],
                             max_rotations=5000)
        result = bvar(y, p_var;
                      prior=FlatPrior(),
                      identification=sr,
                      K=ndraws, hor=hor_val, fhor=4,
                      verbose=false, rng=Random.MersenneTwister(20))
        @test result.ndraws == ndraws
        @test result.identification isa SignRestriction
        @test size(result.irsign_draws) == (K_var, hor_val, K_var, ndraws)
        # Check that sign-restricted IRFs are populated (at least some non-NaN)
        non_nan_count = count(isfinite, result.irsign_draws)
        @test non_nan_count > 0
    end

    # ─── Narrative + Sign ───────────────────────────────────────────────
    @testset "bvar NarrativeSignRestriction" begin
        nsr = NarrativeSignRestriction(
            signs=["y(1,1,1)>0"],
            narrative=["v(1,1)>0"],
            max_rotations=5000)
        result = bvar(y, p_var;
                      prior=FlatPrior(),
                      identification=nsr,
                      K=ndraws, hor=hor_val, fhor=4,
                      verbose=false, rng=Random.MersenneTwister(30))
        @test result.ndraws == ndraws
        @test result.identification isa NarrativeSignRestriction
        @test size(result.irnarrsign_draws) == (K_var, hor_val, K_var, ndraws)
    end

    # ─── Zero + Sign ───────────────────────────────────────────────────
    @testset "bvar ZeroSignRestriction" begin
        zsr = ZeroSignRestriction(
            restrictions=["ys(1,2)=0", "y(1,1)=1"])
        result = bvar(y, p_var;
                      prior=FlatPrior(),
                      identification=zsr,
                      K=ndraws, hor=hor_val, fhor=4,
                      verbose=false, rng=Random.MersenneTwister(40))
        @test result.ndraws == ndraws
        @test result.identification isa ZeroSignRestriction
        @test size(result.irzerosign_draws) == (K_var, hor_val, K_var, ndraws)
        # Check zero restrictions are satisfied in non-NaN draws
        for d in 1:ndraws
            ir_d = result.irzerosign_draws[:, :, :, d]
            if all(isfinite, ir_d)
                @test abs(ir_d[1, 1, 2]) < 1e-8
            end
        end
    end

    # ─── Proxy ──────────────────────────────────────────────────────────
    @testset "bvar ProxyIdentification" begin
        rng_prx = Random.MersenneTwister(50)
        # Estimate to get residuals, then create instrument
        v_ols = var_estimate(y, p_var; constant=true)
        T_res = size(v_ols.residuals, 1)
        z = v_ols.residuals[:, 1] + 0.5 * randn(rng_prx, T_res)
        instrument = reshape(z, :, 1)

        pid = ProxyIdentification(instrument)
        result = bvar(y, p_var;
                      prior=FlatPrior(),
                      identification=pid,
                      K=ndraws, hor=hor_val, fhor=4,
                      verbose=false, rng=Random.MersenneTwister(60))
        @test result.ndraws == ndraws
        @test result.identification isa ProxyIdentification
        @test size(result.irproxy_draws) == (K_var, hor_val, K_var, ndraws)
        # At least first shock column should have non-zero values
        @test any(result.irproxy_draws[:, :, 1, :] .!= 0)
    end

    # ─── Heteroskedasticity ─────────────────────────────────────────────
    @testset "bvar HeteroskedIdentification" begin
        # Create data with regime shift for heterosked
        rng_het = Random.MersenneTwister(70)
        T_het = 300
        K_het = 2
        A_true = [1.0 0.0; 0.3 0.8]
        Phi_true = [0.5 0.1; 0.0 0.4]

        y_het = zeros(T_het + 100, K_het)
        for t in 2:(T_het + 100)
            scale = t <= (T_het + 100) ÷ 2 ? 1.0 : 2.5
            eps_t = scale * randn(rng_het, K_het)
            y_het[t, :] = Phi_true' * y_het[t-1, :] + A_true * eps_t
        end
        y_het = y_het[101:end, :]

        # Regime vector: based on effective sample (T - p observations)
        T_eff = T_het - 1  # p=1
        regimes = vcat(fill(1, T_eff ÷ 2), fill(2, T_eff - T_eff ÷ 2))

        hid = HeteroskedIdentification(regimes)
        result = bvar(y_het, 1;
                      prior=FlatPrior(),
                      identification=hid,
                      K=ndraws, hor=hor_val, fhor=4,
                      verbose=false, rng=Random.MersenneTwister(80))
        @test result.ndraws == ndraws
        @test result.identification isa HeteroskedIdentification
        @test size(result.irheterosked_draws) == (K_het, hor_val, K_het, ndraws)
        # At least some heterosked IRFs should be finite
        @test any(isfinite, result.irheterosked_draws)
    end

    # ─── Heteroskedasticity SVD properties ──────────────────────────────
    @testset "compute_irf_heterosked — SVD properties" begin
        rng_svd = Random.MersenneTwister(42)
        K_het = 2
        T_het = 400
        Phi_true = [0.5 0.1; 0.0 0.4]
        y_het = zeros(T_het + 100, K_het)
        for t in 2:(T_het + 100)
            scale = t <= (T_het + 100) ÷ 2 ? 1.0 : 3.0
            y_het[t, :] = Phi_true' * y_het[t-1, :] + scale * randn(rng_svd, K_het)
        end
        y_het = y_het[101:end, :]

        v = var_estimate(y_het, 1; constant=true)
        T_eff = v.nobs
        regimes = vcat(fill(1, T_eff ÷ 2), fill(2, T_eff - T_eff ÷ 2))

        ir, Omega = compute_irf_heterosked(v.Phi, v.residuals, regimes, 8, 1)
        @test size(ir) == (K_het, 8, K_het)
        @test size(Omega) == (K_het, K_het)
        # Omega = V from SVD should be orthonormal
        @test Omega' * Omega ≈ I(K_het) atol = 1e-10
        # IRFs should be finite
        @test all(isfinite, ir)
    end

    # ─── Minnesota prior with identification ────────────────────────────
    @testset "bvar Minnesota + SignRestriction" begin
        sr = SignRestriction(restrictions=["y(1,1,1)>0"],
                             max_rotations=5000)
        result = bvar(y, p_var;
                      prior=MinnesotaPrior(),
                      identification=sr,
                      K=ndraws, hor=hor_val, fhor=4,
                      verbose=false, rng=Random.MersenneTwister(90))
        @test result.ndraws == ndraws
        @test result.prior isa MinnesotaPrior
        @test result.identification isa SignRestriction
    end
end
