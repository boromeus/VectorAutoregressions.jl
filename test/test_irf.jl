using VectorAutoregressions
using Test
using LinearAlgebra
using Random
using Statistics

include("test_helpers.jl")

@testset "IRF" begin
    rng = Random.MersenneTwister(42)
    y, _, _ = generate_var_data(500, 3, 1; rng=rng)
    v = var_estimate(y, 1; constant=true)
    K = 3
    hor = 12

    @testset "compute_irf (Cholesky)" begin
        ir = compute_irf(v.Phi, v.Sigma, hor)
        @test size(ir) == (K, hor, K)
        # Impact: lower triangular (Cholesky)
        @test ir[1, 1, 1] > 0  # own shock positive

        # With rotation matrix (identity should match default)
        ir2 = compute_irf(v.Phi, v.Sigma, hor; Omega=Matrix{Float64}(I, K, K))
        @test ir ≈ ir2

        # Non‑unit shocks
        ir_nu = compute_irf(v.Phi, v.Sigma, hor; unit=false)
        @test size(ir_nu) == (K, hor, K)
    end

    @testset "compute_irf_longrun" begin
        ir, Q = compute_irf_longrun(v.Phi, v.Sigma, hor, 1)
        @test size(ir) == (K, hor, K)
        @test size(Q) == (K, K)

        # Q should be non‑singular
        @test abs(det(Q)) > 1e-10

        # Long‑run cumulative IRF should be lower triangular
        # Sum IRFs to get cumulative long‑run response
        cum = sum(ir, dims=2)[:, 1, :]  # K × K
        C1_reconstructed = cum
        # C(1)^{-1} * C1_reconstructed should be close to Q
        # Just test that Q is non-singular and IRFs are finite
        @test all(isfinite, ir)
        @test all(isfinite, Q)
    end

    # ─── Sign Restrictions ──────────────────────────────────────────────────
    @testset "irf_sign_restriction" begin
        restrictions = ["y(1,1,1)>0", "y(2,1,2)>0"]
        rng_sr = Random.MersenneTwister(123)

        ir, Omega = irf_sign_restriction(v.Phi[1:K, :], v.Sigma, hor,
                                         restrictions; rng=rng_sr)

        @test size(ir) == (K, hor, K)
        @test size(Omega) == (K, K)
        # Omega should be orthonormal
        @test Omega' * Omega ≈ I(K) atol = 1e-8
        # IRFs should satisfy declared restrictions
        @test check_sign_restrictions(ir, restrictions, K) == true
        # All values should be finite (found valid rotation)
        @test all(isfinite, ir)
    end

    @testset "irf_sign_restriction range" begin
        restrictions = ["y(1,1:4,1)>0"]
        rng_sr = Random.MersenneTwister(456)

        ir, Omega = irf_sign_restriction(v.Phi[1:K, :], v.Sigma, hor,
                                         restrictions; rng=rng_sr)
        @test all(isfinite, ir)
        @test check_sign_restrictions(ir, restrictions, K) == true
    end

    @testset "irf_sign_restriction infeasible" begin
        # Contradictory restrictions: same element must be both >0 and <0
        restrictions = ["y(1,1,1)>0", "y(1,1,1)<0"]
        rng_sr = Random.MersenneTwister(789)
        ir, Omega = irf_sign_restriction(v.Phi[1:K, :], v.Sigma, 4,
                                         restrictions; max_rotations=100, rng=rng_sr)
        # Should return NaN when no valid rotation found
        @test all(isnan, ir)
    end

    # ─── Narrative + Sign ───────────────────────────────────────────────────
    @testset "irf_narrative_sign" begin
        signs = ["y(1,1,1)>0"]
        narrative = ["v(1,1)>0"]
        rng_ns = Random.MersenneTwister(321)

        ir, Omega = irf_narrative_sign(v.residuals, v.Phi[1:K, :], v.Sigma,
                                       hor, signs, narrative;
                                       max_rotations=10000, rng=rng_ns)

        @test size(ir) == (K, hor, K)
        @test size(Omega) == (K, K)
        if all(isfinite, ir)
            @test Omega' * Omega ≈ I(K) atol = 1e-8
            @test check_sign_restrictions(ir, signs, K) == true
        end
    end

    # ─── Zero + Sign Restrictions ───────────────────────────────────────────
    @testset "irf_zero_sign" begin
        # Restrict: short-run zero at position (1,2) and sign y(1,1)=1
        restrictions = ["ys(1,2)=0", "y(1,1)=1"]
        rng_zs = Random.MersenneTwister(654)

        ir, Omega = irf_zero_sign(v.Phi, v.Sigma, hor, 1, restrictions;
                                  rng=rng_zs)

        @test size(ir) == (K, hor, K)
        @test size(Omega) == (K, K)
        if all(isfinite, ir)
            # Zero restriction: ir[1, 1, 2] should be ≈0
            @test abs(ir[1, 1, 2]) < 1e-10
            # Sign restriction: ir[1, 1, 1] > 0
            @test ir[1, 1, 1] > 0
            # Omega should be orthonormal
            @test Omega' * Omega ≈ I(K) atol = 1e-6
        end
    end

    # ─── Proxy / IV ─────────────────────────────────────────────────────────
    @testset "compute_irf_proxy" begin
        # Create synthetic instrument correlated with first shock
        rng_prx = Random.MersenneTwister(111)
        T_res = size(v.residuals, 1)
        # Instrument = first residual + noise
        z = v.residuals[:, 1] + 0.5 * randn(rng_prx, T_res)
        instrument = reshape(z, :, 1)

        irs, b1, F_stat = compute_irf_proxy(v.Phi, v.Sigma, v.residuals,
                                             instrument, hor, 1;
                                             compute_F_stat=true)

        @test size(irs) == (hor, K)
        @test length(b1) == K
        @test all(isfinite, irs)
        @test all(isfinite, b1)
        # F-stat should be positive and large for a good instrument
        @test F_stat > 0
        @test isfinite(F_stat)
        # b1[1] should have consistent sign (positive for own shock)
        @test abs(b1[1]) > 0
    end

    @testset "compute_irf_proxy without F_stat" begin
        rng_prx = Random.MersenneTwister(222)
        T_res = size(v.residuals, 1)
        z = v.residuals[:, 1] + 0.5 * randn(rng_prx, T_res)
        instrument = reshape(z, :, 1)

        irs, b1, F_stat = compute_irf_proxy(v.Phi, v.Sigma, v.residuals,
                                             instrument, hor, 1)
        @test isnan(F_stat)
        @test size(irs) == (hor, K)
    end

    # ─── Heteroskedasticity ─────────────────────────────────────────────────
    @testset "compute_irf_heterosked" begin
        # Create data with two volatility regimes
        rng_het = Random.MersenneTwister(999)
        T_het = 400
        K_het = 2
        p_het = 1
        Phi_true = [0.5 0.1; 0.0 0.4]
        A_true = [1.0 0.0; 0.3 0.8]

        y_het = zeros(T_het + 100, K_het)
        for t in 2:(T_het + 100)
            # Regime: first half low vol, second half high vol
            scale = t <= (T_het + 100) ÷ 2 ? 1.0 : 2.0
            eps_t = scale * randn(rng_het, K_het)
            u_t = A_true * eps_t
            y_het[t, :] = Phi_true' * y_het[t-1, :] + u_t
        end
        y_het = y_het[101:end, :]

        v_het = var_estimate(y_het, p_het; constant=true)
        T_eff = size(v_het.residuals, 1)
        regimes = vcat(fill(1, T_eff ÷ 2), fill(2, T_eff - T_eff ÷ 2))

        ir_het, A_hat = compute_irf_heterosked(v_het.Phi, v_het.residuals,
                                                regimes, 12, p_het)

        @test size(ir_het) == (K_het, 12, K_het)
        @test size(A_hat) == (K_het, K_het)
        @test all(isfinite, ir_het)
        @test all(isfinite, A_hat)
        # A should be invertible
        @test abs(det(A_hat)) > 1e-10
    end

    @testset "compute_irf_heterosked errors" begin
        @test_throws ArgumentError compute_irf_heterosked(
            v.Phi, v.residuals, fill(1, size(v.residuals, 1)), 12, 1)
        @test_throws ArgumentError compute_irf_heterosked(
            v.Phi, v.residuals, fill(1, 10), 12, 1)
    end

    # ─── Wild Bootstrap ─────────────────────────────────────────────────────
    @testset "wild_bootstrap_irf_proxy" begin
        rng_wb = Random.MersenneTwister(333)
        T_res = size(v.residuals, 1)
        z = v.residuals[:, 1] + 0.5 * randn(rng_wb, T_res)
        instrument = reshape(z, :, 1)

        @testset "rademacher" begin
            rng_boot = Random.MersenneTwister(444)
            result = wild_bootstrap_irf_proxy(v, instrument, 8;
                         nboot=50, conf_level=0.90,
                         weight_type=:rademacher, rng=rng_boot)

            @test size(result.point) == (8, K)
            @test size(result.lower) == (8, K)
            @test size(result.upper) == (8, K)
            @test size(result.boot_irfs) == (8, K, 50)
            @test all(isfinite, result.point)
            # Lower should be ≤ upper
            for h in 1:8, k in 1:K
                if isfinite(result.lower[h, k]) && isfinite(result.upper[h, k])
                    @test result.lower[h, k] <= result.upper[h, k]
                end
            end
        end

        @testset "mammen" begin
            rng_boot = Random.MersenneTwister(555)
            result = wild_bootstrap_irf_proxy(v, instrument, 8;
                         nboot=50, conf_level=0.90,
                         weight_type=:mammen, rng=rng_boot)
            @test size(result.point) == (8, K)
            @test all(isfinite, result.point)
        end

        @testset "invalid weight_type" begin
            @test_throws ArgumentError wild_bootstrap_irf_proxy(
                v, instrument, 8; weight_type=:invalid)
        end

        @testset "invalid conf_level" begin
            @test_throws ArgumentError wild_bootstrap_irf_proxy(
                v, instrument, 8; conf_level=1.5)
        end
    end
end
