using VectorAutoregressions
using Test
using LinearAlgebra
using Random

include("test_helpers.jl")

@testset "Identification" begin

    # ─── parse_sign_restriction ─────────────────────────────────────────────
    @testset "parse_sign_restriction" begin
        @testset "single horizon" begin
            r = parse_sign_restriction("y(1,2,3)>0", 3)
            @test length(r) == 1
            @test r[1].var == 1
            @test r[1].horizon == 2
            @test r[1].shock == 3
            @test r[1].sign == 1
        end

        @testset "negative sign" begin
            r = parse_sign_restriction("y(2,1,1)<0", 3)
            @test length(r) == 1
            @test r[1].var == 2
            @test r[1].horizon == 1
            @test r[1].shock == 1
            @test r[1].sign == -1
        end

        @testset "range horizons" begin
            r = parse_sign_restriction("y(1,1:4,1)>0", 3)
            @test length(r) == 4
            for (i, ri) in enumerate(r)
                @test ri.var == 1
                @test ri.horizon == i
                @test ri.shock == 1
                @test ri.sign == 1
            end
        end

        @testset "range horizon negative" begin
            r = parse_sign_restriction("y(3,2:5,2)<0", 3)
            @test length(r) == 4
            @test r[1].horizon == 2
            @test r[end].horizon == 5
            @test all(ri -> ri.sign == -1, r)
        end

        @testset "narrative restriction" begin
            r = parse_sign_restriction("v(1,2)>0", 3)
            @test length(r) == 1
            @test r[1].type == :narrative
            @test r[1].times == [1]
            @test r[1].shock == 2
            @test r[1].sign == 1
        end

        @testset "narrative range" begin
            r = parse_sign_restriction("v(1:5,2)>0", 3)
            @test length(r) == 1
            @test r[1].type == :narrative
            @test r[1].times == collect(1:5)
            @test r[1].shock == 2
        end

        @testset "narrative negative" begin
            r = parse_sign_restriction("v(3,1)<0", 3)
            @test r[1].sign == -1
        end

        @testset "parse error" begin
            @test_throws ErrorException parse_sign_restriction("garbage", 3)
            @test_throws ErrorException parse_sign_restriction("z(1,2,3)>0", 3)
        end
    end

    # ─── check_sign_restrictions ────────────────────────────────────────────
    @testset "check_sign_restrictions" begin
        K = 3
        hor = 4

        @testset "satisfied restrictions" begin
            ir = ones(K, hor, K)
            restrictions = ["y(1,1,1)>0", "y(2,2,2)>0"]
            @test check_sign_restrictions(ir, restrictions, K) == true
        end

        @testset "violated restrictions" begin
            ir = ones(K, hor, K)
            ir[1, 1, 1] = -1.0  # violates y(1,1,1)>0
            restrictions = ["y(1,1,1)>0"]
            @test check_sign_restrictions(ir, restrictions, K) == false
        end

        @testset "negative sign restriction satisfied" begin
            ir = -ones(K, hor, K)
            restrictions = ["y(1,1,1)<0"]
            @test check_sign_restrictions(ir, restrictions, K) == true
        end

        @testset "horizon exceeds ir size" begin
            ir = ones(K, 2, K)
            restrictions = ["y(1,5,1)>0"]
            @test check_sign_restrictions(ir, restrictions, K) == false
        end

        @testset "range restriction" begin
            ir = ones(K, hor, K)
            ir[2, 3, 1] = -1.0  # violates at horizon 3
            restrictions = ["y(2,1:4,1)>0"]
            @test check_sign_restrictions(ir, restrictions, K) == false
        end

        @testset "multiple restrictions all satisfied" begin
            ir = ones(K, hor, K)
            ir[3, :, 2] .= -1.0
            restrictions = ["y(1,1,1)>0", "y(3,1:4,2)<0"]
            @test check_sign_restrictions(ir, restrictions, K) == true
        end
    end

    # ─── check_narrative_restrictions ───────────────────────────────────────
    @testset "check_narrative_restrictions" begin
        K = 3

        @testset "satisfied" begin
            shocks = ones(10, K)
            narrative = ["v(1,1)>0", "v(5,2)>0"]
            @test check_narrative_restrictions(shocks, narrative, K) == true
        end

        @testset "violated" begin
            shocks = ones(10, K)
            shocks[1, 1] = -1.0
            narrative = ["v(1,1)>0"]
            @test check_narrative_restrictions(shocks, narrative, K) == false
        end

        @testset "negative narrative" begin
            shocks = -ones(10, K)
            narrative = ["v(3,2)<0"]
            @test check_narrative_restrictions(shocks, narrative, K) == true
        end

        @testset "range narrative" begin
            shocks = ones(10, K)
            shocks[3, 1] = -1.0
            narrative = ["v(1:5,1)>0"]
            @test check_narrative_restrictions(shocks, narrative, K) == false
        end

        @testset "out of bounds time" begin
            shocks = ones(10, K)
            narrative = ["v(15,1)>0"]
            @test check_narrative_restrictions(shocks, narrative, K) == false
        end
    end

    # ─── max_horizon_sign ───────────────────────────────────────────────────
    @testset "max_horizon_sign" begin
        @test max_horizon_sign(["y(1,3,1)>0"], 3) == 3
        @test max_horizon_sign(["y(1,1:6,1)>0"], 3) == 6
        @test max_horizon_sign(["y(1,2,1)>0", "y(2,5,2)<0"], 3) == 5
        # Narrative‑only restrictions should leave max at 1
        @test max_horizon_sign(["v(1,1)>0"], 3) == 1
    end

    # ─── parse_zero_sign_restrictions ───────────────────────────────────────
    @testset "parse_zero_sign_restrictions" begin
        K = 3

        @testset "short-run zero" begin
            f, sr = parse_zero_sign_restrictions(["ys(1,2)=0"], K)
            @test size(f) == (2K, K)
            @test f[1, 2] == 0.0   # ys row for (1,2)
            @test f[1, 1] == 1.0   # not restricted
        end

        @testset "long-run zero" begin
            f, sr = parse_zero_sign_restrictions(["yr(2,1,3)=0"], K)
            @test f[K + 2, 3] == 0.0  # yr row for (2,3)
        end

        @testset "sign restriction" begin
            f, sr = parse_zero_sign_restrictions(["y(1,2)=1"], K)
            @test sr[1, 2] == 1.0
            @test isnan(sr[1, 1])  # unrestricted entry
        end

        @testset "negative sign" begin
            f, sr = parse_zero_sign_restrictions(["y(1,2)=-1"], K)
            @test sr[1, 2] == -1.0
        end

        @testset "mixed restrictions" begin
            f, sr = parse_zero_sign_restrictions(["ys(1,2)=0", "y(1,1)=1", "yr(3,1,1)=0"], K)
            @test f[1, 2] == 0.0    # short‑run zero
            @test f[K + 3, 1] == 0.0  # long‑run zero
            @test sr[1, 1] == 1.0    # sign restriction
        end

        @testset "parse error" begin
            @test_throws ErrorException parse_zero_sign_restrictions(["garbage"], K)
        end
    end

    # ─── findQs ─────────────────────────────────────────────────────────────
    @testset "findQs" begin
        K = 3

        @testset "exactly identified" begin
            # Build f matrix that yields exact identification
            f = ones(2K, K)
            f[1, 2] = 0.0  # ys(1,2)=0
            f[2, 3] = 0.0  # ys(2,3)=0
            Q_cells, index, flag = findQs(K, f)
            @test length(Q_cells) == K
            @test length(index) == K
            # flag should be 0 (exact) or -1 (under) depending on count
            @test flag in [-1, 0]
        end

        @testset "over-identified" begin
            f = zeros(2K, K)  # all zeros = all restricted → over-identified
            Q_cells, index, flag = findQs(K, f)
            @test flag == 1  # over-identified
        end

        @testset "output structure" begin
            f = ones(2K, K)
            f[1, 2] = 0.0
            Q_cells, index, flag = findQs(K, f)
            for q in Q_cells
                @test isa(q, Matrix{Float64})
            end
        end
    end

    # ─── findP ──────────────────────────────────────────────────────────────
    @testset "findP" begin
        rng_fp = Random.MersenneTwister(42)
        K = 2
        p = 1

        Phi = [0.5 0.1; 0.0 0.4]
        Sigma = [1.0 0.2; 0.2 0.8]
        C = cholesky(Hermitian(Sigma)).L

        # Set up restrictions with at least one zero constraint per column
        # so findP has non-trivial Q_cells
        restrictions = ["ys(1,2)=0", "y(1,1)=1"]
        f, sr = parse_zero_sign_restrictions(restrictions, K)
        Q_cells, index, flag = findQs(K, f)

        P = findP(C, Phi, Q_cells, p, K, index)
        @test size(P) == (K, K)
        # P should be orthonormal (or close)
        @test P' * P ≈ I(K) atol = 1e-8
    end

    # ─── _check_sign_with_flip ──────────────────────────────────────────────
    @testset "_check_sign_with_flip" begin
        K = 2
        hor = 4

        @testset "original satisfies" begin
            ir = ones(K, hor, K)
            restrictions = ["y(1,1,1)>0"]
            sat, fsign = VectorAutoregressions._check_sign_with_flip(ir, restrictions, K)
            @test sat == true
            @test fsign == 1
        end

        @testset "negated satisfies" begin
            # IRF is all negative → original fails y>0, but -ir satisfies
            ir = -ones(K, hor, K)
            restrictions = ["y(1,1,1)>0"]
            sat, fsign = VectorAutoregressions._check_sign_with_flip(ir, restrictions, K)
            @test sat == true
            @test fsign == -1
        end

        @testset "neither satisfies" begin
            ir = ones(K, hor, K)
            # Require both positive and negative — impossible
            restrictions = ["y(1,1,1)>0", "y(2,1,1)<0"]
            ir[2, 1, 1] = 1.0  # violates <0
            sat, fsign = VectorAutoregressions._check_sign_with_flip(ir, restrictions, K)
            # negated: ir[1,1,1]=-1 violates >0, ir[2,1,1]=-1 satisfies <0
            # so neither original nor negated satisfies both
            @test sat == false
        end
    end

    # ─── irf_sign_restriction with sign-flip ────────────────────────────────
    @testset "irf_sign_restriction — sign flip doubles acceptance" begin
        rng_sr = Random.MersenneTwister(123)
        K = 2
        p = 1
        Phi = [0.5 0.1; 0.0 0.4]
        Sigma = [1.0 0.2; 0.2 0.8]

        # Simple restriction that should be found
        restrictions = ["y(1,1,1)>0"]
        ir, Omeg = irf_sign_restriction(Phi, Sigma, 8, restrictions;
                                         max_rotations=5000, rng=rng_sr)
        @test !any(isnan.(ir))
        @test !any(isnan.(Omeg))
        # Verify the restriction is satisfied
        @test ir[1, 1, 1] > 0
    end
end
