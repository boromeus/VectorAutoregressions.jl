# VectorAutoregressions.jl — Implementation Plan

## Overview

Port all features from the MATLAB BVAR\_ toolbox (~100+ functions) into a clean, modular
Julia package following the [SciML Style Guide](https://docs.sciml.ai/SciMLStyle/stable/).

**Current status:** Phase 1 complete — all 18 source files created, 87/87 tests passing.

---

## Phase 1: Foundation — Core VAR + BVAR Estimation ✅ COMPLETE

Rewrote the package from scratch with 18 modular source files covering:

- **types.jl** — Full type hierarchy: `AbstractPrior` (Flat, Minnesota, Conjugate),
  `AbstractIdentification` (Cholesky, Sign, NarrativeSign, ZeroSign, Proxy, LongRun,
  Heterosked), result types (`VAREstimate`, `BVARResult`, `IRFResult`, `FEVDResult`,
  `ForecastResult`, `HistDecompResult`, `ConnectednessResult`, `LPResult`, `PanelVARResult`,
  `FAVARResult`)
- **utils.jl** — `lagmatrix`, `companion_form`, `check_stability`, `rand_inverse_wishart`,
  `vech`/`ivech`, `commutation_matrix`, `duplication_matrix`, `elimination_matrix`,
  `var2ma`, `ols_svd`, `generate_rotation_matrix`, `matrictint`, `ggammaln`
- **estimation.jl** — `var_estimate`, `var_lagorder`, `information_criteria`, `rfvar3`
- **priors.jl** — `get_prior_moments`, `build_dummy_observations`,
  `compute_prior_posterior` (dispatched on Flat/Minnesota/Conjugate)
- **bayesian.jl** — `bvar` (Gibbs sampler), `classical_var` (bootstrap)
- **marginal\_likelihood.jl** — `compute_marginal_likelihood`,
  `optimize_hyperparameters`
- **irf.jl** — `compute_irf`, `compute_irf_longrun`, `compute_irf_proxy`
- **identification.jl** — `parse_sign_restriction`, `check_sign_restrictions`,
  `check_narrative_restrictions`, `irf_sign_restriction`, `irf_narrative_sign`,
  `irf_zero_sign`, `findQs`, `findP`
- **fevd.jl** — `compute_fevd`, `fevd_posterior`
- **historical\_decomp.jl** — `historical_decomposition`
- **forecasting.jl** — `forecast_unconditional`, `forecast_conditional`
- **connectedness.jl** — `compute_connectedness`, `connectedness_posterior`
- **local\_projections.jl** — `lp_irf`, `lp_lagorder`
- **kalman.jl** — `kalman_filter` (forward filter + RTS smoother with NaN handling)
- **panel.jl** — `panel_var` (pooled and unit-by-unit)
- **favar\_new.jl** — `principal_components`, `rescale_favar`, `favar`
- **filters.jl** — `hp_filter`, `bk_filter`, `cf_filter`, `hamilton_filter`
- **plotting.jl** — RecipesBase plot recipes for IRF, FEVD, Forecast, HistDecomp

Dependencies: Distributions, LinearAlgebra, Random, RecipesBase, SpecialFunctions,
Statistics.

Test results: **87 passed, 0 failed** across 16 test sets.

---

## Phase 2: SciML Style Guide Compliance

Refactor all source files so the codebase conforms to the
[SciML Style Guide](https://docs.sciml.ai/SciMLStyle/stable/). This is a
style-only phase — no new features, no behavioural changes. All 87 tests must
continue to pass after every change.

### 2.1 Formatter setup

- Add `.JuliaFormatter.toml` with `style = "sciml"` to the repo root.
- Run `JuliaFormatter.format(".")` to auto-fix whitespace, indentation,
  line-length, and operator spacing.
- Verify tests still pass.

### 2.2 For-loop syntax

- Replace every `for i = ...` with `for i in ...` across all files.
- Legacy files `bvar.jl` and `favar.jl` have the most instances (~13 total).
- New source files are already compliant.

### 2.3 Import / using organisation

**SciML rule:** `import` and `using` statements must be separated by a blank
line; prefer explicit imports (`using Foo: bar, baz`).

- Module file (`VectorAutoregressions.jl`) already uses explicit imports —
  verify ordering and blank-line separation.
- Legacy `bvar.jl` line 1 mixes many packages on one `using` line — fix or
  remove (superseded by new code).
- Legacy `favar.jl` line 1 — same issue.

### 2.4 Naming audit

**SciML rules:**
- Functions: `snake_case` ✓ (already compliant)
- Types/structs: `CamelCase` ✓ (already compliant)
- Abstract types: prefixed with `Abstract` ✓
- Constants: `UPPER_SNAKE_CASE`
- Variables: `snake_case`
- Private internals: prefix with `__`

Action items:
- Rename internal helper functions that are not exported but lack `_` or `__`
  prefix (e.g. `_build_lp_rhs`, `_newey_west_se` are already prefixed — verify
  all helpers follow suit).
- Audit any constants and ensure `UPPER_SNAKE_CASE`.

### 2.5 Function type annotations — generalise signatures

**SciML rule:** Function annotations should be as general as possible
(`AbstractArray` not `Array`, `Integer` not `Int`, `Real` not `Float64`).

Files to audit (the new source files are mostly compliant but need a pass):
- `priors.jl` — recently changed to `AbstractVector{<:Real}`; verify all
  functions.
- `bayesian.jl` — `bvar` already uses `AbstractMatrix{<:Real}`; check helpers.
- `identification.jl` — verify restriction parsing functions.
- `filters.jl`, `panel.jl`, `kalman.jl` — verify input signatures.

### 2.6 Struct field annotations

**SciML rule:** Struct fields should be concretely typed or parametric — never
abstract. Untyped fields must be explicitly typed `Any`.

- Audit all structs in `types.jl`.
- Current structs use concrete `Float64`, `Int`, `Matrix{Float64}`,
  `Array{Float64,3}` — this is correct per SciML.
- Check `Union{Nothing,...}` fields — acceptable but keep to ≤ 3 types.

### 2.7 Error handling

**SciML rule:** Prefer custom exception types over `error("string")`. Avoid
`try/catch` where possible.

- Audit all `error()` calls across source files.
- Define domain-specific exceptions where appropriate:
  - `DimensionMismatchError` — for incompatible matrix sizes
  - `ConvergenceError` — for Gibbs sampler / optimiser failures
  - `IdentificationError` — for sign-restriction failures
- Replace bare `error("...")` calls with `throw(SpecificError(...))`.
- Minimise `try/catch` blocks (currently used in `bayesian.jl` marginal
  likelihood and `irf.jl` Cholesky fallback).

### 2.8 Data initialisation

**SciML rule:** Always default to constructs that initialise data. Avoid
`Array{T}(undef, ...)` unless performance-critical and fully initialised before
return.

- Audit for `undef` usage — replace with `zeros()`, `fill()`, or `similar()`
  + immediate fill where possible.
- Legacy `bvar.jl` uses `Array{Float64}(undef, 0)` — fix or remove file.

### 2.9 Closures

**SciML rule:** Closures should be avoided wherever possible.

- Audit for anonymous functions (`->`) and closures that capture outer scope.
- Replace with named functions or `Base.Fix1`/`Base.Fix2` where feasible.

### 2.10 Line length (92 chars)

- JuliaFormatter handles most cases automatically.
- Manually review long function signatures in `local_projections.jl`,
  `bayesian.jl`, and `identification.jl`.

### 2.11 Whitespace and formatting

- 4-space indentation ✓ (already compliant in new files)
- No extraneous blank lines between short-form definitions ✓
- Blank line between multi-line blocks ✓
- No trailing whitespace
- Unix line endings (`\n`)

### 2.12 Numbers

**SciML rule:** Floats must have explicit leading/trailing zero (`0.1` not `.1`,
`2.0` not `2.`). Prefer `Int` over `Int32`/`Int64`.

- Grep for `\.(\d)` and `(\d)\.(\s|,|\))` patterns.
- Replace `Int64` with `Int` wherever the bit-size is not semantically required.

### 2.13 Test structure

**SciML rules:**
- `runtests.jl` should only shuttle to other test files.
- Each test set should use `@safetestset` (from SafeTestsets.jl).
- Every test script should be fully reproducible in isolation.

Action items:
- Split `test/runtests.jl` into separate files per module (e.g.
  `test/test_utils.jl`, `test/test_estimation.jl`, `test/test_bvar.jl`, etc.).
- Top-level `runtests.jl` becomes include-only:
  ```julia
  using SafeTestsets
  @time @safetestset "Utilities" include("test_utils.jl")
  @time @safetestset "Estimation" include("test_estimation.jl")
  # ...
  ```
- Add `SafeTestsets` to `test/Project.toml`.
- Each test file includes its own `using` statements for full isolation.

### 2.14 Documentation (docstrings)

**SciML rule:** All exported functions should have docstrings following the
template: signature, description, arguments, keywords, returns.

- Add/update docstrings for all exported functions.
- Use the SciML docstring template format.

### 2.15 Remove legacy files

- Delete `src/bvar.jl` (superseded by `bayesian.jl` + `priors.jl`).
- Delete `src/favar.jl` (superseded by `favar_new.jl`).
- Delete `run_tests.jl` (test helper workaround, no longer needed).
- These files are non-compliant and unmaintained.

### 2.16 Package version specifications

**SciML rule:** Every dependency should have a compat bound. Use semver. Avoid
caret specifiers.

- Verify all deps in `Project.toml` have `[compat]` entries.
- Ensure lower bounds match last tested versions.

### 2.17 Verification

- Run `JuliaFormatter.format(".")` — no changes should remain.
- Run `Pkg.test()` — all 87 tests must pass.
- Run `Aqua.test_all(VectorAutoregressions)` for ambiguity / piracy checks
  (optional but recommended).

---

## Phase 3: Advanced Identification Schemes

*Depends on Phase 2 completion.*

Ensure all 6+ identification methods are thoroughly tested and integrated
into the BVAR Gibbs loop.

### Steps

1. **Sign restrictions** — Validate acceptance-sampling loop against known
   DGPs. Add tests for convergence.
2. **Narrative + sign** — Test with historical event constraints.
3. **Zero + sign** — Test exact zero restrictions produce zeros in IRFs.
4. **Proxy / IV** — Validate against Gertler-Karadi test data. Port wild
   bootstrap from original code.
5. **Long-run** — Test Blanchard-Quah identification against analytical case.
6. **Heteroskedasticity** — Regime-based identification (stub exists in types;
   implement estimation).

### Verification

- Rotation matrix orthonormality: `Q'Q ≈ I`.
- Sign-restricted IRFs satisfy all declared sign restrictions.
- Proxy IRFs match existing CSV test data.
- Zero restrictions produce exact zeros at specified positions.

---

## Phase 4: Analysis & Forecasting Tools

*Can run in parallel with Phase 3 for FEVD/histdecomp (only needs Cholesky).*

### Steps

1. **FEVD** — Already implemented. Add posterior HPD-band tests.
2. **Historical decomposition** — Already implemented. Test that decomposition
   sums to observed data.
3. **Unconditional forecasts** — Already implemented. Test posterior predictive
   distribution.
4. **Conditional forecasts** — Already implemented (Waggoner-Zha SVD approach).
   Test that conditions are satisfied exactly.
5. **Connectedness** — Already implemented (Diebold-Yilmaz). Test row sums =
   100%.

### Verification

- FEVD rows sum to 100 at each horizon.
- Historical decomposition sums to observed data.
- Flat-prior BVAR forecast ≈ OLS forecast.
- Conditional forecast hits target path.
- Connectedness spillover table rows sum to 100%.

---

## Phase 5: Extensions

*Sub-steps are largely independent of each other.*

### Steps

1. **Local projections** — Already implemented (OLS, lag selection). Add LP-IV
   and LP-Bayesian methods from MATLAB `directmethods.m`.
2. **Kalman filter** — Already implemented with NaN handling. Add
   mixed-frequency VAR integration.
3. **Panel VAR** — Already implemented (pooled, unit-by-unit). Add
   exchangeable / partial-pooling method.
4. **FAVAR** — Already implemented (Gibbs sampler, PC extraction). Add
   Bernanke-Boivin-Eliasz rotation and loading-matrix estimation.
5. **Regularisation** — Ridge, Lasso, Elastic Net (from MATLAB
   `Ridge_`/`Lasso_`/`ElasticNet_` options).
6. **Business cycle** — Peak/trough dating, recession probabilities, BN
   decomposition.

### Verification

- LP: preserve all existing test assertions.
- Kalman on complete data matches OLS.
- Panel pooled = stacked OLS.
- FAVAR factors are orthogonal.
- Ridge(λ=0) = OLS.

---

## Phase 6: Documentation, Examples & Polish

*Final phase after all features are stable.*

### Steps

1. **Documenter.jl site** — Expand `docs/` with tutorials and API reference.
2. **Tutorials** — Port key MATLAB examples:
   - Classical VAR with Cholesky IRFs
   - Minnesota-prior BVAR
   - Sign-restricted BVAR
   - Proxy SVAR (Gertler-Karadi)
   - Conditional forecasting
   - FAVAR
   - Local projections
3. **README** — Usage examples, badges (SciML style, CI, codecov).
4. **CI** — GitHub Actions for Julia LTS + stable, format check, codecov.
5. **Registration** — Register package in General registry.

### Verification

- All doctests pass.
- All example scripts run without error.
- CI green on LTS and stable Julia.
- Documentation site builds and deploys.

---

## Decisions Log

| Decision | Rationale |
|----------|-----------|
| Clean redesign (not incremental) | Original code too tightly coupled |
| Removed GrowableArrays, Parameters | Replaced with stdlib constructs |
| Added SpecialFunctions.jl | For `loggamma` in marginal likelihood |
| RecipesBase.jl for plotting | Lightweight; no Plots.jl dependency |
| Restriction string syntax `y(1,1:4,1)>0` | Matches MATLAB for user familiarity |
| No `eval` in restriction checking | Security; parse to matrix form at construction |
| `rng::AbstractRNG` argument | Reproducibility without global `seed!` |
| SciML style compliance | Community standard; JuliaFormatter automated |
