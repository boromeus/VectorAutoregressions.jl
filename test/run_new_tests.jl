using VectorAutoregressions
using Test
using LinearAlgebra
using Random
using Statistics
using Distributions

include("test_helpers.jl")

println("Running B2 tests (heterosked_weights)...")
include("test_heterosked_weights.jl")
println("B2 DONE")

println("Running B4 tests (robust_bayes)...")
include("test_robust_bayes.jl")
println("B4 DONE")

println("Running B5 tests (nowcast)...")
include("test_nowcast.jl")
println("B5 DONE")

println("Running B6 tests (exogenous_block)...")
include("test_exogenous_block.jl")
println("B6 DONE")

println("Running B1 tests (marginal_likelihood)...")
include("test_marginal_likelihood.jl")
println("B1 DONE")

println("Running B3 tests (forecasting)...")
include("test_forecasting.jl")
println("B3 DONE")

println("ALL B-PHASE TESTS COMPLETE")
