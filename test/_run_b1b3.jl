using VectorAutoregressions
using Test
using LinearAlgebra
using Random
include("test_helpers.jl")
println("Running B1 tests (marginal_likelihood)...")
include("test_marginal_likelihood.jl")
println("B1 DONE")
println("Running B3 tests (forecasting)...")
include("test_forecasting.jl")
println("B3 DONE")
