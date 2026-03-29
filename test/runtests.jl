using SafeTestsets

@time @safetestset "Utilities" include("test_utils.jl")
@time @safetestset "Estimation" include("test_estimation.jl")
@time @safetestset "IRF" include("test_irf.jl")
@time @safetestset "Priors" include("test_priors.jl")
@time @safetestset "BVAR" include("test_bvar.jl")
@time @safetestset "FEVD" include("test_fevd.jl")
@time @safetestset "Forecasting" include("test_forecasting.jl")
@time @safetestset "Connectedness" include("test_connectedness.jl")
@time @safetestset "Historical Decomposition" include("test_histdecomp.jl")
@time @safetestset "Marginal Likelihood" include("test_marginal_likelihood.jl")
@time @safetestset "Filters" include("test_filters.jl")
@time @safetestset "Local Projections" include("test_local_projections.jl")
@time @safetestset "Panel VAR" include("test_panel.jl")
@time @safetestset "Kalman Filter" include("test_kalman.jl")
@time @safetestset "Principal Components" include("test_principal_components.jl")

