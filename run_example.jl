ENV["BVAR_K"] = get(ENV, "BVAR_K", "500")  # use 500 draws for testing speed
ENV["BVAR_LAG_MIN"] = get(ENV, "BVAR_LAG_MIN", "10")
ENV["BVAR_LAG_MAX"] = get(ENV, "BVAR_LAG_MAX", "14")

example_num = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 1

println("Testing example $example_num...")
try
    include(joinpath(@__DIR__, "examples", "example_$(example_num)_" * 
        ["classical", "minn", "irf", "mfvar", "favar", "VARX", "LP", 
         "panels", "prediction", "heterosked", "connectedness", "bdfm", 
         "nonGaussian"][example_num] * ".jl"))
    println("\n✓ Example $example_num passed")
catch e
    println("\n✗ Example $example_num failed:")
    showerror(stdout, e, catch_backtrace())
end
