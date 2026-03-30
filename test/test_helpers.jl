# Shared test helper: generate simple VAR data for testing
if !@isdefined(generate_var_data)
function generate_var_data(T::Int, K::Int, p::Int; rng = Random.MersenneTwister(42))
    Phi_true = zeros(K * p, K)
    # Simple diagonal AR(1) coefficients
    for i in 1:K
        Phi_true[i, i] = 0.5
    end
    Sigma_true = Matrix{Float64}(I, K, K) * 0.5
    Sigma_true[1, 2] = 0.1
    Sigma_true[2, 1] = 0.1
    if K > 2
        Sigma_true[1, 3] = 0.05
        Sigma_true[3, 1] = 0.05
    end

    L = cholesky(Hermitian(Sigma_true)).L
    y = zeros(T + 100, K)  # burn-in
    for t in (p + 1):(T + 100)
        for lag in 1:p
            y[t, :] += Phi_true[((lag - 1) * K + 1):(lag * K), :]' * y[t - lag, :]
        end
        y[t, :] += L * randn(rng, K)
    end
    return y[101:end, :], Phi_true, Sigma_true
end
end  # if !@isdefined
