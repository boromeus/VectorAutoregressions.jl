#=
connectedness.jl — Diebold–Yilmaz connectedness measures
Port of MATLAB connectedness.m
=#

"""
    compute_connectedness(Phi, Sigma, horizon; Omega=nothing)

Compute the Diebold–Yilmaz (2012) connectedness measures.

If `Omega` is `nothing`, use Pesaran–Shin generalized IRF (default).
Otherwise use the supplied identification matrix.

# Returns
A `ConnectednessResult`.
"""
function compute_connectedness(Phi::AbstractMatrix, Sigma::AbstractMatrix,
                               horizon::Int; Omega=nothing)
    m, ny = size(Phi)
    lags = if rem(m, ny) == 0
        m ÷ ny
    else
        (m - 1) ÷ ny
    end

    pesaran_shin = Omega === nothing
    if pesaran_shin
        Omega = Sigma
    end

    # AR part only
    Phi0 = Phi[1:lags*ny, :]
    # MA representation
    Psi = var2ma(Phi0, horizon)

    # Denominator: DD(ss,ss) = Σ_h Psi_h * Σ * Psi_h'
    DD = zeros(ny, ny)
    for hh in 0:horizon-1
        DD += Psi[:, :, hh+1] * Sigma * Psi[:, :, hh+1]'
    end

    # Numerator and theta
    theta = zeros(ny, ny)
    for ss in 1:ny
        for j in 1:ny
            NN = 0.0
            for hh in 0:horizon-1
                AA = Psi[:, :, hh+1] * Omega
                NN += AA[ss, j]^2
            end
            theta[ss, j] = NN / DD[ss, ss]
            if pesaran_shin
                theta[ss, j] *= Sigma[j, j]^(-0.5)
            end
        end
    end

    # Normalize rows to sum to 1
    Theta = theta ./ sum(theta, dims=2)

    # Remove diagonal
    Theta0 = Theta - Diagonal(diag(Theta))

    index = sum(Theta0) / ny * 100
    from_all = vec(sum(Theta0, dims=2)) / (ny - 1) * 100
    from_unit = vec(sum(Theta0, dims=1)) / (ny - 1) * 100
    net = from_unit - from_all

    return ConnectednessResult(index, from_all, from_unit, net, theta)
end

"""
    connectedness_posterior(result::BVARResult; horizon=12, conf_level=0.68)

Compute posterior distribution of connectedness index.
"""
function connectedness_posterior(result::BVARResult; horizon::Int=12,
                                conf_level::Float64=0.68)
    ndraws = result.ndraws
    K = result.nvar
    indices = zeros(ndraws)

    for d in 1:ndraws
        Phi = result.Phi_draws[:, :, d]
        Sigma = result.Sigma_draws[:, :, d]
        c = compute_connectedness(Phi, Sigma, horizon)
        indices[d] = c.index
    end

    alpha = (1 - conf_level) / 2
    return (median=median(indices),
            lower=quantile(indices, alpha),
            upper=quantile(indices, 1 - alpha),
            draws=indices)
end
