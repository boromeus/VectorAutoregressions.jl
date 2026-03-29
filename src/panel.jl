#=
panel.jl — Panel VAR estimation (pooled and unit‑by‑unit)
=#

"""
    panel_var(data, p; method=:pooled, unit_col=1, constant=true)

Estimate a panel VAR.

# Arguments
- `data`:     Vector of T_i × K matrices (one per unit), or a single matrix with a unit‑id column.
- `p`:        lag length.
- `method`:   `:pooled` (stack all units) or `:unit` (unit‑by‑unit).
- `unit_col`: column index for unit identifiers if `data` is a single matrix. Set to 0 if using a Vector of matrices.

# Returns
`PanelVARResult`.
"""
function panel_var(panels::Vector{<:AbstractMatrix}, p::Int;
                   method::Symbol=:pooled, constant::Bool=true)
    n_units = length(panels)
    K = size(panels[1], 2)

    if method == :unit
        # Estimate each unit separately
        unit_results = VAREstimate[]
        for panel in panels
            v = var_estimate(panel, p; constant=constant)
            push!(unit_results, v)
        end
        # Average coefficients and covariance
        Phi_avg = mean([r.Phi for r in unit_results])
        Sigma_avg = mean([r.Sigma for r in unit_results])
        resid_all = vcat([r.residuals for r in unit_results]...)
        return PanelVARResult(Phi_avg, Sigma_avg, resid_all, unit_results, :unit)
    else
        # Pooled estimation: stack all data
        Y_all = Matrix{Float64}(undef, 0, K)
        X_all = Matrix{Float64}(undef, 0, 0)
        resid_all = Matrix{Float64}(undef, 0, K)
        unit_results = VAREstimate[]

        first = true
        for panel in panels
            v = var_estimate(panel, p; constant=constant)
            push!(unit_results, v)
            Y_all = vcat(Y_all, v.Y)
            if first
                X_all = v.X
                first = false
            else
                X_all = vcat(X_all, v.X)
            end
            resid_all = vcat(resid_all, v.residuals)
        end

        # Pooled OLS
        Phi_pooled = (X_all' * X_all) \ (X_all' * Y_all)
        resid_pooled = Y_all - X_all * Phi_pooled
        Sigma_pooled = (resid_pooled' * resid_pooled) / size(resid_pooled, 1)

        return PanelVARResult(Phi_pooled, Sigma_pooled, resid_pooled,
                              unit_results, :pooled)
    end
end

"""
    panel_var(data, unit_ids, p; method=:pooled, constant=true)

Alternative interface: single data matrix with unit identifiers.
"""
function panel_var(data::AbstractMatrix, unit_ids::AbstractVector, p::Int;
                   method::Symbol=:pooled, constant::Bool=true)
    unique_ids = sort(unique(unit_ids))
    panels = [data[unit_ids .== id, :] for id in unique_ids]
    return panel_var(panels, p; method=method, constant=constant)
end
