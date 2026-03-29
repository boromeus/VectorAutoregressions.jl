#=
plotting.jl — Plot recipes using RecipesBase
=#

using RecipesBase

# ─── IRF plot recipe ────────────────────────────────────────────────────────────

@recipe function f(r::IRFResult; varnames = nothing, shocknames = nothing)
    K = size(r.irf, 1)
    hor = r.horizon

    layout := (K, K)
    legend := false
    grid := true
    linewidth := 2

    for shock in 1:K
        for var in 1:K
            @series begin
                subplot := (var - 1) * K + shock
                fillrange := r.lower[var, :, shock]
                fillalpha := 0.2
                seriescolor := :blue
                label := ""
                title := _make_title(varnames, shocknames, var, shock)
                0:hor, r.irf[var, :, shock]
            end
            @series begin
                subplot := (var - 1) * K + shock
                seriestype := :line
                seriescolor := :blue
                linestyle := :dash
                label := ""
                0:hor, r.upper[var, :, shock]
            end
            @series begin
                subplot := (var - 1) * K + shock
                seriestype := :hline
                seriescolor := :black
                linestyle := :dot
                label := ""
                [0.0]
            end
        end
    end
end

function _make_title(varnames, shocknames, var, shock)
    vn = varnames !== nothing ? varnames[var] : "Var $var"
    sn = shocknames !== nothing ? shocknames[shock] : "Shock $shock"
    return "$vn ← $sn"
end

# ─── FEVD plot recipe ──────────────────────────────────────────────────────────

@recipe function f(r::FEVDPosteriorResult; varnames = nothing)
    K = size(r.median, 1)
    nhorizons = size(r.median, 3)

    layout := (1, K)
    legend := true
    seriestype := :bar

    for var in 1:K
        @series begin
            subplot := var
            title := varnames !== nothing ? varnames[var] : "Var $var"
            label := permutedims(["Shock $j" for j in 1:K])
            r.median[var, :, end]'  # last horizon
        end
    end
end

# ─── Forecast plot recipe ──────────────────────────────────────────────────────

@recipe function f(r::ForecastResult; varnames = nothing, history = nothing)
    K = size(r.point_no_shocks, 2)
    fhor = size(r.point_no_shocks, 1)

    layout := (1, K)
    legend := false

    for var in 1:K
        if history !== nothing
            @series begin
                subplot := var
                seriescolor := :black
                linewidth := 1.5
                label := "Data"
                title := varnames !== nothing ? varnames[var] : "Var $var"
                1:size(history, 1), history[:, var]
            end
        end
        @series begin
            subplot := var
            seriescolor := :red
            linewidth := 2
            label := "Forecast"
            h_offset = history !== nothing ? size(history, 1) : 0
            (h_offset + 1):(h_offset + fhor), r.point_no_shocks[:, var]
        end
    end
end

# ─── Historical decomposition plot recipe ───────────────────────────────────────

@recipe function f(r::HistDecompResult; varnames = nothing, shocknames = nothing)
    T = size(r.decomposition, 1)
    K_var = size(r.decomposition, 2)
    K_shock = size(r.decomposition, 3)

    layout := (K_var, 1)
    seriestype := :bar
    bar_position := :stack

    for var in 1:K_var
        for shock in 1:K_shock
            @series begin
                subplot := var
                title := varnames !== nothing ? varnames[var] : "Var $var"
                label := shocknames !== nothing && shock <= length(shocknames) ?
                         shocknames[shock] : "Shock $shock"
                1:T, r.decomposition[:, var, shock]
            end
        end
    end
end
