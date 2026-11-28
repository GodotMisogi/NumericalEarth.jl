# %%
# ==============================================================================
# Directions for finite-difference scans along a flux gradient restricted to regions: per region, the gradient on
# the wet cells whose centers lie in a longitude-latitude box and whose columns are convecting (at least one
# statically unstable interface, averaged over the records of a forward_health.jl file), stable (none) or either.
# A restriction is an orthogonal projection, so the gradient's inner product with the unit direction is the
# restricted part's norm. Writes one results-like file per region (`gradient_heat_flux`, `gradient_freshwater_flux`,
# one of them zero; zero `gradient_u` and `gradient_v`; `J`) for the `fd_directions` of a test_omip_adjoint.jl
# `fd_scan` stage.
#
#   julia +1.12 --project=adjoint adjoint/region_directions.jl adjoint/configs/<config>.json
#
# Config keys (paths relative to adjoint/):
#   gradient   results file holding the flux gradients
#   output     path prefix of the direction files
#   health     forward_health.jl results file of the same window (needed for `columns` other than "all")
#   regions    {label: {"flux": "heat_flux" | "freshwater_flux", "longitudes": [west, east] (°E, −180–180),
#              "latitudes": [south, north], "columns": "all" | "convecting" | "stable"}}
# ==============================================================================

using JLD2
using JSON
using Printf
using Statistics: mean

P = JSON.parsefile(ARGS[1])

# %%
gradient = JLD2.load(joinpath(@__DIR__, P["gradient"]))
geometry = gradient["geometry"]
Nx, Ny = size(geometry.land)
wrap(λ) = mod(λ + 180, 360) - 180
# Cell centers from the corners; longitudes as offsets from one corner, so cells across the dateline stay whole.
λ = [wrap(geometry.λ[i, j] + sum(wrap(geometry.λ[i + a, j + b] - geometry.λ[i, j]) for a in 0:1, b in 0:1) / 4) for i in 1:Nx, j in 1:Ny]
φ = [sum(geometry.φ[i + a, j + b] for a in 0:1, b in 0:1) / 4 for i in 1:Nx, j in 1:Ny]
unstable = haskey(P, "health") ?
    jldopen(f -> mean(Float64.(r.unstable_columns) for r in f["records"]), joinpath(@__DIR__, P["health"])) : nothing
column_filters = Dict("all" => u -> true, "convecting" => u -> u >= 1, "stable" => u -> u == 0)
fluxes = ("heat_flux", "freshwater_flux")

for (label, region) in P["regions"]
    flux, (west, east), (south, north) = region["flux"], region["longitudes"], region["latitudes"]
    columns = get(region, "columns", "all")
    columns == "all" || unstable !== nothing || error("region $label selects $columns columns; give `health`")
    keep = column_filters[columns]
    selected = [!geometry.land[i, j] && west <= λ[i, j] <= east && south <= φ[i, j] <= north &&
                (columns == "all" || keep(unstable[i, j])) for i in 1:Nx, j in 1:Ny]
    g = gradient["gradient_$flux"]
    d = selected .* g
    @printf("%-28s %-16s %5d cells  |part|² / |g|² %.4f\n", label, flux, count(selected), sum(abs2, d) / sum(abs2, g))
    path = joinpath(@__DIR__, "$(P["output"])_$label.jld2")
    jldopen(path, "w") do file
        for other in fluxes
            file["gradient_$other"] = other == flux ? d : zero(gradient["gradient_$other"])
        end
        file["gradient_u"] = zero(gradient["gradient_u"])
        file["gradient_v"] = zero(gradient["gradient_v"])
        file["J"] = gradient["J"]
    end
    println("  wrote ", path)
end
