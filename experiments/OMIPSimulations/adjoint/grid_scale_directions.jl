# %%
# ==============================================================================
# Directions for finite-difference scans along the grid-scale and smooth parts of a flux gradient: in each
# non-overlapping 2×2 block of wet cells, the alternating part is the gradient minus its block mean (the checker
# and the two stripes of grid_scale_share.jl) and the smooth part is the block mean. Both are orthogonal
# projections, so the gradient's inner product with each unit direction is that part's norm. Writes one results-like
# file per part and flux (`gradient_heat_flux`, `gradient_freshwater_flux`, one of them zero; zero `gradient_u` and
# `gradient_v`; `J`) for the `fd_directions` of a test_omip_adjoint.jl `fd_scan` stage. With a forward_health.jl file
# of the same window, also the alternating part restricted to the blocks whose four columns never convect (no
# statically unstable interface, averaged over the records) and to those whose four columns convect (at least one
# on average), also orthogonal projections.
#
#   julia +1.12 --project=adjoint adjoint/grid_scale_directions.jl adjoint/configs/<config>.json
#
# Config keys:
#   gradient   results file holding the flux gradients (relative to adjoint/)
#   output     path prefix of the direction files (relative to adjoint/)
#   health     optional forward_health.jl results file (relative to adjoint/)
# ==============================================================================

using JLD2
using JSON
using Printf
using Statistics: mean

P = JSON.parsefile(ARGS[1])

# Block means over the wet cells of each non-overlapping 2×2 block, slice by slice; land stays zero.
function smooth_part(g, land)
    s = zero(g)
    Nx, Ny = size(land)
    for n in axes(g, 4), j in 1:2:Ny, i in 1:2:Nx
        block = CartesianIndices((i:min(i + 1, Nx), j:min(j + 1, Ny)))
        wet = [c for c in block if !land[c]]
        isempty(wet) && continue
        m = sum(g[c, 1, n] for c in wet) / length(wet)
        foreach(c -> s[c, 1, n] = m, wet)
    end
    return s
end

# Cells of the non-overlapping 2×2 blocks whose columns all satisfy `predicate` on their mean unstable interfaces.
function blocks_where(predicate, unstable)
    Nx, Ny = size(unstable)
    selected = falses(Nx, Ny)
    for j in 1:2:Ny, i in 1:2:Nx
        block = CartesianIndices((i:min(i + 1, Nx), j:min(j + 1, Ny)))
        all(c -> predicate(unstable[c]), block) && (selected[block] .= true)
    end
    return selected
end

# %%
gradient = JLD2.load(joinpath(@__DIR__, P["gradient"]))
land = gradient["geometry"].land
unstable = haskey(P, "health") ?
    jldopen(f -> mean(Float64.(r.unstable_columns) for r in f["records"]), joinpath(@__DIR__, P["health"])) : nothing
fluxes = ("heat_flux", "freshwater_flux")
for flux in fluxes
    g = gradient["gradient_$flux"]
    smooth = smooth_part(g, land)
    parts = (alternating = g .- smooth, smooth = smooth)
    if unstable !== nothing
        parts = (; parts..., alternating_stable = blocks_where(==(0), unstable) .* parts.alternating,
                 alternating_convecting = blocks_where(>=(1), unstable) .* parts.alternating)
    end
    for (name, d) in pairs(parts)
        @printf("%-16s %-12s |part| %.6e  |part|² / |g|² %.4f  gᵀpart − |part|² %.2e\n", flux, name,
                sqrt(sum(abs2, d)), sum(abs2, d) / sum(abs2, g), sum(g .* d) - sum(abs2, d))
        path = joinpath(@__DIR__, "$(P["output"])_$(flux)_$(name).jld2")
        jldopen(path, "w") do file
            for other in fluxes
                file["gradient_$other"] = other == flux ? d : zero(d)
            end
            file["gradient_u"] = zero(gradient["gradient_u"])
            file["gradient_v"] = zero(gradient["gradient_v"])
            file["J"] = gradient["J"]
        end
        println("  wrote ", path)
    end
end
