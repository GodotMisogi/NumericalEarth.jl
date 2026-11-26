# %%
# ==============================================================================
# Grid-scale share of the forcing sensitivities in test_omip_adjoint.jl results: in each 2×2 block of cells, the
# energy of the three alternating components (checker, stripe in i, stripe in j) over the total. White noise gives
# 0.75, a smooth field near 0. For each results file, prints the share of the stress-x, heat-flux and
# freshwater-flux sensitivities in each forcing slice by latitude band of the Atlantic (260–380°E). With a
# forward_health.jl file of the same grid first, also prints the stress-x and heat-flux shares (each slice, Atlantic
# 0–80°N) in convecting blocks (all four columns with mean statically unstable interfaces ≥ 1) and in stable ones (none in any).
#
#   julia +1.12 --project=adjoint adjoint/grid_scale_share.jl [health.jld2] results.jld2 ...
# ==============================================================================

using JLD2
using Printf
using Statistics: mean

bands = ((-35, 0), (0, 22), (22, 32), (32, 45), (45, 80))
in_atlantic(λ) = (260 <= mod(λ, 360)) | (mod(λ, 360) < 20)

blocks(h) = (h[1:end-1, 1:end-1], h[2:end, 1:end-1], h[1:end-1, 2:end], h[2:end, 2:end])

# Share of the alternating energy over the blocks where `mask` holds; land cells are NaN and drop out.
function grid_scale_share(h, mask)
    a, b, c, d = blocks(h)
    alternating = ((a .- b .- c .+ d) .^ 2 .+ (a .- b .+ c .- d) .^ 2 .+ (a .+ b .- c .- d) .^ 2) ./ 16
    mean_part = ((a .+ b .+ c .+ d) ./ 4) .^ 2
    m = mask .& isfinite.(alternating)
    return sum(alternating[m]) / (sum(alternating[m]) + sum(mean_part[m]))
end

# %%
paths = copy(ARGS)
isempty(paths) && error("usage: grid_scale_share.jl [health.jld2] results.jld2 ...")
unstable = jldopen(paths[1]) do file
    haskey(file, "records") || return nothing
    popfirst!(paths)
    mean(Float64.(r.unstable_columns) for r in file["records"])
end

for path in paths
    file = jldopen(path)
    geometry = file["geometry"]
    Nx, Ny = size(geometry.land)
    # λ can carry one more column than the grid (the periodic halo); block centers use the grid's own cells.
    λc, φc = geometry.λ[1:Nx-1, 1:Ny-1], geometry.φ[1:Nx-1, 1:Ny-1]
    println(basename(path))
    for name in ("gradient_u", "gradient_heat_flux", "gradient_freshwater_flux")
        g = file[name]
        for s in 1:size(g, 4)
            h = ifelse.(geometry.land, NaN, g[:, :, 1, s])
            all(iszero, filter(isfinite, h)) && continue   # the slice after the window carries no sensitivity
            shares = [grid_scale_share(h, (south .<= φc .< north) .& in_atlantic.(λc)) for (south, north) in bands]
            @printf("  %-26s slice %d:%s\n", name, s,
                    join([@sprintf("  %3d–%2d°N %.2f", b..., x) for (b, x) in zip(bands, shares)]))
        end
    end
    if unstable !== nothing
        size(unstable) == size(geometry.land) ||
            error("the health file's grid $(size(unstable)) differs from $(basename(path))'s $(size(geometry.land))")
        ua, ub, uc, ud = blocks(unstable)
        atlantic = (0 .<= φc .< 80) .& in_atlantic.(λc)
        for name in ("gradient_u", "gradient_heat_flux"), s in axes(file[name], 4)
            h = ifelse.(geometry.land, NaN, file[name][:, :, 1, s])
            all(iszero, filter(isfinite, h)) && continue
            @printf("  %-18s slice %d, Atlantic 0–80°N: convecting %.2f, stable %.2f\n", name, s,
                    grid_scale_share(h, atlantic .& (min.(ua, ub, uc, ud) .>= 1)),
                    grid_scale_share(h, atlantic .& (max.(ua, ub, uc, ud) .== 0)))
        end
    end
    close(file)
end
