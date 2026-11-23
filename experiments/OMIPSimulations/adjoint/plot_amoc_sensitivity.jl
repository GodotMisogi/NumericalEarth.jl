# %%
# ==============================================================================
# Map of the AMOC objective's sensitivity to the 10-m winds, from a gradient summary written by
# reduce_gradients.jl (run that next to the full results on the cluster; plot locally).
#
#   julia --project=adjoint/plots adjoint/plot_amoc_sensitivity.jl adjoint/configs/plot_<config>.json
#
# Config keys (paths relative to adjoint/):
#   reduced  the reduce_gradients.jl output
#   label    the gradient's label in it
#   output   the image to write
#   title    figure title
#   limits   optional map limits [[west, east], [south, north]] in degrees (default the North Atlantic)
# ==============================================================================

using CairoMakie
using GeoMakie
using JLD2
using JSON
using Printf

config_path = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "configs", "plot_twodegree_nz20_2step_unrolled_gradient.json")
P = JSON.parsefile(config_path)
limits = Tuple(Tuple.(get(P, "limits", [[-100, 10], [0, 60]])))

# %%
##### Gradient, summed over the forcing slices: the response to a wind perturbation held over the window

summary = jldopen(f -> f["gradient/" * P["label"]], joinpath(@__DIR__, P["reduced"]))
Gu, Gv, J = summary.block_u, summary.block_v, summary.J

# Cell centers of the regular 0–360° atmosphere grid, shifted to −180–180° for the projection.
Nx, Ny = size(Gu)
λ = (collect(1:Nx) .- 0.5) .* 360 / Nx
φ = -90 .+ (collect(1:Ny) .- 0.5) .* 180 / Ny
shift = Nx ÷ 2
λ = circshift(ifelse.(λ .> 180, λ .- 360, λ), -shift)
Gu, Gv = circshift(Gu, (-shift, 0)), circshift(Gv, (-shift, 0))

# %%
##### Figure

mSv = 1e3
colorrange = (-1, 1) .* mSv .* max(maximum(abs, Gu), maximum(abs, Gv))
coastlines = GeoMakie.coastlines()

fig = Figure(size = (1500, 720), fontsize = 16)
Label(fig[0, 1:2], P["title"]; fontsize = 20, font = :bold)
panels = (("∂J/∂u₁₀, zonal wind", Gu, summary.sum_u), ("∂J/∂v₁₀, meridional wind", Gv, summary.sum_v))
for (column, (name, G, uniform)) in enumerate(panels)
    ax = GeoAxis(fig[1, column]; dest = "+proj=eqc", limits,
                 title = @sprintf("%s (uniform 1 m/s: %+.3f Sv)", name, uniform))
    heatmap!(ax, λ, φ, mSv .* G; colormap = :balance, colorrange)
    lines!(ax, coastlines; color = :gray35, linewidth = 0.8)
    lines!(ax, [-80, -14], [26.5, 26.5]; color = :black, linestyle = :dash, linewidth = 1.4)
end
Colorbar(fig[1, 3]; colormap = :balance, colorrange, label = "AMOC sensitivity (mSv per m/s per 2.25° box)")
Label(fig[2, 1:2], @sprintf("J = %.4f Sv. Gradient summed over the forcing slices; dashed: the RAPID section, 26.5°N, 80°W–14°W.", J);
      fontsize = 13, color = :gray30)

output_path = joinpath(@__DIR__, P["output"])
mkpath(dirname(output_path))
save(output_path, fig; px_per_unit = 1.5)
println("saved ", output_path)
