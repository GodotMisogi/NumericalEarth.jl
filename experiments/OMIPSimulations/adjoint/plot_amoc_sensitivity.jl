# %%
# ==============================================================================
# Map of the AMOC objective's sensitivity to the 10-m winds, from a test_omip_adjoint.jl results file.
#
#   julia --project=adjoint/plots adjoint/plot_amoc_sensitivity.jl adjoint/configs/plot_<config>.json
#
# Config keys (paths relative to adjoint/):
#   results  the results file (gradient_u, gradient_v, J)
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

results = jldopen(joinpath(@__DIR__, P["results"]))
gu = dropdims(sum(results["gradient_u"], dims = (3, 4)), dims = (3, 4))
gv = dropdims(sum(results["gradient_v"], dims = (3, 4)), dims = (3, 4))
J = results["J"]
close(results)

# Each 2° ocean cell reads its 4 neighboring 0.5625° atmosphere cells, so the per-cell gradient is dotted.
# 4 × 4 block sums give the response to a uniform perturbation over each 2.25° box instead.
block_sum(g, b) = [sum(view(g, (i-1)*b+1:i*b, (j-1)*b+1:j*b)) for i in 1:size(g, 1)÷b, j in 1:size(g, 2)÷b]
Gu, Gv = block_sum(gu, 4), block_sum(gv, 4)

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
panels = (("∂J/∂u₁₀, zonal wind", Gu, sum(gu)), ("∂J/∂v₁₀, meridional wind", Gv, sum(gv)))
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
