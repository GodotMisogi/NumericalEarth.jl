# %%
# ==============================================================================
# Full against frozen-mixing gradients: the finite-difference secant slopes of the AMOC along the full
# gradient's direction (the `fd_scan` stage) against each gradient's prediction there, one panel per window.
#
#   julia --project=adjoint/plots adjoint/plot_frozen_comparison.jl adjoint/configs/plot_frozen_comparison.json
#
# Config keys (paths relative to adjoint/):
#   reduced  the reduce_gradients.jl output
#   panels   [[full label, frozen label, title], ...]; the full label also names the FD scan
#   h_min    smallest step size shown (m/s); below it the jumps in J dominate the secants
#   output   the image to write
# ==============================================================================

using CairoMakie
using JLD2
using JSON

config_path = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "configs", "plot_frozen_comparison.json")
P = JSON.parsefile(config_path)

# %%
##### Figure

blue, orange, gray = colorant"#2a78d6", colorant"#eb6834", colorant"#52514e"

fig = Figure(size = (1300, 480), fontsize = 15)
jldopen(joinpath(@__DIR__, P["reduced"])) do f
    for (column, (full, frozen, title)) in enumerate(P["panels"])
        scan = f["fd_scan/" * full]
        pair = f["pair/$full/$frozen"]
        rows = filter(r -> r.h >= P["h_min"], scan.rows)
        h = [r.h for r in rows]
        central = [(r.Δplus - r.Δminus) / (2r.h) for r in rows]
        forward = [r.Δplus / r.h for r in rows]

        ax = Axis(fig[1, column]; xscale = log10, title, xlabel = "h (m/s along g_full/|g_full|)",
                  ylabel = column == 1 ? "slope along d (Sv per m/s)" : "")
        scatterlines!(ax, h, central; color = gray, markersize = 9, linewidth = 1.5, label = "central secant")
        scatterlines!(ax, h, forward; color = gray, linestyle = :dot, marker = :utriangle, markersize = 9,
                      linewidth = 1.5, label = "forward secant")
        hlines!(ax, pair.reference_along_reference; color = blue, linewidth = 2, label = "full adjoint gᵀd")
        hlines!(ax, pair.other_along_reference; color = orange, linewidth = 2, linestyle = :dash,
                label = "frozen-mixing gᵀd")
        column == 1 && Legend(fig[2, :], ax; orientation = :horizontal, framevisible = false)
    end
end
Label(fig[0, :], "Full vs frozen-mixing adjoint against finite differences of the AMOC"; fontsize = 18, font = :bold)

output_path = joinpath(@__DIR__, P["output"])
mkpath(dirname(output_path))
save(output_path, fig; px_per_unit = 1.5)
println("saved ", output_path)
