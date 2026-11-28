# %%
# ==============================================================================
# Finite-difference scans along several labeled directions (an `fd_scan` stage with `fd_directions`), one
# panel per direction: the forward, backward and central slopes over the adjoint's gᵀd against the predicted
# response h gᵀd. A switch in the forward model shows as one side leaving 1 while the other stays.
#
#   julia --project=adjoint/plots adjoint/plot_fd_directions.jl adjoint/configs/plot_fd_<config>.json
#
# Config keys (paths relative to adjoint/):
#   results  results file holding the `fd_scan` group, one entry per direction
#   panels   [[label in the file, panel title], ...] in drawing order
#   title    figure title
#   output   the image to write
# ==============================================================================

using CairoMakie
using JLD2
using JSON

P = JSON.parsefile(ARGS[1])
scans = jldopen(f -> Dict(k => f["fd_scan"][k] for k in keys(f["fd_scan"])), joinpath(@__DIR__, P["results"]))

# %%
##### Figure

blue, orange, black = colorant"#2a78d6", colorant"#eb6834", colorant"#222222"
n_columns = 3
n_rows = cld(length(P["panels"]), n_columns)
fig = Figure(size = (1350, 380 * n_rows + 80), fontsize = 15)
for (n, (label, title)) in enumerate(P["panels"])
    haskey(scans, label) || error("no fd_scan entry $label in $(P["results"]); have $(sort(collect(keys(scans))))")
    scan = scans[label]
    signal = [r.h * scan.gd for r in scan.rows]
    row, column = fldmod1(n, n_columns)
    ax = Axis(fig[row, column]; xscale = log10, title, limits = (nothing, (-0.5, 3.5)),
              xticks = ([1e-4, 3e-4, 1e-3, 3e-3, 1e-2], ["10⁻⁴", "3×10⁻⁴", "10⁻³", "3×10⁻³", "10⁻²"]),
              xlabel = row == n_rows ? "predicted response h gᵀd (Sv)" : "",
              ylabel = column == 1 ? "finite difference / gᵀd" : "")
    hlines!(ax, [1.0]; color = :gray70, linewidth = 1)
    for (values, color, name, marker) in (([r.Δplus for r in scan.rows] ./ signal, blue, "+h", :utriangle),
                                          ([-r.Δminus for r in scan.rows] ./ signal, orange, "−h", :dtriangle),
                                          ([1 + r.central for r in scan.rows], black, "central", :circle))
        lines!(ax, signal, values; color, linewidth = name == "central" ? 2.5 : 1.2, label = name)
        scatter!(ax, signal, values; color, marker, markersize = 10)
    end
    n == 1 && axislegend(ax; position = :lt, framevisible = false)
end
Label(fig[0, :], P["title"]; fontsize = 18, font = :bold)

output_path = joinpath(@__DIR__, P["output"])
mkpath(dirname(output_path))
save(output_path, fig; px_per_unit = 1.5)
println("saved ", output_path)
