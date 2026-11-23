# %%
# ==============================================================================
# Finite-difference scans of the AMOC objective along a saved gradient (the `fd_scan` stage of
# test_omip_adjoint.jl), one column per window length.
#
#   julia --project=adjoint/plots adjoint/plot_fd_scans.jl adjoint/configs/plot_fd_scans.json
#
# Config keys (paths relative to adjoint/):
#   scans   results files holding `fd_scan`
#   labels  one column title per scan
#   output  the image to write
# ==============================================================================

using CairoMakie
using JLD2
using JSON

config_path = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "configs", "plot_fd_scans.json")
P = JSON.parsefile(config_path)

# %%
##### Figure

blue, orange, gray = colorant"#2a78d6", colorant"#eb6834", colorant"#8a8986"
floor_value = 1e-14   # exact zeros on a log axis

fig = Figure(size = (1500, 820), fontsize = 15)
for (column, (path, label)) in enumerate(zip(P["scans"], P["labels"]))
    scan = jldopen(f -> f["fd_scan"], joinpath(@__DIR__, path))
    h = [r.h for r in scan.rows]
    clip(x) = max.(abs.(x), floor_value)

    top = Axis(fig[1, column]; xscale = log10, yscale = log10, title = label,
               xlabel = "h (m/s along g/|g|)", ylabel = column == 1 ? "|FD slope / gᵀd − 1|" : "")
    lines!(top, h, clip([r.central for r in scan.rows]); color = blue, linewidth = 2, label = "central")
    scatter!(top, h, clip([r.central for r in scan.rows]); color = blue, markersize = 8)
    lines!(top, h, clip([r.forward for r in scan.rows]); color = orange, linewidth = 2, label = "forward")
    scatter!(top, h, clip([r.forward for r in scan.rows]); color = orange, markersize = 8)

    # Residuals of the linear prediction on each side: a jump in f shows as a plateau that ignores h.
    bottom = Axis(fig[2, column]; xscale = log10, yscale = log10,
                  xlabel = "h (m/s along g/|g|)", ylabel = column == 1 ? "|f(x₀ ± h d) − f₀ ∓ h gᵀd| (Sv)" : "")
    lines!(bottom, h, h .* abs(scan.gd); color = gray, linestyle = :dash, linewidth = 1.5, label = "h gᵀd (signal)")
    lines!(bottom, h, clip([r.Δplus - r.h * scan.gd for r in scan.rows]); color = blue, linewidth = 2, label = "+h")
    scatter!(bottom, h, clip([r.Δplus - r.h * scan.gd for r in scan.rows]); color = blue, markersize = 8)
    lines!(bottom, h, clip([r.Δminus + r.h * scan.gd for r in scan.rows]); color = orange, linewidth = 2, label = "−h")
    scatter!(bottom, h, clip([r.Δminus + r.h * scan.gd for r in scan.rows]); color = orange, markersize = 8)

    if column == 1
        axislegend(top; position = :lb, framevisible = false)
        axislegend(bottom; position = :lt, framevisible = false)
    end
end
Label(fig[0, :], "Finite differences of the AMOC at 26.5°N along the adjoint gradient (2° OMIP, Nz = 20, Δt = 2 h)";
      fontsize = 19, font = :bold)

output_path = joinpath(@__DIR__, P["output"])
mkpath(dirname(output_path))
save(output_path, fig; px_per_unit = 1.5)
println("saved ", output_path)
