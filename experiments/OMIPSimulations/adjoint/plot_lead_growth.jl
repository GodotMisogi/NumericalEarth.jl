# %%
# ==============================================================================
# Growth of the adjoint with lead: the largest |∂J/∂τx| over cells for each forcing slice, against the slice's
# lead before the end of the window, for several ECCO4 results files of test_omip_adjoint.jl. Non-finite slices
# are marked at the top of the axis.
#
#   julia --project=adjoint/plots adjoint/plot_lead_growth.jl adjoint/configs/plot_lead_growth.json
#
# Config keys (paths relative to adjoint/): runs ([[results file, label], ...]), output.
# ==============================================================================

using CairoMakie
using JLD2
using JSON

config_path = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "configs", "plot_lead_growth.json")
P = JSON.parsefile(config_path)

# %%
##### Figure

colors = [colorant"#2a78d6", colorant"#eb6834", colorant"#1baf7a", colorant"#eda100", colorant"#e87ba4"]
fig = Figure(size = (900, 520), fontsize = 15)
ax = Axis(fig[1, 1]; xlabel = "lead before the end of the window (days)", ylabel = "max |∂J/∂τx| (Sv per N m⁻²)",
          yscale = log10, title = "Adjoint growth with lead (ECCO4 forcing, frozen mixing, 30-day-mean AMOC)")
ceiling = 1e180   # where non-finite slices are drawn
for (n, (path, label)) in enumerate(P["runs"])
    lead, peak = jldopen(joinpath(@__DIR__, path)) do f
        g = f["gradient_u"]
        lead = (f["window_seconds"] .- f["slice_times"]) ./ 86400
        peak = [all(!isfinite, g[:, :, 1, k]) || any(isnan, g[:, :, 1, k]) ? ceiling :
                maximum(abs, filter(isfinite, g[:, :, 1, k])) for k in axes(g, 4)]
        lead, peak
    end
    keep = (lead .>= -1) .& (peak .> 0)
    scatterlines!(ax, lead[keep], peak[keep]; color = colors[n], label, markersize = 9, linewidth = 2)
end
hlines!(ax, ceiling; color = :gray60, linestyle = :dash)
text!(ax, 5, ceiling; text = "NaN", align = (:left, :top), color = :gray40, fontsize = 13)
ylims!(ax, 1e-2, 1e200)
axislegend(ax; position = :rb, framevisible = false)

output_path = joinpath(@__DIR__, P["output"])
mkpath(dirname(output_path))
save(output_path, fig; px_per_unit = 2)
println("saved ", output_path)
