# %%
# ==============================================================================
# Growth of the adjoint during the reverse sweep: the largest prognostic adjoint (u, v, T, S, η) at each traced
# state, against the model day of that state, for several traced results files of test_omip_adjoint.jl
# (`adjoint_trace_every`). The sweep runs from the right (end of the window) to the left. A curve stops where
# the adjoint is no longer finite.
#
#   julia --project=adjoint/plots adjoint/plot_adjoint_trace.jl adjoint/configs/plot_adjoint_trace.json
#
# Config keys (paths relative to adjoint/): runs ([[results file, label], ...]), output; optional ylims.
# ==============================================================================

using CairoMakie
using JLD2
using JSON

config_path = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "configs", "plot_adjoint_trace.json")
P = JSON.parsefile(config_path)
prognostic = ("velocities.u", "velocities.v", "tracers.T", "tracers.S", "free_surface.η")

# %%
# Day and largest prognostic adjoint of each traced state, back to the first state that is not finite or zero
# (an overflowed sweep records zeros).
function sweep_maximum(path)
    records, Δt = jldopen(f -> (f["adjoint_trace"], f["config"]["dt"]), path)
    days, peaks = Float64[], Float64[]
    for step in sort(unique(r.step for r in records); rev = true)
        values = [r.max_abs for r in records if r.step == step && r.name in prognostic]
        (all(isfinite, values) && maximum(values) > 0) || break
        push!(days, step * Δt / 86400); push!(peaks, maximum(values))
    end
    return days, peaks
end

# %%
##### Figure

colors = [colorant"#2a78d6", colorant"#eb6834", colorant"#1baf7a", colorant"#eda100", colorant"#e87ba4"]
fig = Figure(size = (900, 520), fontsize = 15)
ax = Axis(fig[1, 1]; xlabel = "model day (the reverse sweep runs from right to left)",
          ylabel = "largest prognostic adjoint", yscale = log10)
for (n, (path, label)) in enumerate(P["runs"])
    days, peaks = sweep_maximum(joinpath(@__DIR__, path))
    lines!(ax, days, peaks; color = colors[n], label, linewidth = 2)
end
ylims!(ax, get(P, "ylims", [1e0, 1e40])...)
axislegend(ax; position = :lt, framevisible = false)

output_path = joinpath(@__DIR__, P["output"])
mkpath(dirname(output_path))
save(output_path, fig; px_per_unit = 2)
println("saved ", output_path)
