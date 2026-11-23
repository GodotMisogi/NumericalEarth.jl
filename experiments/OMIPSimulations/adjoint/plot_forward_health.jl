# %%
# ==============================================================================
# Figure of a forward_health.jl diagnostics file: time series (objective, SST error, grid-scale noise, unstable
# interfaces) and maps of the surface grid-scale temperature residual and CATKE diffusivity at two snapshots.
#
#   julia --project=adjoint/plots adjoint/plot_forward_health.jl adjoint/configs/plot_<config>.json
#
# Config keys (paths relative to adjoint/): results, output, days (two snapshot days for the maps).
# ==============================================================================

using CairoMakie
using JLD2
using JSON

config_path = length(ARGS) >= 1 ? ARGS[1] : error("usage: plot_forward_health.jl <config.json>")
P = JSON.parsefile(config_path)
R = jldopen(f -> (records = f["records"], λ = f["λ"], φ = f["φ"], wet = f["surface_wet"]), joinpath(@__DIR__, P["results"]))

# %%
##### Figure

days = [r.day for r in R.records]
blue, orange, gray = colorant"#2a78d6", colorant"#eb6834", colorant"#52514e"
fig = Figure(size = (1500, 900), fontsize = 14)

series = (("J (Sv)", [r.J for r in R.records]), ("SST − ECCO4, rms (°C)", [r.sst_rms for r in R.records]),
          ("surface T grid-scale residual, rms (K)", [r.noise_rms["T"] for r in R.records]),
          ("statically unstable interfaces", [r.unstable for r in R.records]))
top = fig[1, 1:4] = GridLayout()
for (n, (label, y)) in enumerate(series)
    ax = Axis(top[1, n]; xlabel = "day", title = label)
    lines!(ax, days, Float64.(y); color = blue, linewidth = 2)
end

λ = mod.(R.λ .+ 180, 360) .- 180
for (row, day) in enumerate(P["days"])
    r = R.records[argmin(abs.(days .- day))]
    noise = ifelse.(R.wet, Float64.(r.noise_T), NaN)
    ax = Axis(fig[row + 1, 1:2]; title = "surface T minus 4-neighbor mean (K), day $(round(Int, r.day))",
              xlabel = "longitude", ylabel = "latitude")
    hm = scatter!(ax, vec(λ), vec(R.φ); color = vec(noise), colormap = :balance, colorrange = (-3, 3),
                  marker = :rect, markersize = 6, nan_color = :transparent)
    Colorbar(fig[row + 1, 3], hm; label = "K")
    ax = Axis(fig[row + 1, 4]; title = "surface CATKE κ (m²/s), day $(round(Int, r.day))", xlabel = "longitude")
    κ = ifelse.(R.wet, Float64.(r.κc_surface), NaN)
    scatter!(ax, vec(λ), vec(R.φ); color = vec(log10.(max.(κ, 1e-6))), colormap = :viridis, colorrange = (-5, 0.5),
             marker = :rect, markersize = 6, nan_color = :transparent)
end

output_path = joinpath(@__DIR__, P["output"])
mkpath(dirname(output_path))
save(output_path, fig)
println("saved ", output_path)
