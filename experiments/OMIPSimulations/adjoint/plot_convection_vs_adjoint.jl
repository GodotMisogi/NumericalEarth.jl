# %%
# ==============================================================================
# Where an adjoint grows against where its forward state is statically unstable: for each pair of a
# forward_health.jl file and a test_omip_adjoint.jl results file (ECCO4 forcing), a map of the adjoint's heat-flux
# sensitivity in one forcing slice and a map of the statically unstable interfaces per column, averaged over the
# health snapshots in a day range. Prints, for each pair, the fraction of the adjoint's largest cells that lie in
# unstable columns against the fraction of all wet columns that are unstable.
#
#   julia --project=adjoint/plots adjoint/plot_convection_vs_adjoint.jl adjoint/configs/plot_convection_vs_adjoint.json
#
# Config keys (paths relative to adjoint/): pairs ([[health file, adjoint file, label], ...]), slice (forcing-slice
# index), days ([first, last] health day), top (number of largest adjoint cells), output.
# ==============================================================================

using CairoMakie
using JLD2
using JSON
using Printf
using Statistics: mean

config_path = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "configs", "plot_convection_vs_adjoint.json")
P = JSON.parsefile(config_path)
first_day, last_day = P["days"]

# %%
##### Figure

pairs = P["pairs"]
fig = Figure(size = (1500, 380 * length(pairs) + 60), fontsize = 14)
for (row, (health_path, adjoint_path, label)) in enumerate(pairs)
    H = jldopen(f -> (records = f["records"], λ = f["λ"], φ = f["φ"], wet = f["surface_wet"]), joinpath(@__DIR__, health_path))
    G = jldopen(f -> f["gradient_heat_flux"][:, :, 1, P["slice"]], joinpath(@__DIR__, adjoint_path))
    chosen = [r for r in H.records if first_day <= r.day <= last_day]
    unstable = mean(Float64.(r.unstable_columns) for r in chosen)
    λ = vec(mod.(H.λ .+ 180, 360) .- 180)
    φ = vec(H.φ)

    magnitude = ifelse.(H.wet .& isfinite.(G), abs.(G), NaN)
    order = sortperm(vec(map(x -> isfinite(x) ? x : -Inf, magnitude)); rev = true)
    top = order[1:P["top"]]
    in_unstable = count(>(0), vec(unstable)[top]) / length(top)
    wet_unstable = count(>(0), unstable[H.wet]) / count(H.wet)
    @printf("%-28s top-%d adjoint cells in unstable columns: %.0f %% (all wet columns: %.0f %%)\n",
            label, P["top"], 100in_unstable, 100wet_unstable)

    ax = Axis(fig[row, 1]; title = "$label: log₁₀|∂J/∂Q|, slice $(P["slice"])", ylabel = "latitude")
    hm = scatter!(ax, λ, φ; color = vec(log10.(magnitude)), colormap = :magma, marker = :rect, markersize = 6,
                  nan_color = :transparent)
    scatter!(ax, λ[top], φ[top]; color = :transparent, strokecolor = :cyan, strokewidth = 1.2, markersize = 9)
    Colorbar(fig[row, 2], hm)
    ax = Axis(fig[row, 3]; title = @sprintf("unstable interfaces per column, days %d–%d", first_day, last_day))
    hm = scatter!(ax, λ, φ; color = vec(ifelse.(H.wet, unstable, NaN)), colormap = :viridis, colorrange = (0, 4),
                  marker = :rect, markersize = 6, nan_color = :transparent)
    scatter!(ax, λ[top], φ[top]; color = :transparent, strokecolor = :red, strokewidth = 1.2, markersize = 9)
    Colorbar(fig[row, 4], hm)
end
Label(fig[0, :], "Adjoint growth one month before the end of a 1-year window against the forward state's static instability";
      fontsize = 17, font = :bold)

output_path = joinpath(@__DIR__, P["output"])
mkpath(dirname(output_path))
save(output_path, fig)
println("saved ", output_path)
