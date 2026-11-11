# %%
# ==============================================================================
# Lead-time-resolved AMOC sensitivity to the surface fluxes, after Fig. 1 of Pillar et al. (2016, J. Climate),
# from a test_omip_adjoint.jl results file of the ECCO4 prescribed-flux ocean (gradients with respect to the
# monthly stress, heat flux and freshwater flux slices, plus the surface geometry).
#
#   julia --project=adjoint/plots adjoint/plot_lead_sensitivity.jl adjoint/configs/plot_lead_<config>.json
#
# Config keys (paths relative to adjoint/):
#   results  the results file
#   output   the image to write
#   slices   optional forcing-slice indices to draw, one row each (default: every slice at least 15 days
#            before the end of the window); the row label is the slice's lead before the window's end
#
# Columns: zonal and meridional stress, heat flux and freshwater flux. Values are per unit area (the gradient
# with respect to a cell's flux divided by the cell's area) with Pillar's signs: positive means more eastward
# or northward stress, more upward heat flux or more evaporation raises the AMOC. ECCO4's heat and freshwater
# fluxes are positive into the ocean, so those columns flip. Colors are logarithmic over three decades below
# the power of ten above each column's 99.5th percentile; smaller values are white.
# ==============================================================================

using CairoMakie
using GeoMakie
using JLD2
using JSON
using Printf

config_path = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "configs", "plot_lead_ecco4_nz20_90day_full.json")
P = JSON.parsefile(config_path)

const longitude_range = (-100.0, 30.0)   # Pillar et al.'s Atlantic sector
const latitude_range = (-65.0, 75.0)
const n_decades = 3
const levels_per_decade = 3

R = jldopen(f -> Dict(k => f[k] for k in keys(f)), joinpath(@__DIR__, P["results"]))
geometry = R["geometry"]
lead_days = (R["window_seconds"] .- R["slice_times"]) ./ 86400
slices = haskey(P, "slices") ? Int.(P["slices"]) : sort(findall(>=(15), lead_days); by = n -> lead_days[n])

# (results key, sign to Pillar's convention, column title, units of the per-area sensitivity)
columns = [("gradient_u",                1.0, "Zonal wind stress",               "Sv N⁻¹"),
           ("gradient_v",                1.0, "Meridional wind stress",          "Sv N⁻¹"),
           ("gradient_heat_flux",       -1.0, "Heat flux (upward +)",            "Sv W⁻¹"),
           ("gradient_freshwater_flux", -1.0, "Freshwater flux (evaporation +)", "Sv (kg s⁻¹)⁻¹")]

# %%
##### Cells as triangles: two per quadrilateral of corners, longitudes unwrapped about the first corner and
##### split across the ±180° seam.

function cell_mesh(λc, φc, d)
    points = Point2f[]
    triangles = NTuple{3, Int}[]
    cell_index = Int[]
    linear = LinearIndices(d)
    for j in axes(d, 2), i in axes(d, 1)
        isfinite(d[i, j]) || continue
        λq = (λc[i, j], λc[i+1, j], λc[i+1, j+1], λc[i, j+1])
        φq = (φc[i, j], φc[i+1, j], φc[i+1, j+1], φc[i, j+1])
        λ₀ = mod(λq[1] + 180, 360) - 180
        λu = map(λ -> λ₀ + mod(λ - λq[1] + 180, 360) - 180, λq)
        shifts = [0.0]
        maximum(λu) > 180 && push!(shifts, -360.0)
        minimum(λu) < -180 && push!(shifts, 360.0)
        for shift in shifts
            n = length(points)
            append!(points, Point2f.(clamp.(λu .+ shift, -180, 180), φq))
            push!(triangles, (n + 1, n + 2, n + 3), (n + 1, n + 3, n + 4))
            append!(cell_index, fill(linear[i, j], 4))
        end
    end
    faces = isempty(triangles) ? zeros(Int, 0, 3) : permutedims(reduce(hcat, collect.(triangles)))
    return points, faces, cell_index
end

center(λc, φc) = (λ = [(λc[i, j] + λc[i+1, j] + λc[i+1, j+1] + λc[i, j+1]) / 4 for i in 1:size(λc, 1)-1, j in 1:size(λc, 2)-1],
                  φ = [(φc[i, j] + φc[i+1, j] + φc[i+1, j+1] + φc[i, j+1]) / 4 for i in 1:size(φc, 1)-1, j in 1:size(φc, 2)-1])

# Signed log scale: sign(v) (log₁₀|v| − log₁₀ vmin) / n_decades ∈ [−1, 1]; |v| < vmin is not drawn.
function symlog(v, vmax)
    vmin = vmax / 10.0^n_decades
    (isfinite(v) && abs(v) >= vmin) || return NaN
    return sign(v) * min(1.0, log10(abs(v) / vmin) / n_decades)
end

# %%
##### Figure

λc, φc = geometry.λ, geometry.φ
centers = center(λc, φc)
in_box = (longitude_range[1] .<= mod.(centers.λ .+ 180, 360) .- 180 .<= longitude_range[2]) .&
         (latitude_range[1] .<= centers.φ .<= latitude_range[2])
ocean = .!geometry.land .& in_box
land = map((l, b) -> (l && b) ? 0.0 : NaN, geometry.land, in_box)

n_levels = n_decades * levels_per_decade
base = cgrad(:RdBu)   # 0 → red, 1 → blue
level_colors = vcat([base[0.5 + 0.5 * i / n_levels] for i in n_levels:-1:1],
                    [base[0.5 - 0.5 * i / n_levels] for i in 1:n_levels])
colormap = cgrad(level_colors, 2n_levels; categorical = true)
decade_label(s, e) = s == 0 ? L"0" : (s > 0 ? L"10^{%$e}" : L"-10^{%$e}")

n_rows = length(slices)
fig = Figure(size = (1500, 300 + 330 * n_rows), fontsize = 15)
for (c, (key, sgn, title, units)) in enumerate(columns)
    per_area = sgn .* dropdims(R[key]; dims = 3) ./ geometry.area
    values = sort(filter(isfinite, abs.(per_area[repeat(ocean, 1, 1, size(per_area, 3))])))
    q = values[clamp(round(Int, 0.995 * length(values)), 1, length(values))]
    q > 0 || error("$key is identically zero in the plotted box")
    vmax = 10.0^ceil(log10(q))
    for (row, n) in enumerate(slices)
        d = map((v, o) -> o ? symlog(v, vmax) : NaN, per_area[:, :, n], ocean)
        ax = GeoAxis(fig[row, c]; dest = "+proj=moll +lon_0=-35", limits = (longitude_range, latitude_range),
                     xticks = -90:30:30, yticks = -60:30:60, xticklabelsvisible = false, yticklabelsvisible = false,
                     xgridcolor = RGBAf(0, 0, 0, 0.15), ygridcolor = RGBAf(0, 0, 0, 0.15))
        lp, lf, _ = cell_mesh(λc, φc, land)
        isempty(lf) || mesh!(ax, lp, lf; color = :lightgray, shading = NoShading)
        p, f, index = cell_mesh(λc, φc, d)
        isempty(f) || mesh!(ax, p, f; color = Float32.(d[index]), colormap, colorrange = (-1, 1), shading = NoShading)
        text!(ax, 0.03, 0.03; text = @sprintf("−%d days", round(Int, lead_days[n])), space = :relative,
              font = :bold, fontsize = 15)
        Label(fig[row, c, TopLeft()], "($('a' + (c - 1) * n_rows + row - 1))"; font = :bold, padding = (0, 5, 0, 0))
        row == 1 && (ax.title = title)
    end
    e_max = round(Int, log10(vmax))
    ticks = collect(-1.0:1/n_decades:1.0)
    labels = [decade_label(sign(t), e_max - n_decades + round(Int, abs(t) * n_decades)) for t in ticks]
    Colorbar(fig[n_rows + 1, c]; colormap, limits = (-1, 1), vertical = false, flipaxis = false,
             ticks = (ticks, labels), label = units, width = Relative(0.9))
end
Label(fig[0, :], @sprintf("Sensitivity of the %.0f-day-mean AMOC at 26.5°N (ψ at 1000 m, J = %.2f Sv) to surface fluxes per unit area, by lead",
                          R["average_seconds"] / 86400, R["J"]); fontsize = 18, font = :bold)
rowgap!(fig.layout, 8)
colgap!(fig.layout, 8)

output_path = joinpath(@__DIR__, P["output"])
mkpath(dirname(output_path))
save(output_path, fig)
println("saved ", output_path)
