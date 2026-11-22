# %%
# ==============================================================================
# Grid-independent comparison of test_omip_adjoint.jl results files (ECCO4 prescribed-flux ocean): for each
# forcing slice at least 15 days before the window's end, the AMOC response to a uniform unit change of each
# flux over the Atlantic box of plot_lead_sensitivity.jl (the sum of the per-cell gradient), in total and by
# latitude band, and the largest per-area sensitivity. Signs are ECCO4's (heat and freshwater flux positive
# into the ocean).
#
#   julia --project=adjoint/plots adjoint/uniform_response.jl <results.jld2> [<results.jld2> ...]
# ==============================================================================
using JLD2, Printf
const box_λ = (-100.0, 30.0); const box_φ = (-65.0, 75.0)
function load(path)
    R = jldopen(f -> Dict(k => f[k] for k in keys(f)), path)
    g = R["geometry"]; λc, φc = g.λ, g.φ
    λ = [(λc[i, j] + λc[i+1, j] + λc[i+1, j+1] + λc[i, j+1]) / 4 for i in 1:size(λc, 1)-1, j in 1:size(λc, 2)-1]
    φ = [(φc[i, j] + φc[i+1, j] + φc[i+1, j+1] + φc[i, j+1]) / 4 for i in 1:size(φc, 1)-1, j in 1:size(φc, 2)-1]
    λ = mod.(λ .+ 180, 360) .- 180
    mask = .!g.land .& (box_λ[1] .<= λ .<= box_λ[2]) .& (box_φ[1] .<= φ .<= box_φ[2])
    lead = (R["window_seconds"] .- R["slice_times"]) ./ 86400
    return (; R, φ, mask, lead, area = g.area)
end
bands = ((-65.0, -10.0), (-10.0, 10.0), (10.0, 26.5), (26.5, 45.0), (45.0, 75.0))
files = ARGS
data = load.(files)
for key in ("gradient_u", "gradient_v", "gradient_heat_flux", "gradient_freshwater_flux")
    println("== ", key, "  (sum over the box; rows: lead; columns: total, then bands ", bands, ")")
    for d in data
        G = dropdims(d.R[key]; dims = 3)
        for n in sort(findall(>=(15), d.lead); by = n -> d.lead[n])
            g = G[:, :, n]
            total = sum(g[d.mask])
            parts = [sum(g[d.mask .& (b[1] .<= d.φ .< b[2])]) for b in bands]
            peak = maximum(abs, filter(isfinite, (g ./ d.area)[d.mask]))
            @printf("  %3dx%3d lead %3.0f d: total %+.3e | %s | max per area %.2e\n", size(g)..., d.lead[n], total,
                    join((@sprintf("%+.2e", p) for p in parts), " "), peak)
        end
    end
end
