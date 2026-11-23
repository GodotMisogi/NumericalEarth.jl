# %%
# ==============================================================================
# Stored against inline buoyancy gradients on the CPU ECCO4 prescribed-flux ocean (tripolar grid). Builds the
# model twice, with `materialize_buoyancy_gradients` off and on, compares the stored ∂xᵣb, ∂yᵣb, ∂zb with the
# inline functions at every index the stencil reaches (halos included), then steps both models side by side
# and compares u, v, T, S and η after each step. The two forward models agree to rounding unless the stored
# gradients differ from the inline ones somewhere, or at some time, the model reads them.
#
#   SCRIPT=adjoint/buoyancy_gradient_check.jl CONFIG=adjoint/configs/<config>.json sbatch adjoint/omip_adjoint.slurm
#
# Config keys: n_months, Nz, steps, dt, substeps, output; optional start_date (ISO date).
# ==============================================================================

using OMIPAdjoint
using OMIPAdjoint: ocean_model, refresh_diagnostics!
using Oceananigans
using Oceananigans.BuoyancyFormulations: ∂xᵣ_b, ∂yᵣ_b, ∂z_b
using Oceananigans.TimeSteppers: time_step!
using JSON
using JLD2
using Printf
using Dates: DateTime, format, now

config_path = length(ARGS) >= 1 ? ARGS[1] : error("usage: buoyancy_gradient_check.jl <config.json>")
P = JSON.parsefile(config_path)
n_steps, Δt = P["steps"], Float64(P["dt"])
date_kwargs = haskey(P, "start_date") ? (; start_date = DateTime(P["start_date"])) : (;)
stamp(msg...) = (println("[", format(now(), "HH:MM:SS"), "] ", msg...); flush(stdout))

setup(materialize) = ecco_adjoint_setup(CPU(); Nz = P["Nz"], Δt, substeps = P["substeps"], n_months = P["n_months"],
                                        materialize_buoyancy_gradients = materialize, date_kwargs...)
inline_setup, stored_setup = setup(false), setup(true)
# The setup writes T and S after building the ocean; refresh the diagnostics (stored gradients included) as the
# adjoint's windows do.
foreach(s -> refresh_diagnostics!(s.model), (inline_setup, stored_setup))
inline_ocean, stored_ocean = ocean_model(inline_setup.model), ocean_model(stored_setup.model)
grid = inline_ocean.grid
Nx, Ny, Nz = size(grid)
stamp("built both CPU models; Nx, Ny, Nz = $Nx, $Ny, $Nz")

# %%
# Where the stored and inline values of one gradient differ, by interior (1:Nx, 1:Ny, 1:Nz) and halo indices.
# The outer two layers of each halo are skipped: the inline stencil reads one or two cells beyond its index.
function gradient_mismatch(name, inline_function, stored)
    data = stored.data
    ranges = map(r -> (first(r) + 2):(last(r) - 2), axes(data))
    scale = 0.0
    found = Dict("interior" => Tuple{Float64, Int, Int, Int}[], "halo" => Tuple{Float64, Int, Int, Int}[])
    for k in ranges[3], j in ranges[2], i in ranges[1]
        a = inline_function(i, j, k, grid, inline_ocean.buoyancy, inline_ocean.tracers)
        isfinite(a) && (scale = max(scale, abs(a)))
        Δ = abs(a - data[i, j, k])
        Δ > 0 || continue
        region = (1 <= i <= Nx && 1 <= j <= Ny && 1 <= k <= Nz) ? "interior" : "halo"
        push!(found[region], (Δ, i, j, k))
    end
    for region in ("interior", "halo")
        d = found[region]
        if isempty(d)
            @printf("  %-6s %-8s identical\n", name, region)
            continue
        end
        Δmax, i, j, k = d[argmax(first.(d))]
        rows = sort(unique(x[3] for x in d)); levels = sort(unique(x[4] for x in d))
        @printf("  %-6s %-8s %7d differ, max |Δ| %.3e (%.1e of max |∂b|) at (%d, %d, %d); rows j %s; levels k %s\n",
                name, region, length(d), Δmax, Δmax / scale, i, j, k, short(rows), short(levels))
    end
    return (; name, scale, interior = found["interior"], halo = found["halo"])
end

short(v) = length(v) <= 12 ? string(v) : string(first(v, 6), " … ", last(v, 6), " (", length(v), ")")

function compare_gradients(label)
    stamp("stored vs inline buoyancy gradients, ", label)
    g = stored_ocean.buoyancy.gradients
    return [gradient_mismatch("∂xᵣb", ∂xᵣ_b, g.∂xᵣ_b), gradient_mismatch("∂yᵣb", ∂yᵣ_b, g.∂yᵣ_b),
            gradient_mismatch("∂zb", ∂z_b, g.∂z_b)]
end

# %%
# Largest interior difference of each prognostic field between the two models.
prognostic(om) = (u = om.velocities.u, v = om.velocities.v, T = om.tracers.T, S = om.tracers.S,
                  η = om.free_surface.displacement)

function compare_states(step)
    records = map(keys(prognostic(inline_ocean))) do name
        a = interior(prognostic(inline_ocean)[name]); b = interior(prognostic(stored_ocean)[name])
        Δ = abs.(a .- b)
        I = argmax(Δ)
        (; step, name, max_abs = Δ[I], i = I[1], j = I[2], k = I[3], n_differ = count(>(0), Δ),
           rows = sort(unique(c[2] for c in CartesianIndices(Δ) if Δ[c] > 0)))
    end
    line = join((@sprintf("%s %.2e (%d at %d,%d,%d)", r.name, r.max_abs, r.n_differ, r.i, r.j, r.k) for r in records), "  ")
    stamp("step $step: ", line)
    return records
end

# %%
initial = compare_gradients("initial state")
states = collect(compare_states(0))
for n in 1:n_steps
    time_step!(inline_setup.model, Δt)
    time_step!(stored_setup.model, Δt)
    append!(states, compare_states(n))
end
final = compare_gradients("after $n_steps steps")
for r in filter(r -> r.step == n_steps && r.n_differ > 0, states)
    println("  ", r.name, " rows j differing after $n_steps steps: ", short(r.rows))
end

output_path = joinpath(@__DIR__, P["output"])
mkpath(dirname(output_path))
jldsave(output_path; config = P, initial, final, states)
stamp("saved ", output_path)
