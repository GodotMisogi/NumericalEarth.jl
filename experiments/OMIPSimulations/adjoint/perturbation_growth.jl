# %%
# ==============================================================================
# Growth of small initial perturbations in the forward model: twin runs of the compiled Reactant `time_step!` (the
# step the adjoint differentiates) on the ECCO4 prescribed-flux ocean, one from the unperturbed initial state and
# one per amplitude ε with ε × (unit-variance noise) added to the wet cells' temperature. If the differences grow
# at the adjoint's rate, the adjoint growth is the forward model's own instability; if they stay bounded, the
# adjoint is not the transpose of the forward model's linearization.
#
#   SCRIPT=adjoint/perturbation_growth.jl CONFIG=adjoint/configs/<config>.json sbatch adjoint/omip_adjoint.slurm
#
# Config keys: start_date (ISO date), n_months, Nz, steps, dt, substeps, backend, roll_barotropic_substeps,
# snapshot_hours, amplitudes (K; 0 checks that the twin is bitwise reproducible), seed, output.
# ==============================================================================

using OMIPAdjoint
using OMIPAdjoint: ocean_model, refresh_diagnostics!, restore_state!
using Oceananigans
using Oceananigans.Architectures: ReactantState
using Oceananigans.Grids: inactive_node, halo_size, λnode, φnode, znode
using Oceananigans.TimeSteppers: time_step!
using Reactant
using JSON
using JLD2
using Printf
using Random
using Dates: DateTime, format, now

config_path = length(ARGS) >= 1 ? ARGS[1] : error("usage: perturbation_growth.jl <config.json>")
P = JSON.parsefile(config_path)
n_steps, Δt = P["steps"], Float64(P["dt"])
snapshot_steps = round(Int, P["snapshot_hours"] * 3600 / Δt)
stamp(msg...) = (println("[", format(now(), "HH:MM:SS"), "] ", msg...); flush(stdout))

# %%
##### Model and compiled step

Reactant.set_default_backend(P["backend"])
roll_barotropic_substeps!(P["roll_barotropic_substeps"])
setup = ecco_adjoint_setup(ReactantState(); Nz = P["Nz"], Δt, substeps = P["substeps"], n_months = P["n_months"],
                           start_date = DateTime(P["start_date"]))
model, cpu_grid = setup.model, setup.cpu_grid
ocean = ocean_model(model)
state = snapshot_state(model)
options = adjoint_compile_options()
refresh = @compile compile_options = options refresh_diagnostics!(model)
step = @compile compile_options = options time_step!(model, Δt)
stamp("compiled refresh and step")

Nx, Ny, Nz = size(cpu_grid)
Hx, Hy, Hz = halo_size(cpu_grid)
c, f = Center(), Face()
wet = (T = [!inactive_node(i, j, k, cpu_grid, c, c, c) for i in 1:Nx, j in 1:Ny, k in 1:Nz],
       u = [!inactive_node(i, j, k, cpu_grid, f, c, c) for i in 1:Nx, j in 1:Ny, k in 1:Nz],
       v = [!inactive_node(i, j, k, cpu_grid, c, f, c) for i in 1:Nx, j in 1:Ny, k in 1:Nz])
wet = (; wet..., S = wet.T)
fields = (T = ocean.tracers.T, S = ocean.tracers.S, u = ocean.velocities.u, v = ocean.velocities.v)
snapshot() = map(f -> Array(interior(f)), fields)

# Restore the initial state, add ε × noise to the wet temperature, and run, returning the snapshots.
function twin_run(ε, rng)
    restore_state!(model, state)
    if ε != 0
        T = Array(parent(fields.T))
        T[Hx+1:Hx+Nx, Hy+1:Hy+Ny, Hz+1:Hz+Nz] .+= ε .* randn(rng, Nx, Ny, Nz) .* wet.T
        copyto!(parent(fields.T), T)
    end
    refresh(model)
    snapshots = [snapshot()]
    for n in 1:n_steps
        step(model, Δt)
        n % snapshot_steps == 0 && push!(snapshots, snapshot())
    end
    return snapshots
end

# %%
##### Twin runs

t0 = time()
reference = twin_run(0, Xoshiro(P["seed"]))
stamp(@sprintf("reference run: %d steps in %.1f s", n_steps, time() - t0))
growth = Dict{String, Any}()
for ε in P["amplitudes"]
    perturbed = twin_run(ε, Xoshiro(P["seed"]))
    records = map(eachindex(reference)) do s
        names = keys(fields)
        δ = NamedTuple{names}(map(name -> (perturbed[s][name] .- reference[s][name]) .* wet[name], names))
        rms = NamedTuple{names}(map(name -> sqrt(sum(abs2, δ[name]) / count(wet[name])), names))
        i, j, k = Tuple(argmax(abs.(δ.T)))
        (; hours = (s - 1) * P["snapshot_hours"], rms, max_T = abs(δ.T[i, j, k]),
           λ = λnode(i, j, k, cpu_grid, c, c, c), φ = φnode(i, j, k, cpu_grid, c, c, c), z = znode(i, j, k, cpu_grid, c, c, c))
    end
    growth[string(ε)] = records
    println("ε = $ε K")
    for r in records[1:max(1, length(records) ÷ 20):end]
        @printf("  %6.1f days  rms ΔT %.3e  ΔS %.3e  Δu %.3e  Δv %.3e  max|ΔT| %.3e at %4.0fE %4.0fN %5.0f m\n",
                r.hours / 24, r.rms.T, r.rms.S, r.rms.u, r.rms.v, r.max_T, mod(r.λ + 180, 360) - 180, r.φ, r.z)
    end
    flush(stdout)
end

output_path = joinpath(@__DIR__, P["output"])
mkpath(dirname(output_path))
jldsave(output_path; config = P, growth)
stamp("saved ", output_path)
