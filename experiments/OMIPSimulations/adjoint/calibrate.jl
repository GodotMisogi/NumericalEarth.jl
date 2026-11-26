# %%
# ==============================================================================
# Twin calibration of the GM/Redi diffusivities of the ECCO4 prescribed-flux ocean: synthetic observations of the
# window-mean temperature and salinity are made with the `truth` diffusivities, then BFGS recovers them from
# `initial` with the adjoint gradient of the misfit (`build_eddy_calibration`). With several `windows` (start dates)
# each MPI rank runs one window on its own GPU; the ranks sum their misfits and gradients, and all run the same
# BFGS on the sums.
#
#   CONFIG=adjoint/configs/<config>.json sbatch -N <length(windows)> adjoint/calibrate_mpi.slurm
#
# Config keys: n_months, Nx, Ny (default 180, 60), Nz, steps, dt, substeps, backend, roll_barotropic_substeps,
# checkpoints, average_days, output; optional frozen_mixing, frozen_buoyancy_gradients, advection, and windows (ISO
# start dates, one per MPI rank; default ["1993-01-01"]).
# truth, initial: [κ_skew, κ_symmetric] (m² s⁻¹). observed_levels (default Nz), observation_stride (default 1),
# sigma_T (K), sigma_S (g kg⁻¹): the observations. stages: "check" (central differences of the misfit in log κ,
# step check_h, a number or a list, against the adjoint) and/or "calibrate" (iterations, initial_step, gtol).
# ==============================================================================

using OMIPAdjoint
using Oceananigans
using Oceananigans.Architectures: ReactantState
using Reactant
using JSON
using JLD2
using MPI
using Printf
using Dates: now, format, DateTime

ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"   # the WOA download goes through DataDeps, which otherwise prompts

# Thread level `:multiple`, which MPICH needs to run a progress thread.
MPI.Init(threadlevel = :multiple)
comm = MPI.COMM_WORLD
rank, nranks = MPI.Comm_rank(comm), MPI.Comm_size(comm)

config_path = length(ARGS) >= 1 ? ARGS[1] : error("usage: calibrate.jl <config.json>")
P = JSON.parsefile(config_path)
windows = get(P, "windows", ["1993-01-01"])
length(windows) == nranks || error("$(length(windows)) windows need as many MPI ranks, got $nranks")
stages = Symbol.(strip.(split(P["stages"], ",")))
n_steps, Δt = P["steps"], Float64(P["dt"])
average_steps = round(Int, P["average_days"] * 86400 / Δt)
stamp(msg...) = (println("[", format(now(), "HH:MM:SS"), "] rank $rank: ", msg...); flush(stdout))
stamp₀(msg...) = rank == 0 && stamp(msg...)
# Sum over the windows, broadcast from rank 0 so that every rank's BFGS sees the same numbers.
total(v) = MPI.bcast(MPI.Allreduce(v, +, comm), 0, comm)
# The rank's own evaluation time, with its GC share and host allocation, then its wait in the reduction (the slowest
# rank's lead).
function reduced(label, evaluate)
    timed = @timed evaluate()
    t0 = time()
    value = map(total, timed.value)
    stamp(@sprintf("%s: evaluation %.1f s (GC %.1f s, %.1f GiB allocated), reduction %.1f s", label, timed.time,
                   timed.gctime, timed.bytes / 2^30, time() - t0))
    return value
end

results = Dict{String, Any}("config" => P)
output_path = joinpath(@__DIR__, P["output"])
function save_results()
    rank == 0 || return nothing
    mkpath(dirname(output_path))
    jldopen(output_path, "w") do file
        for (key, value) in results
            file[key] = value
        end
    end
    return nothing
end

# %%
##### Model and compiled misfit

:NVPTX in Reactant.LLVM.backends() ||
    error("Reactant lowers Oceananigans kernels through NVPTX; this Julia's LLVM lacks it (macOS). Run on the GPU cluster.")
Reactant.set_default_backend(P["backend"])
roll_barotropic_substeps!(P["roll_barotropic_substeps"])
freeze_mixing!(get(P, "frozen_mixing", false))
frozen_gradients = freeze_buoyancy_gradients!(get(P, "frozen_buoyancy_gradients", false))

t0 = time()
setup = ecco_adjoint_setup(ReactantState(); Nx = get(P, "Nx", 180), Ny = get(P, "Ny", 60), Nz = P["Nz"], Δt,
                           substeps = P["substeps"], n_months = P["n_months"],
                           advection = Symbol(get(P, "advection", "weno")),
                           materialize_buoyancy_gradients = frozen_gradients, eddy_diffusivity_fields = true,
                           start_date = DateTime(windows[rank + 1]))
model = setup.model
state = snapshot_state(model)
observed = observed_cells(setup.cpu_grid; levels = get(P, "observed_levels", P["Nz"]),
                          stride = get(P, "observation_stride", 1))
stamp(@sprintf("window from %s built in %.1f s; %d observed cells", windows[rank + 1], time() - t0, count(observed)))

# Placeholder observations: replaced by the truth run's means below, before any misfit is evaluated.
Nx, Ny, Nz = size(observed)
observations = TracerMisfit(observed, zeros(Nx, Ny, Nz), zeros(Nx, Ny, Nz), Float64(P["sigma_T"]), Float64(P["sigma_S"]))
t0 = time()
calibration = build_eddy_calibration(model, state, observations; Δt, n_steps, average_steps,
                                     checkpoints = P["checkpoints"])
results["compile_seconds"] = time() - t0
stamp(@sprintf("compiled in %.1f min", results["compile_seconds"] / 60))

# %%
##### Synthetic observations from the truth

x_truth = log.(Float64.(P["truth"]))
t0 = time()
T°, S° = calibration.means!(x_truth)
observations.T .= T°
observations.S .= S°
results["forward_seconds"] = time() - t0
stamp(@sprintf("truth window in %.1f s: observed mean T %.4f, S %.4f", results["forward_seconds"],
               sum(T°[observed]) / count(observed), sum(S°[observed]) / count(observed)))
x0 = log.(Float64.(P["initial"]))
misfit!(x) = reduced("misfit", () -> calibration.misfit!(x))
gradient!(x) = reduced("gradient", () -> calibration.gradient!(x))

# %%
##### Gradient check: central differences of the misfit in log κ

if :check in stages
    h = Float64.(vcat(P["check_h"]))
    J0, g0 = gradient!(x0)
    # fd[k, i]: step h[k] in log κ_i
    fd = [(misfit!(x0 .+ hₖ .* e) - misfit!(x0 .- hₖ .* e)) / 2hₖ for hₖ in h, e in ([1.0, 0.0], [0.0, 1.0])]
    for (i, name) in enumerate(("κ_skew", "κ_symmetric")), (k, hₖ) in enumerate(h)
        stamp₀(@sprintf("∂J/∂log %s: adjoint %+.6e, central difference %+.6e (h = %.4f), ratio %.4f", name, g0[i],
                       fd[k, i], hₖ, g0[i] / fd[k, i]))
    end
    results["check"] = (; x0, J0, adjoint = g0, central_difference = fd, h)
    save_results()
end

# %%
##### Calibration

if :calibrate in stages
    J_truth = misfit!(x_truth)
    stamp₀(@sprintf("misfit at the truth %.3e (must be 0)", J_truth))
    records = NamedTuple[]
    callback(r) = begin
        push!(records, r)
        κ = exp.(r.x)
        stamp₀(@sprintf("iteration %2d: κ_skew %7.1f, κ_symmetric %7.1f m² s⁻¹, J %.6e, |∂J/∂log κ| %.3e, step %.3f, %d evaluations",
                       r.iteration, κ[1], κ[2], r.f, sqrt(sum(abs2, r.g)), r.step, r.evaluations))
        results["records"] = copy(records)
        save_results()
        # Device memory before and after a full collection: growth that survives it is held, not garbage.
        stats = Reactant.XLA.allocatorstats()
        GC.gc()
        stamp(@sprintf("device memory in use %.2f GiB (peak %.2f, limit %.2f), %.2f GiB after GC",
                       stats.bytes_in_use / 2^30, stats.peak_bytes_in_use / 2^30,
                       something(stats.bytes_limit, 0) / 2^30, Reactant.XLA.allocatorstats().bytes_in_use / 2^30))
    end
    bfgs_minimize(gradient!, misfit!, x0; iterations = P["iterations"],
                  initial_step = P["initial_step"], gtol = get(P, "gtol", 0.0), callback)
    κ = exp.(records[end].x)
    stamp₀(@sprintf("recovered κ_skew %.1f (truth %.1f), κ_symmetric %.1f (truth %.1f)", κ[1], P["truth"][1], κ[2],
                   P["truth"][2]))
end

stamp₀("saved ", output_path)
MPI.Finalize()
