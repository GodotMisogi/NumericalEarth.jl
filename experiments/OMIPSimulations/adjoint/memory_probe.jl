# %%
# ==============================================================================
# Device and host memory of the checkpointed gradient's compiled pieces, to tell whether a window of a given grid
# fits on one GPU. Builds the ECCO4 prescribed-flux ocean on ReactantState, compiles the checkpointed thunks, then
# runs `steps` forward steps and `steps` reverse steps with a full garbage collection before each, and records
# XLA's allocator statistics after each call. `nvidia-smi` shows only XLA's up-front pool reservation, and a
# gradient run's own peak includes garbage up to the half-pool collection; here the peak is live data plus the
# largest call's temporaries. With the reverse step the largest call, the final peak minus the bytes in use before
# the reverse steps is that step's working set.
#
#   SCRIPT=adjoint/memory_probe.jl CONFIG=adjoint/configs/<config>.json sbatch -A ADCIRC adjoint/omip_adjoint.slurm
#
# Config keys: Nx, Ny, Nz, dt, substeps, n_months, eddy_diffusivity, advection, backend, roll_barotropic_substeps,
# frozen_mixing, frozen_buoyancy_gradients (as in test_omip_adjoint.jl, ECCO4 forcing only), plus
#   steps    forward and reverse steps measured (each is timed; the first of each includes its first-call cost)
#   output   the results file (relative to adjoint/)
# ==============================================================================

using OMIPAdjoint
using OMIPAdjoint: restore_state!, device_arrays
using Oceananigans.Architectures: ReactantState
using Enzyme: make_zero
using Reactant
using JSON
using JLD2
using Printf
using Dates: now, format

ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

config_path = length(ARGS) >= 1 ? ARGS[1] : error("usage: memory_probe.jl <config.json>")
P = JSON.parsefile(config_path)
Nx, Ny, Nz = P["Nx"], P["Ny"], P["Nz"]
n_steps, Δt, substeps = P["steps"], Float64(P["dt"]), P["substeps"]
κ_eddy = get(P, "eddy_diffusivity", 800)
advection = Symbol(get(P, "advection", "weno"))
frozen_gradients = get(P, "frozen_buoyancy_gradients", false)

:NVPTX in Reactant.LLVM.backends() ||
    error("Reactant lowers Oceananigans kernels through NVPTX; this Julia's LLVM lacks it (macOS). Run on the GPU cluster.")

stamp(msg...) = (println("[", format(now(), "HH:MM:SS"), "] ", msg...); flush(stdout))
gib(bytes) = bytes / 2^30

results = Dict{String, Any}("config" => P)
output_path = joinpath(@__DIR__, P["output"])
function save_results()
    mkpath(dirname(output_path))
    jldopen(output_path, "w") do file
        for (key, value) in results
            file[key] = value
        end
    end
    return nothing
end

# Collect first, so the statistics read after a call hold no garbage of earlier calls.
function measure(label)
    GC.gc()
    stats = Reactant.XLA.allocatorstats()
    stamp(@sprintf("%-28s in use %7.2f GiB, peak %7.2f GiB, limit %s", label, gib(stats.bytes_in_use),
                   gib(stats.peak_bytes_in_use),
                   stats.bytes_limit === nothing ? "none" : @sprintf("%.2f GiB", gib(stats.bytes_limit))))
    # XLA's peak never resets: it rises only at a call that needs more than every earlier one.
    record = (; label, bytes_in_use = stats.bytes_in_use, peak_bytes_in_use = stats.peak_bytes_in_use,
                bytes_limit = something(stats.bytes_limit, -1))
    push!(get!(results, "memory", NamedTuple[]), record)
    save_results()
    return stats
end

# %%
##### Model

Reactant.set_default_backend(P["backend"])
roll_barotropic_substeps!(P["roll_barotropic_substeps"])
freeze_mixing!(get(P, "frozen_mixing", false))
freeze_buoyancy_gradients!(frozen_gradients)

t0 = time()
setup = ecco_adjoint_setup(ReactantState(); Nx, Ny, Nz, Δt, substeps, n_months = P["n_months"],
                           κ_skew = κ_eddy, κ_symmetric = κ_eddy, advection,
                           materialize_buoyancy_gradients = frozen_gradients)
model = setup.model
obj = OverturningStrength(setup.cpu_grid)
state = snapshot_state(model)
stamp(@sprintf("%d × %d × %d model built in %.1f min", Nx, Ny, Nz, (time() - t0) / 60))

# A checkpoint is a host copy of these arrays (grids and forcing excluded).
arrays = device_arrays(model)
checkpoint_bytes = sum(a -> length(a) * sizeof(eltype(a)), arrays)
results["checkpoint_bytes"] = checkpoint_bytes
results["host_total_bytes"] = Sys.total_memory()
stamp(@sprintf("checkpoint %.2f GiB in %d arrays; host memory %.1f GiB total, %.1f GiB free",
               gib(checkpoint_bytes), length(arrays), gib(Sys.total_memory()), gib(Sys.free_memory())))
measure("after build")

# %%
##### Compiled pieces

t0 = time()
thunks = compile_checkpointed_thunks(model, obj; Δt, compile_options = adjoint_compile_options())
results["compile_seconds"] = time() - t0
stamp(@sprintf("checkpointed thunks compiled in %.1f min", results["compile_seconds"] / 60))
measure("after compile")

# %%
##### Forward and reverse steps

restore_state!(model, state)
thunks.refresh(model)
forward_seconds = Float64[]
for n in 1:n_steps
    t = time(); thunks.step(model, Δt); push!(forward_seconds, time() - t)
    measure("after forward step $n")
end
results["forward_seconds"] = forward_seconds

dmodel = make_zero(model)
before_reverse = measure("shadow model allocated")
reverse_seconds = Float64[]
for n in 1:n_steps
    t = time(); thunks.reverse_step(model, dmodel, Δt); push!(reverse_seconds, time() - t)
    measure("after reverse step $n")
end
results["reverse_seconds"] = reverse_seconds

final = Reactant.XLA.allocatorstats()
working_set = final.peak_bytes_in_use - before_reverse.bytes_in_use
results["reverse_working_set_bytes"] = working_set
results["peak_bytes_in_use"] = final.peak_bytes_in_use
save_results()
stamp(@sprintf("forward step %.2f s, reverse step %.2f s (after the first call of each)",
               n_steps > 1 ? minimum(forward_seconds[2:end]) : forward_seconds[1],
               n_steps > 1 ? minimum(reverse_seconds[2:end]) : reverse_seconds[1]))
stamp(@sprintf("PEAK %.2f GiB; reverse working set %.2f GiB above %.2f GiB of model, shadow and forcing",
               gib(final.peak_bytes_in_use), gib(working_set), gib(before_reverse.bytes_in_use)))
stamp("saved ", output_path)
