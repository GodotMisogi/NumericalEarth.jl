# %%
# ==============================================================================
# Does the ECCO4 prescribed-flux ocean run, and differentiate, on a grid sharded across the GPUs of one process?
# Builds the model on `Distributed(ReactantState(); partition)` and on one device, then for each: the primal window
# objective and the checkpointed gradient. Each stage is attempted separately and its failure reported, so one job
# shows how far sharding gets; where both succeed, J and every forcing gradient are compared.
#
#   CUDA_VISIBLE_DEVICES=0,1 SCRIPT=adjoint/sharding_probe.jl CONFIG=adjoint/configs/<config>.json \
#       sbatch -p gpu-a100 -A ADCIRC adjoint/omip_adjoint.slurm        (Lonestar6: three A100s per node)
#
# Config keys: n_months, Nz, steps, dt, substeps, backend, roll_barotropic_substeps, checkpoints, partition ([Rx, Ry]),
# output; optional Nx, Ny (default 180, 60).
# ==============================================================================

using OMIPAdjoint
using Oceananigans
using Oceananigans.Architectures: ReactantState
using Oceananigans.DistributedComputations: Distributed, Partition
using Reactant
using JSON
using JLD2
using Printf
using Dates: now, format

config_path = length(ARGS) >= 1 ? ARGS[1] : error("usage: sharding_probe.jl <config.json>")
P = JSON.parsefile(config_path)
n_steps, Δt = P["steps"], Float64(P["dt"])
stamp(msg...) = (println("[", format(now(), "HH:MM:SS"), "] ", msg...); flush(stdout))

Reactant.set_default_backend(P["backend"])
roll_barotropic_substeps!(P["roll_barotropic_substeps"])
stamp("devices: ", length(Reactant.devices()), " ", Reactant.devices())

# Runs `f()` and returns its value, or `nothing` after printing the error and the top of its stack trace.
function attempt(f, label)
    t0 = time()
    try
        value = f()
        stamp(@sprintf("%s: done in %.1f s", label, time() - t0))
        return value
    catch error
        stamp(label, ": FAILED")
        println(sprint(showerror, error)[1:min(end, 2000)])
        for frame in first(stacktrace(catch_backtrace()), 25)
            println("    ", sprint(show, frame)[1:min(end, 200)])
        end
        flush(stdout)
        return nothing
    end
end

function probe(arch, label)
    setup = attempt(() -> ecco_adjoint_setup(arch; Nx = get(P, "Nx", 180), Ny = get(P, "Ny", 60), Nz = P["Nz"], Δt,
                                             substeps = P["substeps"], n_months = P["n_months"]), "$label: build")
    setup === nothing && return (; label)
    model = setup.model
    obj = OverturningStrength(setup.cpu_grid)
    state = attempt(() -> snapshot_state(model), "$label: snapshot")
    state === nothing && return (; label)
    primal! = attempt(() -> build_primal(model, state, obj; n_steps, Δt), "$label: compile primal")
    J_primal = primal! === nothing ? nothing : attempt(primal!, "$label: primal")
    evaluate! = attempt(() -> build_checkpointed_gradient(model, state, obj; n_steps, Δt, checkpoints = P["checkpoints"]),
                        "$label: compile gradient")
    gradient = evaluate! === nothing ? nothing : attempt(() -> evaluate!()[1:2], "$label: gradient")
    J_primal === nothing || stamp(@sprintf("%s: primal J %.10f Sv", label, J_primal))
    gradient === nothing || stamp(@sprintf("%s: gradient J %.10f Sv, max|∂J/∂τx| %.6e", label, gradient[1],
                                           maximum(abs, gradient[2].u)))
    return (; label, J_primal, gradient)
end

# %%
# Sharded first: if the job's time runs out, it does so in the single-device reference.
sharded_arch = attempt(() -> Distributed(ReactantState(); partition = Partition(P["partition"]...)), "sharded architecture")
sharded = sharded_arch === nothing ? (; label = "sharded") : probe(sharded_arch, "sharded $(P["partition"])")
single = probe(ReactantState(), "single device")

# %%
comparison = Dict{String, Any}()
if get(single, :J_primal, nothing) !== nothing && get(sharded, :J_primal, nothing) !== nothing
    comparison["J_primal"] = abs(sharded.J_primal - single.J_primal) / abs(single.J_primal)
end
if get(single, :gradient, nothing) !== nothing && get(sharded, :gradient, nothing) !== nothing
    comparison["J_gradient"] = abs(sharded.gradient[1] - single.gradient[1]) / abs(single.gradient[1])
    for name in keys(single.gradient[2])
        a, b = single.gradient[2][name], sharded.gradient[2][name]
        comparison["gradient_$name"] = maximum(abs, b .- a) / maximum(abs, a)
    end
end
for (key, value) in sort(collect(comparison))
    stamp(@sprintf("sharded vs single, relative difference of %s: %.3e", key, value))
end
output_path = joinpath(@__DIR__, P["output"])
mkpath(dirname(output_path))
jldsave(output_path; config = P, comparison, single_J = get(single, :J_primal, nothing),
        sharded_J = get(sharded, :J_primal, nothing))
stamp("saved ", output_path)
