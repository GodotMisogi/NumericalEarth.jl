# %%
# ==============================================================================
# Does a separately compiled `time_step!`, called once per step, reproduce the compiled window?
# `build_checkpointed_gradient` steps that way; its objective differed from the primal window's by 7.5e-8
# (relative) at 2 steps (job 3565170), where the fused gradient window agrees to 4e-12.
#
#   julia --project=adjoint adjoint/step_consistency.jl adjoint/configs/step_consistency_nz20.json
#
# For n = 1 … `steps`: run the compiled n-step window, keep every device array of the model, then repeat the
# same n steps with the compiled refresh and step and compare the arrays one by one. Also lists the
# host-side numbers in the model (values tracing could bake into a compiled step) whenever they change.
# ==============================================================================

using OMIPAdjoint
using OMIPAdjoint: device_arrays, refresh_diagnostics!, restore_state!, host_number, with_while_unroll_threshold,
                   skip_in_checkpoint
using Oceananigans
using Oceananigans.Architectures: ReactantState
using Oceananigans.TimeSteppers: time_step!
using Reactant
using JSON
using Printf
using Dates: now, format

ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

P = JSON.parsefile(length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "configs", "step_consistency_nz20.json"))
Nz, n_steps, Δt, substeps = P["Nz"], P["steps"], Float64(P["dt"]), P["substeps"]

stamp(msg...) = (println("[", format(now(), "HH:MM:SS"), "] ", msg...); flush(stdout))

# Paths of the arrays `device_arrays` returns, and of every host number, by the same walk.
function walk!(found, visited, x, path)
    T = typeof(x)
    if x isa Reactant.ConcreteRNumber
        push!(found.numbers, path => host_number(x))
        return found
    elseif x isa Number
        push!(found.numbers, path => x)
        return found
    end
    (isbitstype(T) || skip_in_checkpoint(x)) && return found
    if ismutable(x)
        x in visited && return found
        push!(visited, x)
    end
    if x isa Reactant.AbstractConcreteArray
        push!(found.arrays, path => x)
    elseif x isa DenseArray
        isbitstype(eltype(x)) && return found
        for i in eachindex(x)
            isassigned(x, i) && walk!(found, visited, x[i], "$path[$i]")
        end
    else
        for name in fieldnames(T)
            isdefined(x, name) && walk!(found, visited, getfield(x, name), "$path.$name")
        end
    end
    return found
end

model_paths(model) = walk!((arrays = Pair{String, Any}[], numbers = Pair{String, Any}[]), Base.IdSet{Any}(), model, "model")

function print_number_changes(label, before, after)
    b = Dict(before)
    changed = [(p, b[p], v) for (p, v) in after if haskey(b, p) && !isequal(b[p], v)]
    stamp(label, ": ", length(after), " host numbers, ", length(changed), " changed")
    for (p, old, new) in changed
        println("    ", p, ": ", old, " → ", new)
    end
    flush(stdout)
end

function compare_memory(label, reference, other)
    rows = Tuple{Float64, Float64, String}[]
    for ((path, a), (_, b)) in zip(reference, other)
        differences = [abs(x - y) for (x, y) in zip(a, b) if !isequal(x, y)]   # NaN where both are NaN is no difference
        isempty(differences) && continue
        scale = maximum(x -> isfinite(x) ? abs(x) : 0.0, a; init = 0.0)
        difference = maximum(differences)
        push!(rows, (difference / max(scale, floatmin()), difference, path))
    end
    sort!(rows; by = first, rev = true)
    stamp(label, ": ", length(rows), " of ", length(reference), " arrays differ")
    for (relative, absolute, path) in first(rows, 25)
        @printf("    %.3e (abs %.3e)  %s\n", relative, absolute, path)
    end
    flush(stdout)
end

# %%
roll_barotropic_substeps!(P["roll_barotropic_substeps"])
Reactant.set_default_backend(P["backend"])
options = adjoint_compile_options()

t0 = time()
setup = omip_adjoint_setup(ReactantState(); Nz, Δt, substeps, n_atmosphere_slices = P["n_atmosphere_slices"])
model = setup.model
obj = OverturningStrength(setup.cpu_grid)
state = snapshot_state(model)
paths = model_paths(model)
@assert all(map(===, map(last, paths.arrays), device_arrays(model)))
stamp(@sprintf("model built in %.0f s: %d device arrays", time() - t0, length(paths.arrays)))
numbers_built = model_paths(model).numbers

windows = Dict(n => build_primal(model, state, obj; n_steps = n, Δt, compile_options = options) for n in 1:n_steps)
numbers_windows = model_paths(model).numbers
print_number_changes("after compiling the windows", numbers_built, numbers_windows)

thunks = with_while_unroll_threshold(0) do
    (refresh = @compile(compile_options = options, refresh_diagnostics!(model)),
     step = @compile(compile_options = options, time_step!(model, Δt)))
end
numbers_thunks = model_paths(model).numbers
print_number_changes("after compiling refresh and step", numbers_windows, numbers_thunks)
stamp("compiled")

snapshot(model) = [path => Array(a) for (path, a) in model_paths(model).arrays]

for n in 1:n_steps
    J_window = windows[n]()
    J_window_host = objective_host(obj, model)
    memory_window = snapshot(model)
    numbers_window = model_paths(model).numbers

    restore_state!(model, state)
    thunks.refresh(model)
    for _ in 1:n
        thunks.step(model, Δt)
    end
    J_steps_host = objective_host(obj, model)
    memory_steps = snapshot(model)

    stamp(@sprintf("%d step(s): window J %.15f (host %.15f), stepped host %.15f, relative difference %.2e",
                   n, J_window, J_window_host, J_steps_host, abs(J_steps_host - J_window_host) / abs(J_window_host)))
    print_number_changes("host numbers, stepped vs window", numbers_window, model_paths(model).numbers)
    compare_memory("device memory, stepped vs window", memory_window, memory_steps)
end
