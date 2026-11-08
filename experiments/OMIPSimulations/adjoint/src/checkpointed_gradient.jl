#####
##### Step-by-step gradient on a binomial checkpoint schedule
#####
# The window gradient differentiates all the window's steps as one module, so its compile time grows with
# the step count. Here each piece is compiled once, whatever the window length: the diagnostic refresh that
# starts the window, one coupled time step, and the reverse passes of the refresh, the step and the
# objective. The host runs them on the Revolve schedule of Griewank and Walther (2000), taken from
# Checkpointing.jl: the reverse sweep visits the steps last to first, recomputing each step's input from
# the nearest checkpoint, and the shadow model carries the adjoint from one step's reverse pass to the next,
# accumulating the wind gradient in its atmosphere shadows.
#
# A checkpoint holds every device array of the model except the grids and the prescribed forcing. The
# prognostic snapshot is not enough mid-window: the coupled step computes the ocean's stage-1 tendencies
# and CATKE's diffusivities before it updates the interface fluxes, so they carry the previous step's fluxes.

# Grids and forcing time series do not change during a step.
skip_in_checkpoint(x) = x isa Union{AbstractGrid, FieldTimeSeries, Number, Function, Module, Type, Symbol, AbstractString}

function collect_device_arrays!(arrays, visited, x)
    T = typeof(x)
    (isbitstype(T) || skip_in_checkpoint(x)) && return arrays
    if ismutable(x)
        x in visited && return arrays
        push!(visited, x)
    end
    if x isa Reactant.AbstractConcreteArray
        push!(arrays, x)
    elseif x isa DenseArray   # host containers; wrappers (offset arrays, views) are walked by their fields
        isbitstype(eltype(x)) && return arrays
        for i in eachindex(x)
            isassigned(x, i) && collect_device_arrays!(arrays, visited, x[i])
        end
    else
        for name in fieldnames(T)
            isdefined(x, name) && collect_device_arrays!(arrays, visited, getfield(x, name))
        end
    end
    return arrays
end

"""
$(TYPEDSIGNATURES)

Every Reactant device array reachable from `model`, each once, except those of grids and field time
series (the prescribed forcing).
"""
device_arrays(model) = collect_device_arrays!(Reactant.AbstractConcreteArray[], Base.IdSet{Any}(), model)

memory_snapshot(model, arrays) = (arrays = map(Array, arrays), clocks = map(clock_snapshot, component_clocks(model)))

function restore_memory!(model, arrays, snapshot)
    foreach(copyto!, arrays, snapshot.arrays)
    foreach(restore_clock!, component_clocks(model), snapshot.clocks)
    return nothing
end

objective_value(model, obj) = obj(model)

reverse_refresh!(model, dmodel) =
    autodiff(set_strong_zero(Reverse), refresh_diagnostics!, Const, Duplicated(model, dmodel))

reverse_step!(model, dmodel, Δt) =
    autodiff(set_strong_zero(Reverse), time_step!, Const, Duplicated(model, dmodel), Const(Δt))

reverse_objective!(model, dmodel, obj) =
    autodiff(set_strong_zero(ReverseWithPrimal), objective_value, Active, Duplicated(model, dmodel), Const(obj))

# Run the steps forward on the Revolve schedule, then back through their reverse passes. The model starts
# at the window's first step input; `dmodel` ends as the adjoint of that input. Returns the objective.
function revolve_sweep!(model, dmodel, arrays, thunks, obj, n_steps, Δt, checkpoints, slots)
    schedule = Revolve{Nothing}(n_steps, min(checkpoints, n_steps))
    J = NaN
    while true
        action = next_action!(schedule)
        flag = action.actionflag
        if flag == Checkpointing.store
            slots[action.cpnum] = memory_snapshot(model, arrays)
        elseif flag == Checkpointing.restore
            restore_memory!(model, arrays, slots[action.cpnum])
        elseif flag == Checkpointing.forward
            for _ in action.startiteration:(action.iteration - 1)
                thunks.step(model, Δt)
            end
        elseif flag == Checkpointing.firstuturn
            # The model holds the last step's input: step to the end, seed the adjoint with the objective's.
            input = memory_snapshot(model, arrays)
            thunks.step(model, Δt)
            J = host_number(thunks.reverse_objective(model, dmodel, obj)[2])
            restore_memory!(model, arrays, input)
            thunks.reverse_step(model, dmodel, Δt)
        elseif flag == Checkpointing.uturn
            thunks.reverse_step(model, dmodel, Δt)
        elseif flag == Checkpointing.done
            return J
        else
            error("unexpected Revolve action $flag")
        end
    end
end

"""
$(TYPEDSIGNATURES)

Compile the gradient of the `n_steps`-step window objective one step at a time and return `evaluate!()`,
which restores the snapshot `state`, evaluates, and returns `(J, gradient, dmodel)` as from
[`build_gradient`](@ref). The forward and reverse steps run on Revolve's binomial schedule with at most
`checkpoints` stored copies of the model's device arrays (host memory), so the compile cost does not
depend on `n_steps` and the recomputation grows only logarithmically with it.
"""
function build_checkpointed_gradient(model, state, obj; n_steps, Δt, checkpoints,
                                     compile_options = adjoint_compile_options(), while_unroll_threshold = 0)
    n_steps >= 1 || throw(ArgumentError("n_steps must be at least 1, got $n_steps"))
    checkpoints >= 1 || throw(ArgumentError("checkpoints must be at least 1, got $checkpoints"))
    thunks = with_while_unroll_threshold(while_unroll_threshold) do
        dmodel = make_zero(model)
        (refresh = @compile(compile_options = compile_options, refresh_diagnostics!(model)),
         step = @compile(compile_options = compile_options, time_step!(model, Δt)),
         reverse_refresh = @compile(compile_options = compile_options, reverse_refresh!(model, dmodel)),
         reverse_step = @compile(compile_options = compile_options, reverse_step!(model, dmodel, Δt)),
         reverse_objective = @compile(compile_options = compile_options, reverse_objective!(model, dmodel, obj)))
    end
    arrays = device_arrays(model)

    function evaluate!()
        dmodel = make_zero(model)
        restore_state!(model, state)
        start = memory_snapshot(model, arrays)     # the refresh's reverse pass runs from here
        slots = Dict{Int, typeof(start)}()
        thunks.refresh(model)
        J = revolve_sweep!(model, dmodel, arrays, thunks, obj, n_steps, Δt, checkpoints, slots)
        restore_memory!(model, arrays, start)
        thunks.reverse_refresh(model, dmodel)
        return J, wind_gradient(dmodel), dmodel
    end
    return evaluate!
end
