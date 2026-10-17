#####
##### Reactant + Enzyme gradient of a short window of the coupled model
#####
# The differentiated function is a `@trace` loop of `time_step!` followed by the objective. The state
# reset (prognostic fields back to their snapshot, clocks back, `update_state!`) runs outside the
# differentiated window, before every evaluation, so the window starts from a consistent diagnostic
# state without carrying `update_state!` in the gradient graph.

"""
$(TYPEDSIGNATURES)

Compile options for the gradient: kernel raising on, `reshape_propagate = :none` and
`transpose_propagate = :up` by default. At least one propagation option must be `:up`.
Reactant's `while_deadresult` rewrite pattern is excluded by default: on the differentiated
module it takes 25 of the 26 minutes of the first post-Enzyme rewrite to remove 250 of 430k ops.
"""
function adjoint_compile_options(; reshape_propagate = :none, transpose_propagate = :up,
                                   excluded_passes = ["while_deadresult"], kw...)
    transpose_propagate == :up || reshape_propagate == :up ||
        throw(ArgumentError("one of transpose_propagate and reshape_propagate must be :up"))
    return Reactant.CompileOptions(; raise = true, raise_first = true, transpose_propagate, reshape_propagate,
                                     excluded_passes, kw...)
end

# Keep every traced loop rolled while compiling (Reactant unrolls loops shorter than its threshold).
function with_while_unroll_threshold(f, threshold)
    old = Reactant.Compiler.WHILE_UNROLL_THRESHOLD[]
    Reactant.Compiler.WHILE_UNROLL_THRESHOLD[] = threshold
    try
        return f()
    finally
        Reactant.Compiler.WHILE_UNROLL_THRESHOLD[] = old
    end
end

host_number(x::Number) = x
host_number(x::Reactant.ConcreteRNumber) = Reactant.to_number(x)

clock_snapshot(clock) = (time = host_number(clock.time), iteration = host_number(clock.iteration),
                         last_Δt = host_number(clock.last_Δt), last_stage_Δt = host_number(clock.last_stage_Δt),
                         stage = clock.stage)

# Plain numbers only: a `ConcreteRNumber` in the snapshot would alias the buffer the compiled step mutates.
function restore_clock!(clock, from)
    clock.time = from.time
    clock.iteration = from.iteration
    clock.last_Δt = from.last_Δt
    clock.last_stage_Δt = from.last_stage_Δt
    clock.stage = from.stage
    return clock
end

component_clocks(model) = (model.clock, ocean_model(model).clock, model.atmosphere.clock, model.radiation.clock)

"""
$(TYPEDSIGNATURES)

Host-side snapshot of the state a gradient evaluation starts from: the ocean's prognostic fields
(velocities, tracers, closure fields, free surface, tendencies) and every component clock.
"""
function snapshot_state(model)
    om = ocean_model(model)
    fields = (velocities = prognostic_state(om.velocities),
              tracers = prognostic_state(om.tracers),
              closure_fields = prognostic_state(om.closure_fields),
              timestepper = prognostic_state(om.timestepper),
              free_surface = prognostic_state(om.free_surface))
    return (; fields, clocks = map(clock_snapshot, component_clocks(model)))
end

# Writes the host snapshot straight into the device buffers; no compile needed.
function restore_state!(model, state)
    om = ocean_model(model)
    restore_prognostic_state!(om.velocities, state.fields.velocities)
    restore_prognostic_state!(om.tracers, state.fields.tracers)
    restore_prognostic_state!(om.closure_fields, state.fields.closure_fields)
    restore_prognostic_state!(om.timestepper, state.fields.timestepper)
    restore_prognostic_state!(om.free_surface, state.fields.free_surface)
    foreach(restore_clock!, component_clocks(model), state.clocks)
    return nothing
end

# Interface fluxes first: the ocean tendencies read them. The ocean's own `update_state!` refreshes the
# diagnostics and the SplitRK3 stage-1 tendencies, which `prognostic_state` does not cover.
function refresh_diagnostics!(model)
    update_state!(model)
    update_state!(ocean_model(model))
    return nothing
end

"""
$(TYPEDSIGNATURES)

Return `reset!()`, which restores `model` to `state` and recomputes its diagnostic state with a
separately compiled [`refresh_diagnostics!`](@ref) (plain Julia on a CPU model).
"""
function build_reset(model, state)
    refresh! = architecture(ocean_model(model).grid) isa ReactantState ?
        (@compile raise = true raise_first = true refresh_diagnostics!(model)) : refresh_diagnostics!
    return () -> (restore_state!(model, state); refresh!(model); nothing)
end

function window_objective(model, obj, n_steps, Δt)
    @trace mincut = true checkpointing = true track_numbers = false for _ in 1:n_steps
        time_step!(model, Δt)
    end
    return obj(model)
end

function gradient_core!(model, dmodel, obj, n_steps, Δt)
    return autodiff(set_strong_zero(ReverseWithPrimal), window_objective, Active,
                    Duplicated(model, dmodel), Const(obj), Const(n_steps), Const(Δt))
end

# The wind controls: the 10-m velocity slices of the in-memory atmosphere.
wind_fields(model) = (u = model.atmosphere.velocities.u, v = model.atmosphere.velocities.v)

# Interior parent index ranges of a surface field time series.
function interior_ranges(fts)
    Nx, Ny, _ = size(fts.grid)
    Hx, Hy = fts.grid.Hx, fts.grid.Hy
    return (Hx + 1 : Hx + Nx, Hy + 1 : Hy + Ny)
end

# Interior entries of a surface field time series as a host array `(Nx, Ny, 1, Nt)`.
function interior_slices(f)
    ix, iy = interior_ranges(f)
    return Array(parent(f))[ix, iy, :, :]
end

"""
$(TYPEDSIGNATURES)

Gradient of the objective with respect to the 10-m winds, read from the atmosphere shadows of
`dmodel`: `(u, v)` host arrays of size `(Nx, Ny, 1, Nt)` over the forcing slices.
"""
wind_gradient(dmodel) = map(interior_slices, wind_fields(dmodel))

"""
$(TYPEDSIGNATURES)

Compile the `n_steps`-step window objective (no derivatives) and return `primal!()`, which resets
the state and returns the objective as a host number.
"""
function build_primal(model, state, obj; n_steps, Δt, compile_options = adjoint_compile_options(),
                      while_unroll_threshold = 0)
    reset! = build_reset(model, state)
    compiled = with_while_unroll_threshold(while_unroll_threshold) do
        @compile compile_options = compile_options window_objective(model, obj, n_steps, Δt)
    end
    return () -> (reset!(); host_number(compiled(model, obj, n_steps, Δt)))
end

"""
$(TYPEDSIGNATURES)

Compile the reverse-mode gradient of the `n_steps`-step window objective and return `evaluate!()`,
which resets the state, evaluates, and returns `(J, gradient, dmodel)`: the objective, the wind
gradient from [`wind_gradient`](@ref) and the full shadow model (for [`shadow_census`](@ref)).
"""
function build_gradient(model, state, obj; n_steps, Δt, compile_options = adjoint_compile_options(),
                        while_unroll_threshold = 0)
    reset! = build_reset(model, state)
    compiled = with_while_unroll_threshold(while_unroll_threshold) do
        dmodel = make_zero(model)
        @compile compile_options = compile_options gradient_core!(model, dmodel, obj, n_steps, Δt)
    end
    function evaluate!()
        reset!()
        dmodel = make_zero(model)      # fresh zero shadows each call; `make_zero!` does not trace
        result = compiled(model, dmodel, obj, n_steps, Δt)
        return host_number(result[2]), wind_gradient(dmodel), dmodel
    end
    return evaluate!
end

#####
##### Shadow census: where a non-finite gradient comes from
#####

function field_census(name, f, wet)
    a = Array(parent(f))
    nan = isnan.(a)
    n_nan = count(nan)
    levels = Int[]
    columns = Tuple{Int, Int}[]
    nan_wet = nan_dry = 0
    if n_nan > 0 && size(a, 3) > 1
        Hx, Hy, Hz = f.grid.Hx, f.grid.Hy, f.grid.Hz
        levels = [k - Hz for k in axes(a, 3) if any(view(nan, :, :, k))]   # interior k; ≤ 0 or > Nz is halo
        columns = [(i - Hx, j - Hy) for i in axes(a, 1), j in axes(a, 2) if any(view(nan, i, j, :))]
    elseif n_nan > 0 && wet !== nothing && size(wet) == size(a)[1:2]
        nan_wet = count(view(nan, :, :, 1) .& wet)
        nan_dry = n_nan - nan_wet
    end
    return (; name, size = size(a), n_nan, n_inf = count(isinf, a),
              n_nonzero = count(x -> isfinite(x) && x != 0, a), levels, columns, nan_wet, nan_dry)
end

# Wet surface cells of the ocean grid on its parent (halo-inclusive) index range.
function surface_wet_mask(cpu_grid)
    Nx, Ny, Nz = size(cpu_grid)
    Hx, Hy = cpu_grid.Hx, cpu_grid.Hy
    wet = falses(Nx + 2Hx, Ny + 2Hy)
    for j in 1:Ny, i in 1:Nx
        wet[i + Hx, j + Hy] = !inactive_node(i, j, Nz, cpu_grid, Center(), Center(), Center())
    end
    return wet
end

"""
$(TYPEDSIGNATURES)

Per-field census of the Enzyme shadows in `dmodel`: ocean velocities, tracers and free-surface
displacement, and the atmosphere and radiation slices. Each entry holds the counts of NaN, Inf and
non-zero finite entries, the vertical levels and `(i, j)` columns holding NaN (3-D fields, interior
indices; out-of-range ones are halo) and, for ocean surface fields with `cpu_grid` given, the
wet/dry split of the NaN cells.
"""
function shadow_census(dmodel; cpu_grid = nothing)
    wet = cpu_grid === nothing ? nothing : surface_wet_mask(cpu_grid)
    om = ocean_model(dmodel)
    out = NamedTuple[]
    for (group, nt) in ((:velocities, om.velocities), (:tracers, om.tracers))
        for (name, f) in pairs(nt)
            push!(out, field_census("$group.$name", f, wet))
        end
    end
    push!(out, field_census("free_surface.η", om.free_surface.displacement, wet))
    atmos = dmodel.atmosphere
    for (name, f) in (("atmosphere.u", atmos.velocities.u), ("atmosphere.v", atmos.velocities.v),
                      ("atmosphere.T", atmos.temperature), ("atmosphere.q", atmos.specific_humidity),
                      ("atmosphere.p", atmos.pressure),
                      ("radiation.sw", dmodel.radiation.downwelling_shortwave),
                      ("radiation.lw", dmodel.radiation.downwelling_longwave))
        push!(out, field_census(name, f, nothing))
    end
    return out
end

function print_shadow_census(io::IO, census)
    for c in census
        (c.n_nan > 0 || c.n_inf > 0) || continue
        where = !isempty(c.levels) ? "levels $(first(c.levels)):$(last(c.levels)) ($(length(c.levels)) of them)" :
                (c.nan_wet + c.nan_dry > 0 ? "wet $(c.nan_wet), dry $(c.nan_dry)" : "")
        println(io, "  ", rpad(c.name, 20), " NaN ", lpad(c.n_nan, 8), "  Inf ", lpad(c.n_inf, 8),
                "  nonzero ", lpad(c.n_nonzero, 9), "  of ", prod(c.size), "  ", where)
        isempty(c.columns) || println(io, "    columns (i, j): ", join(first(c.columns, 20), " "),
                                      length(c.columns) > 20 ? " … ($(length(c.columns)) in all)" : "")
    end
    bad = count(c -> c.n_nan > 0 || c.n_inf > 0, census)
    finite = [c.name for c in census if c.n_nan == 0 && c.n_inf == 0 && c.n_nonzero > 0]
    println(io, "  shadow census: $bad of $(length(census)) fields hold NaN/Inf; finite non-zero fields: ",
            join(finite, ", "))
end
print_shadow_census(census) = print_shadow_census(stdout, census)

#####
##### Finite-difference verification of the compiled gradient
#####

# The control vector: interior wind entries of every forcing slice, u then v.
wind_control(model) = mapreduce(f -> vec(interior_slices(f)), vcat, wind_fields(model))

function set_wind_control!(model, x)
    offset = 0
    for f in wind_fields(model)
        ix, iy = interior_ranges(f)
        a = Array(parent(f))
        n = length(ix) * length(iy) * size(a, 3) * size(a, 4)
        a[ix, iy, :, :] .= reshape(view(x, offset + 1 : offset + n), length(ix), length(iy), size(a, 3), size(a, 4))
        parent(f) .= a     # halos keep the values set at construction
        offset += n
    end
    return nothing
end

"""
$(TYPEDSIGNATURES)

Taylor-remainder check of the compiled gradient `evaluate!` (from [`build_gradient`](@ref)) along
the gradient direction: for `h = h0 / 2ⁱ`, `i = 0:nh-1`, compares `Δf = f(x₀ + h d) − f(x₀)` with
`h gᵀd` and reports the successive-difference slope, which cancels any constant offset in `Δf` and
must match a non-zero `gᵀd` to `tolerance` ("GRADIENT VERIFIED"). Returns the rows and the verdict.
"""
function taylor_remainder_check(model, evaluate!; h0 = 1.0, nh = 6, tolerance = 1e-3)
    x0 = wind_control(model)
    set_wind_control!(model, x0)
    f0, g, _ = evaluate!()
    gvec = vcat(vec(g.u), vec(g.v))
    d = gvec ./ max(sqrt(sum(abs2, gvec)), eps())
    gd = sum(gvec .* d)
    println("taylor_remainder_check (h0 = $h0, nh = $nh):")
    @printf("  f0 = %.8e   gᵀd = %+.8e\n", f0, gd)
    @printf("  %12s  %15s  %14s  %8s\n", "h", "Δf", "|Δf - h gᵀd|", "rate")
    rows = NamedTuple[]
    previous = NaN
    for i in 0:(nh - 1)
        h = h0 / 2.0^i
        set_wind_control!(model, x0 .+ h .* d)
        Δf = first(evaluate!()) - f0
        R1 = abs(Δf - h * gd)
        rate = isnan(previous) ? NaN : previous / R1
        @printf("  %12.4e  %+15.6e  %14.6e  %8.3f\n", h, Δf, R1, rate)
        push!(rows, (; h, Δf, R1, rate))
        previous = R1
    end
    set_wind_control!(model, x0)
    slopes = [(rows[i].Δf - rows[i+1].Δf) / (rows[i].h - rows[i+1].h) for i in 1:(length(rows) - 1)]
    slope = sum(slopes) / length(slopes)
    relative_error = abs(slope - gd) / max(abs(gd), eps())
    verified = gd != 0 && relative_error < tolerance   # a zero gradient gives d = 0, so Δf = 0 proves nothing
    @printf("  FD slope = %+.8e   adjoint gᵀd = %+.8e   rel_err = %.3e\n", slope, gd, relative_error)
    println("  => ", verified ? "GRADIENT VERIFIED" : "MISMATCH")
    return (; rows, slope, gd, relative_error, verified)
end
