#####
##### Leading growth of one time step's adjoint at a fixed state
#####

# Deterministic noise in [-1/2, 1/2), so the seed needs no random-number dependency.
interior_noise(i, j, k) = (x = 43758.5453 * sin(12.9898i + 78.233j + 37.719k); x - floor(x) - 0.5)

function seed_interior_noise!(f)
    a = zeros(eltype(parent(f)), size(parent(f)))
    Nx, Ny, Nz = size(f)
    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        a[i + f.grid.Hx, j + f.grid.Hy, k + f.grid.Hz] = interior_noise(i, j, k)
    end
    copyto!(parent(f), a)
    return nothing
end

rescale!(arrays, c) = foreach(a -> eltype(a) <: AbstractFloat && copyto!(a, Array(a) .* c), arrays)

"""
$(TYPEDSIGNATURES)

Power iteration of a linear map on the shadow model `dmodel`. Each of the `iterations` rescales the device
`arrays` of `dmodel` so that the [`AdjointTrace`](@ref) fields have unit root sum of squares, then applies
`apply!(dmodel)`. Returns the norms after each application (the growth factors; the iteration stops early at a
non-finite or zero norm) and the per-field summaries after each, with `step` the iteration number.
"""
function power_iteration!(apply!, dmodel, arrays, iterations)
    iterations >= 1 || throw(ArgumentError("power iteration needs iterations ≥ 1, got $iterations"))
    summarize(n) = [field_summary(n, name, f) for (name, f) in traced_shadow_fields(dmodel)]
    norm_of(summaries) = sqrt(sum(r -> r.rss^2, summaries))
    norm = norm_of(summarize(0))
    norm > 0 || throw(ArgumentError("power iteration needs a nonzero starting shadow"))
    growth, records = Float64[], AdjointRecord[]
    for n in 1:iterations
        rescale!(arrays, 1 / norm)
        apply!(dmodel)
        summaries = summarize(n)
        append!(records, summaries)
        norm = norm_of(summaries)
        push!(growth, norm)
        (isfinite(norm) && norm > 0) || break
    end
    return growth, records
end

"""
$(TYPEDSIGNATURES)

Leading growth of one time step's adjoint at fixed states of the window that starts from the snapshot
`state`. For each `s` in the ascending `steps`, the model is stepped to its `s`-th state xₛ and a shadow model,
seeded with deterministic noise in the interior velocities and tracers, goes through `iterations` of
[`power_iteration!`](@ref) with the reverse pass of the step from xₛ. The growth factors tend to the modulus
of the step Jacobian's leading eigenvalue, which the tangent-linear step shares, and the shadow to that
eigenvalue's adjoint mode when it is real. Returns one named tuple per step: the step, the growth factors,
the per-field summaries and the final shadow velocities and tracers (arrays with halos).
"""
function step_adjoint_growth(model, state; steps, iterations, Δt, compile_options = adjoint_compile_options(),
                             while_unroll_threshold = 0)
    (!isempty(steps) && issorted(steps) && first(steps) >= 0) ||
        throw(ArgumentError("steps must be nonnegative and ascending, got $steps"))
    thunks = with_while_unroll_threshold(while_unroll_threshold) do
        dmodel = make_zero(model)
        (refresh = @compile(compile_options = compile_options, refresh_diagnostics!(model)),
         step = @compile(compile_options = compile_options, time_step!(model, Δt)),
         reverse_step = @compile(compile_options = compile_options, reverse_step!(model, dmodel, Δt)))
    end
    arrays = device_arrays(model)
    restore_state!(model, state)
    thunks.refresh(model)
    position = 0
    return map(steps) do s
        for n in (position + 1):s
            thunks.step(model, Δt)
            # Finalizers free the device buffers of earlier steps; host memory pressure alone does not run them.
            n % 20 == 0 && GC.gc()
        end
        position = s
        xₛ = memory_snapshot(model, arrays)
        dmodel = make_zero(model)
        om = ocean_model(dmodel)
        foreach(seed_interior_noise!, (om.velocities.u, om.velocities.v, om.tracers.T, om.tracers.S))
        growth, records = power_iteration!(dmodel, device_arrays(dmodel), iterations) do dm
            restore_memory!(model, arrays, xₛ)
            thunks.reverse_step(model, dm, Δt)
        end
        restore_memory!(model, arrays, xₛ)   # the reverse pass leaves the model past xₛ
        om = ocean_model(dmodel)
        mode = map(f -> Array(parent(f)), (u = om.velocities.u, v = om.velocities.v, T = om.tracers.T, S = om.tracers.S))
        (; step = s, growth, records, mode)
    end
end
