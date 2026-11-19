#####
##### Per-field summaries of the adjoint during the reverse sweep
#####

const AdjointRecord = @NamedTuple{step::Int, name::String, max_abs::Float64, i::Int, j::Int, k::Int,
                                  rss::Float64, n_nonfinite::Int}

"""
$(TYPEDSIGNATURES)

Recorder for the checkpointed gradient's reverse sweep. After the reverse pass of step `m`, when `m - 1` is
divisible by `every` or `m` is the window's last step, each ocean shadow field (the adjoint of the state
`step = m - 1`) is summarized by its largest finite magnitude, the indices of that entry (interior indices:
outside `1:N` is halo; `k = 0` for a single-level field), the root sum of squares of its finite entries and
its count of non-finite entries. `records` is emptied at the start of every evaluation.
"""
struct AdjointTrace
    every :: Int
    records :: Vector{AdjointRecord}

    function AdjointTrace(every::Int)
        every >= 1 || throw(ArgumentError("AdjointTrace needs every ≥ 1, got $every"))
        return new(every, AdjointRecord[])
    end
end

# Every Field reachable through (nested) NamedTuples and Tuples, with dotted or indexed names. With `structs`,
# also the public properties of other structs (closure fields such as CATKE's); names starting with `_` alias
# fields listed under other names.
function named_fields!(out, prefix, x; structs = false)
    if x isa Field
        push!(out, (prefix, x))
    elseif x isa NamedTuple
        for (name, y) in pairs(x)
            named_fields!(out, "$prefix.$name", y; structs)
        end
    elseif x isa Tuple
        for (n, y) in enumerate(x)
            named_fields!(out, "$prefix[$n]", y; structs)
        end
    elseif structs && !isbits(x) && !(x isa AbstractArray)
        for name in propertynames(x)
            startswith(String(name), "_") || named_fields!(out, "$prefix.$name", getproperty(x, name); structs)
        end
    end
    return out
end

# The ocean shadows that carry the adjoint from one step's reverse pass to the next.
function traced_shadow_fields(dmodel)
    om = ocean_model(dmodel)
    fs = om.free_surface
    out = Tuple{String, Field}[]
    named_fields!(out, "velocities", om.velocities)
    named_fields!(out, "tracers", om.tracers)
    named_fields!(out, "free_surface.η", fs.displacement)
    named_fields!(out, "free_surface.barotropic", fs.barotropic_velocities)
    named_fields!(out, "free_surface.filtered", fs.filtered_state)
    named_fields!(out, "pressure", om.pressure)
    named_fields!(out, "closure_fields", om.closure_fields; structs = true)
    named_fields!(out, "timestepper.Gⁿ", om.timestepper.Gⁿ)
    named_fields!(out, "timestepper.Ψ⁻", om.timestepper.Ψ⁻)
    return out
end

function field_summary(step, name, f)
    a = Array(parent(f))
    max_abs, imax = 0.0, CartesianIndex(1, 1, 1)
    sum_squares, n_nonfinite = 0.0, 0
    for I in CartesianIndices(a)
        x = a[I]
        if !isfinite(x)
            n_nonfinite += 1
        else
            sum_squares += x^2
            if abs(x) > max_abs
                max_abs = abs(x)
                imax = I
            end
        end
    end
    i, j, k = Tuple(imax)
    k = size(a, 3) > 1 ? k - f.grid.Hz : 0
    return (; step, name, max_abs, i = i - f.grid.Hx, j = j - f.grid.Hy, k, rss = sqrt(sum_squares), n_nonfinite)
end

record_adjoint!(::Nothing, dmodel, m, n_steps) = nothing

function record_adjoint!(trace::AdjointTrace, dmodel, m, n_steps)
    ((m - 1) % trace.every == 0 || m == n_steps) || return nothing
    for (name, f) in traced_shadow_fields(dmodel)
        push!(trace.records, field_summary(m - 1, name, f))
    end
    return nothing
end
