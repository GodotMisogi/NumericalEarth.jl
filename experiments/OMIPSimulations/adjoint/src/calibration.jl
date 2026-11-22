#####
##### Calibration against window-mean tracer observations
#####
# The misfit J = ½ Σ_observed [(T̄ − T°)² / σ_T² + (S̄ − S°)² / σ_S²] compares the model's mean temperature and
# salinity over the window's last `average_steps` states with observed means. J is quadratic in the means, but its
# gradient is the gradient of the linear functional Σ w_T T + Σ w_S S averaged over the same states, with the
# weights w = ∂J/∂T̄ (∂J/∂S̄) fixed by a forward pass. The checkpointed gradient handles averaged linear objectives,
# so one forward pass for the means and one checkpointed gradient give ∇J exactly.

"""
    TracerWeights

Linear objective Σ `T` ⋅ T + Σ `S` ⋅ S over the ocean interior, with device arrays of weights that can be
overwritten between calls of a compiled objective.
"""
struct TracerWeights{W}
    T :: W
    S :: W
end

(obj::TracerWeights)(model) = sum(obj.T .* interior(ocean_model(model).tracers.T)) +
                              sum(obj.S .* interior(ocean_model(model).tracers.S))

"""
$(TYPEDSIGNATURES)

A [`TracerWeights`](@ref) of zeros on the device of `model`.
"""
function TracerWeights(model)
    sz = size(ocean_model(model).grid)
    return TracerWeights(Reactant.to_rarray(zeros(sz)), Reactant.to_rarray(zeros(sz)))
end

"""
    TracerMisfit

Observed window means `T` and `S` (host arrays of the interior size) on the cells where `observed` is true, with
error scales `σT` (K) and `σS` (g kg⁻¹).
"""
struct TracerMisfit{M, A}
    observed :: M
    T :: A
    S :: A
    σT :: Float64
    σS :: Float64
end

"""
$(TYPEDSIGNATURES)

The wet cells of `cpu_grid` in the top `levels` levels, every `stride`-th column in each horizontal direction.
"""
function observed_cells(cpu_grid; levels = size(cpu_grid, 3), stride = 1)
    Nx, Ny, Nz = size(cpu_grid)
    return [k > Nz - levels && mod(i - 1, stride) == 0 && mod(j - 1, stride) == 0 &&
            !inactive_node(i, j, k, cpu_grid, Center(), Center(), Center()) for i in 1:Nx, j in 1:Ny, k in 1:Nz]
end

"""
$(TYPEDSIGNATURES)

The misfit of the means `T̄`, `S̄` against `m`, and its gradient with respect to them, `(; J, wT, wS)`.
"""
function misfit(m::TracerMisfit, T̄, S̄)
    rT = ifelse.(m.observed, (T̄ .- m.T) ./ m.σT, 0.0)
    rS = ifelse.(m.observed, (S̄ .- m.S) ./ m.σS, 0.0)
    return (J = (sum(abs2, rT) + sum(abs2, rS)) / 2, wT = rT ./ m.σT, wS = rS ./ m.σS)
end

function accumulate_tracers!(sums, model)
    sums.T .+= interior(ocean_model(model).tracers.T)
    sums.S .+= interior(ocean_model(model).tracers.S)
    return nothing
end

"""
$(TYPEDSIGNATURES)

Forward window from the snapshot `state` with the compiled `thunks` ([`compile_checkpointed_thunks`](@ref)):
returns the host means of the ocean temperature and salinity over the last `average_steps` of `n_steps` states,
the states the checkpointed gradient averages. `accumulate` is the compiled `accumulate_tracers!(sums, model)`.
"""
function window_tracer_means!(thunks, accumulate, sums, model, state; n_steps, Δt, average_steps)
    restore_state!(model, state)
    thunks.refresh(model)
    foreach(a -> copyto!(a, zeros(size(a))), (sums.T, sums.S))
    for n in 1:n_steps
        thunks.step(model, Δt)
        n > n_steps - average_steps && accumulate(sums, model)
    end
    return Array(sums.T) ./ average_steps, Array(sums.S) ./ average_steps
end

"""
$(TYPEDSIGNATURES)

Misfit evaluation for calibrating the GM/Redi diffusivities of `model` (built with `eddy_diffusivity_fields =
true`) against `observations`, a [`TracerMisfit`](@ref). The controls are `x = log.([κ_skew, κ_symmetric])`.
Compiles the checkpointed gradient of a [`TracerWeights`](@ref) objective once and returns `(; misfit!,
gradient!, means!)`: `misfit!(x)` runs the forward window and returns J; `gradient!(x)` returns `(J, ∂J/∂x)`,
reusing the forward window of the last `misfit!` at the same `x`; `means!(x)` returns the window means.
`n_steps`, `checkpoints` and `average_steps` are those of [`checkpointed_gradient`](@ref).
"""
function build_eddy_calibration(model, state, observations; Δt, n_steps, average_steps, checkpoints,
                                compile_options = adjoint_compile_options(), while_unroll_threshold = 0)
    weights = TracerWeights(model)
    thunks = compile_checkpointed_thunks(model, weights; Δt, compile_options, while_unroll_threshold)
    sz = size(ocean_model(model).grid)
    sums = (T = Reactant.to_rarray(zeros(sz)), S = Reactant.to_rarray(zeros(sz)))
    accumulate = @compile compile_options = compile_options accumulate_tracers!(sums, model)
    evaluate! = checkpointed_gradient(thunks, model, state, weights; n_steps, Δt, checkpoints, average_steps)
    names = (:κ_skew, :κ_symmetric)
    cached = Ref{Any}(nothing)

    function means!(x)
        foreach((name, xᵢ) -> set_eddy_diffusivity!(model, name, exp(xᵢ)), names, x)
        return window_tracer_means!(thunks, accumulate, sums, model, state; n_steps, Δt, average_steps)
    end

    function misfit!(x)
        result = misfit(observations, means!(x)...)
        cached[] = (x = copy(x), result)
        return result.J
    end

    function gradient!(x)
        (cached[] !== nothing && cached[].x == x) || misfit!(x)
        result = cached[].result
        copyto!(weights.T, result.wT)
        copyto!(weights.S, result.wS)
        _, g, _ = evaluate!()
        # ∂J/∂log κ = κ ∂J/∂κ
        return result.J, [exp(x[1]) * g.κ_skew, exp(x[2]) * g.κ_symmetric]
    end

    return (; misfit!, gradient!, means!)
end

"""
$(TYPEDSIGNATURES)

Minimize `f` from `x0` by BFGS with an Armijo backtracking line search: `gradient!(x)` returns `(f(x), ∇f(x))`
and `misfit!(x)` returns `f(x)`. The first step has length `initial_step`; a step is halved at most `backtracks`
times. Stops after `iterations` accepted steps, when the gradient norm falls below `gtol`, or when no step
decreases `f`. Calls `callback(record)` after each iteration and returns the iteration records.
"""
function bfgs_minimize(gradient!, misfit!, x0; iterations = 20, initial_step = 0.2, gtol = 0.0, c₁ = 1e-4,
                       backtracks = 6, callback = identity)
    x = copy(x0)
    f, g = gradient!(x)
    H = (initial_step / max(sqrt(sum(abs2, g)), eps())) * [i == j for i in eachindex(x), j in eachindex(x)]
    records = [(; iteration = 0, x = copy(x), f, g = copy(g), step = 0.0, evaluations = 1)]
    callback(records[end])
    for iteration in 1:iterations
        sqrt(sum(abs2, g)) <= gtol && break
        d = -H * g
        slope = sum(g .* d)   # negative: H stays positive definite
        α, evaluations, accepted = 1.0, 0, false
        for _ in 0:backtracks
            evaluations += 1
            if misfit!(x .+ α .* d) <= f + c₁ * α * slope
                accepted = true
                break
            end
            α /= 2
        end
        accepted || break
        x_new = x .+ α .* d
        f_new, g_new = gradient!(x_new)
        s, y = x_new .- x, g_new .- g
        sy = sum(s .* y)
        if sy > 0   # the curvature condition keeps H positive definite
            ρ = 1 / sy
            V = [i == j for i in eachindex(x), j in eachindex(x)] .- ρ .* (s * y')
            H = V * H * V' .+ ρ .* (s * s')
        end
        x, f, g = x_new, f_new, g_new
        push!(records, (; iteration, x = copy(x), f, g = copy(g), step = α, evaluations))
        callback(records[end])
    end
    return records
end
