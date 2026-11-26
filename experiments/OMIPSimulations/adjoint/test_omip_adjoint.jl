# %%
# ==============================================================================
# Differentiable OMIP ocean test: Reactant + Enzyme gradient of the AMOC at 26.5°N with respect to
# the 10-m winds for the ocean-only 2° OMIP model.
# Optional `forcing` (default "jra55"): "ecco4" drives the ocean with ECCO4's monthly wind stress, net heat
# flux and net freshwater flux for `n_months` from 1993-01 (`ecco_adjoint_setup`) instead of the JRA55
# atmosphere and bulk formulae for `n_atmosphere_slices`; the gradient is then with respect to the eastward
# and northward stress.
#
#   julia --project=adjoint adjoint/test_omip_adjoint.jl adjoint/configs/<config>.json
#
# Stages (config `stages`, comma separated):
#   cpu       build the model on CPU(), step it, record the host objective (the forward twin)
#   primal    compile the window objective on ReactantState(), compare with the CPU twin, check repeats
#   gradient  compile the gradient, evaluate twice, shadow census, Taylor remainder check (when the config
#             gives `taylor_h0` and `taylor_nh`)
#   fd_scan   finite differences of the primal window along a saved gradient (`fd_gradient`, a results file
#             of the same configuration) for `fd_nh` step sizes halving from `fd_h0`; needs `primal`, or with
#             `average_days` the window average from the checkpointed steps. Optional `fd_controls` (default
#             ["u", "v"]) names the forcing controls perturbed; optional `fd_directions` ({label: results file})
#             scans along each file's `gradient_<control>` instead of the gradient's own direction, and `fd_h0`
#             may then be {label: h0}
#   step_jacobian  leading growth of one step's adjoint at the states `jacobian_steps` (step indices) by
#             `jacobian_iterations` of power iteration (`step_adjoint_growth`)
#   step_dot  one-step dot-product test of the reverse pass at the states `dot_steps` (`step_dot_products`), with
#             input perturbation amplitudes `dot_scales` ({field: amplitude}) and step factors `dot_epsilons`;
#             optional `dot_gc_every` (default 20) steps between host garbage collections
# Optional `excluded_passes`: pattern names left out of Reactant's rewrite lists (default none).
# Optional `while_unroll_threshold` (default 0, every loop rolled): Reactant's unroll pattern acts on
# loops of at most this many iterations in every rewrite application, before and after Enzyme.
# Optional `frozen_mixing` (default false): the CATKE viscosity and diffusivities enter the gradient as
# constants (`freeze_mixing!`); the forward model is unchanged, so the Taylor check then measures the
# frozen-mixing approximation, not the adjoint's exactness.
# Optional `unroll_steps` (default false): unroll the time-step loop at trace time instead of tracing
# it, so Enzyme's reverse pass is straight-line too (no loop-carried shadow state).
# Optional `checkpoints`: compile one step and its reverse pass instead of the whole window, and run them on
# Revolve's binomial schedule with this many stored model copies (`build_checkpointed_gradient`);
# `unroll_steps` then does not apply.
# Optional `Nx`, `Ny` (default 180, 60): the tripolar grid size.
# Optional `eddy_diffusivity` (default 800 m² s⁻¹): the GM/Redi skew and symmetric diffusivity; null turns
# GM/Redi off.
# Optional `horizontal_diffusivity` (m² s⁻¹; default none): Laplacian horizontal diffusion of every tracer.
# Optional `biharmonic_tracer_timescale` (days; default none): biharmonic tracer diffusion Az² / λ, the form of the
# biharmonic viscosity.
# Optional `advection` (default "weno"): "upwind3" replaces the WENO schemes with linear third-order upwind.
# Optional `convective_adjustment` (m² s⁻¹; default none): implicit convective adjustment of tracers where N² < 0.
# Optional `laplacian_viscosity` (m² s⁻¹; default none): Laplacian horizontal viscosity on top of the biharmonic one.
# Optional `frozen_buoyancy_gradients` (default false): store the buoyancy gradients in fields and hold them
# constant in derivatives (`freeze_buoyancy_gradients!`), so the GM/Redi slopes and tapers enter the gradient as
# constants; the forward model is unchanged.
# Optional `start_date` (ISO date, e.g. "1993-11-01"): the window's first day, for the initial state and the forcing.
# Optional `eddy_diffusivity_fields` (ecco4 only; default false): hold the GM/Redi diffusivities in uniform fields, so
# the gradient also has ∂J/∂κ_skew and ∂J/∂κ_symmetric; with `eddy_taylor_h0` and `eddy_taylor_nh` (m² s⁻¹) the gradient
# stage then runs a Taylor remainder check of each (`eddy_diffusivity_taylor_check`).
# Optional `adjoint_viscosity` and `adjoint_diffusivity` (m² s⁻¹; need `checkpoints`): adjoint-only horizontal Laplacian
# damping of the adjoint velocities and tracers after each reverse step (`AdjointDamping`); the gradient is then
# approximate, so the Taylor check is refused.
# Optional `adjoint_trace_every` (steps; needs `checkpoints`): record per-field summaries of the ocean adjoint every
# this many steps of the reverse sweep (`AdjointTrace`), saved as `adjoint_trace`.
# Optional `repeat_evaluation` (default true): evaluate the gradient a second time and require the same result.
# Optional `average_days` (needs `checkpoints`): the objective is averaged over the window's last
# `average_days` instead of taken at its end; the primal stage evaluates the end value, so it does not apply.
# ==============================================================================

using OMIPAdjoint
using Oceananigans
using Oceananigans.Architectures: ReactantState
using Oceananigans.TimeSteppers: time_step!
using Reactant
using JSON
using JLD2
using Printf
using Test
using Dates: now, format, DateTime

ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"   # the WOA download goes through DataDeps, which otherwise prompts

config_path = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "configs", "twodegree_nz20_1step.json")
P = JSON.parsefile(config_path)
stages = Symbol.(strip.(split(P["stages"], ",")))
Nz, n_steps, Δt, substeps = P["Nz"], P["steps"], Float64(P["dt"]), P["substeps"]
Nx, Ny = get(P, "Nx", 180), get(P, "Ny", 60)   # 2° tripolar by default
forcing = get(P, "forcing", "jra55")
κ_eddy = get(P, "eddy_diffusivity", 800)   # GM/Redi skew and symmetric diffusivity (m² s⁻¹); null turns GM/Redi off
advection = Symbol(get(P, "advection", "weno"))
κ_horizontal = get(P, "horizontal_diffusivity", nothing)   # Laplacian tracer diffusivity (m² s⁻¹)
# Biharmonic tracer diffusion timescale, days → s
λ_tracer = haskey(P, "biharmonic_tracer_timescale") ? P["biharmonic_tracer_timescale"] * 86400.0 : nothing
κ_convective = get(P, "convective_adjustment", nothing)   # convective adjustment diffusivity (m² s⁻¹)
ν_laplacian = get(P, "laplacian_viscosity", nothing)   # Laplacian horizontal viscosity (m² s⁻¹)
frozen_gradients = get(P, "frozen_buoyancy_gradients", false)   # stored buoyancy gradients, constant in derivatives
date_kwargs = haskey(P, "start_date") ? (; start_date = DateTime(P["start_date"])) : (;)
eddy_fields = get(P, "eddy_diffusivity_fields", false)   # GM/Redi diffusivities as fields, with a gradient
if forcing == "ecco4"
    n_months = P["n_months"]
    (n_months - 1) * 28 * 86400 > n_steps * Δt || error("$(n_months) monthly slices do not cover $(n_steps) steps of $(Δt) s")
    model_setup = arch -> ecco_adjoint_setup(arch; Nx, Ny, Nz, Δt, substeps, n_months,
                                             κ_skew = κ_eddy, κ_symmetric = κ_eddy, advection,
                                             horizontal_diffusivity = κ_horizontal,
                                             biharmonic_tracer_timescale = λ_tracer,
                                             convective_adjustment = κ_convective, laplacian_viscosity = ν_laplacian,
                                             materialize_buoyancy_gradients = frozen_gradients,
                                             eddy_diffusivity_fields = eddy_fields, date_kwargs...)
    gradient_units = "Sv per N m⁻²"
elseif forcing == "jra55"
    eddy_fields && error("eddy_diffusivity_fields needs the ecco4 forcing")
    n_slices = P["n_atmosphere_slices"]
    n_slices * 3 * 3600 > n_steps * Δt || error("$(n_slices) three-hourly slices do not cover $(n_steps) steps of $(Δt) s")
    model_setup = arch -> omip_adjoint_setup(arch; Nx, Ny, Nz, Δt, substeps, n_atmosphere_slices = n_slices,
                                             κ_skew = κ_eddy, κ_symmetric = κ_eddy, advection,
                                             horizontal_diffusivity = κ_horizontal,
                                             biharmonic_tracer_timescale = λ_tracer,
                                             convective_adjustment = κ_convective, laplacian_viscosity = ν_laplacian,
                                             materialize_buoyancy_gradients = frozen_gradients,
                                             date_kwargs...)
    gradient_units = "Sv per m/s"
else
    error("forcing must be \"jra55\" or \"ecco4\", got \"$forcing\"")
end

damped = haskey(P, "adjoint_viscosity") || haskey(P, "adjoint_diffusivity")
damped && !haskey(P, "checkpoints") && error("adjoint damping needs the checkpointed gradient (config `checkpoints`)")
damped && haskey(P, "taylor_h0") && error("the damped adjoint is not the exact gradient; drop taylor_h0")
haskey(P, "adjoint_trace_every") && !haskey(P, "checkpoints") &&
    error("adjoint_trace_every needs the checkpointed gradient (config `checkpoints`)")
adjoint_trace = haskey(P, "adjoint_trace_every") ? AdjointTrace(P["adjoint_trace_every"]) : nothing

average_steps = haskey(P, "average_days") ? round(Int, P["average_days"] * 86400 / Δt) : 1
if average_steps > 1
    haskey(P, "checkpoints") || error("average_days needs the checkpointed gradient (config `checkpoints`)")
    :primal in stages && error("the primal stage evaluates the window's end, not the average over average_days")
end

stamp(msg...) = (println("[", format(now(), "HH:MM:SS"), "] ", msg...); flush(stdout))
elapsed(t) = @sprintf("%.1f s", time() - t)

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

# %%
##### CPU forward twin

J_cpu = NaN
if :cpu in stages
    t0 = time()
    cpu = model_setup(CPU())
    obj_cpu = OverturningStrength(cpu.cpu_grid)
    stamp("CPU model built in ", elapsed(t0), "; AMOC section rows ", obj_cpu.rows, ", initial AMOC ",
          @sprintf("%.4f", objective_host(obj_cpu, cpu.model)), " Sv")
    t1 = time()
    for _ in 1:n_steps
        time_step!(cpu.model, Δt)
    end
    J_cpu = objective_host(obj_cpu, cpu.model)
    stamp(@sprintf("CPU twin: %d steps in %s, AMOC %.4f Sv", n_steps, elapsed(t1), J_cpu))
    @test isfinite(J_cpu)
    results["J_cpu"] = J_cpu
    save_results()
end

# %%
##### Reactant model

reactant_stages = (:primal, :fd_scan, :gradient, :step_jacobian, :step_dot)
if any(in(stages), reactant_stages)
    :NVPTX in Reactant.LLVM.backends() ||
        error("Reactant lowers Oceananigans kernels through NVPTX; this Julia's LLVM lacks it (macOS). Run on the GPU cluster.")
    Reactant.set_default_backend(P["backend"])
    roll_barotropic_substeps!(P["roll_barotropic_substeps"])
    freeze_mixing!(get(P, "frozen_mixing", false))
    freeze_buoyancy_gradients!(frozen_gradients)
    options = haskey(P, "excluded_passes") ? adjoint_compile_options(; excluded_passes = String.(P["excluded_passes"])) :
                                             adjoint_compile_options()
    while_unroll_threshold = get(P, "while_unroll_threshold", 0)
    unroll_steps = get(P, "unroll_steps", false)

    t0 = time()
    setup = model_setup(ReactantState())
    model = setup.model
    obj = OverturningStrength(setup.cpu_grid)
    adjoint_damping = damped ? AdjointDamping(setup.cpu_grid; viscosity = get(P, "adjoint_viscosity", 0),
                                              diffusivity = get(P, "adjoint_diffusivity", 0), Δt) : nothing
    state = snapshot_state(model)
    results["geometry"] = surface_geometry(setup.cpu_grid)
    results["slice_times"] = Array(setup.times)   # a device array under ReactantState
    results["window_seconds"] = n_steps * Δt
    results["average_seconds"] = average_steps * Δt
    stamp("Reactant model built in ", elapsed(t0), "; initial AMOC ",
          @sprintf("%.4f", objective_host(obj, model)), " Sv")
end

if :primal in stages
    t0 = time()
    primal! = build_primal(model, state, obj; n_steps, Δt, compile_options = options, while_unroll_threshold, unroll_steps)
    stamp("primal window compiled in ", elapsed(t0))
    t1 = time(); J1 = primal!(); stamp(@sprintf("primal 1: AMOC %.4f Sv in %s", J1, elapsed(t1)))
    t2 = time(); J2 = primal!(); stamp(@sprintf("primal 2: AMOC %.4f Sv in %s, repeat |ΔJ| %.2e", J2, elapsed(t2), abs(J2 - J1)))
    @test isfinite(J1)
    @test abs(J2 - J1) <= 1e-10 * max(abs(J1), 1)
    if isfinite(J_cpu)
        rel = abs(J1 - J_cpu) / max(abs(J_cpu), 1e-12)
        stamp(@sprintf("primal vs CPU twin: %.4f vs %.4f Sv, relative difference %.2e", J1, J_cpu, rel))
        @test rel < 5e-2
    end
    results["J_primal"] = J1; results["J_primal_repeat"] = J2; results["primal_compile_seconds"] = time() - t0
    save_results()
end

if :fd_scan in stages
    if average_steps > 1
        t0 = time()
        thunks = compile_checkpointed_thunks(model, obj; Δt, compile_options = options, while_unroll_threshold)
        stamp("checkpointed steps compiled in ", elapsed(t0))
        objective! = () -> window_average_objective!(thunks, model, state, obj; n_steps, Δt, average_steps)
    else
        :primal in stages || error("the fd_scan stage evaluates the compiled primal window; add \"primal\" to stages")
        objective! = primal!
    end
    controls = Tuple(Symbol.(get(P, "fd_controls", ["u", "v"])))
    read_controls(path) = jldopen(joinpath(@__DIR__, path)) do f
        NamedTuple{controls}(map(c -> f["gradient_$c"], controls))
    end
    g = read_controls(P["fd_gradient"])
    J_saved = jldopen(f -> f["J"], joinpath(@__DIR__, P["fd_gradient"]))
    for (label, path) in get(P, "fd_directions", Dict("" => P["fd_gradient"]))
        t0 = time()
        h0 = P["fd_h0"] isa AbstractDict ? P["fd_h0"][label] : P["fd_h0"]
        stamp("finite-difference scan ", label)
        scan = finite_difference_scan(model, objective!, g; direction = read_controls(path), h0, nh = P["fd_nh"])
        stamp(@sprintf("finite-difference scan in %s; f0 vs the saved gradient's J: relative difference %.2e",
                       elapsed(t0), abs(scan.f0 - J_saved) / abs(J_saved)))
        results[isempty(label) ? "fd_scan" : "fd_scan/$label"] = scan
        save_results()
    end
end

if :gradient in stages
    t0 = time()
    checkpoints = get(P, "checkpoints", nothing)
    evaluate! = checkpoints === nothing ?
        build_gradient(model, state, obj; n_steps, Δt, compile_options = options, while_unroll_threshold, unroll_steps) :
        build_checkpointed_gradient(model, state, obj; n_steps, Δt, checkpoints, average_steps, adjoint_damping,
                                    adjoint_trace, compile_options = options, while_unroll_threshold)
    compile_seconds = time() - t0
    stamp(@sprintf("COMPILED %d-step gradient in %.1f min", n_steps, compile_seconds / 60))

    stamp("evaluating gradient 1...")
    t1 = time(); J, g, dmodel = evaluate!(); first_seconds = time() - t1
    finite = all(a -> all(isfinite, a), g)
    max_g = max(maximum(abs, g.u), maximum(abs, g.v))
    stamp(@sprintf("gradient 1 in %.1f s: AMOC %.4f Sv, max|g| %.3e %s, finite %s", first_seconds, J, max_g, gradient_units, finite))
    if haskey(results, "J_primal")
        rel = abs(J - results["J_primal"]) / max(abs(results["J_primal"]), 1e-12)
        stamp(@sprintf("gradient vs primal objective: relative difference %.2e", rel))
        results["J_gradient_vs_primal"] = rel
        rel < 1e-8 || @warn "gradient's objective differs from the primal window's" rel
    end
    haskey(g, :κ_skew) && stamp(@sprintf("∂J/∂κ_skew %+.6e, ∂J/∂κ_symmetric %+.6e Sv per m² s⁻¹", g.κ_skew, g.κ_symmetric))
    if haskey(g, :κ_skew)
        dclosure = OMIPAdjoint.eddy_closure(dmodel)
        stamp(@sprintf("interior sums of the κ shadows: κ_skew %+.6e, κ_symmetric %+.6e",
                       sum(Array(interior(dclosure.κ_skew))), sum(Array(interior(dclosure.κ_symmetric)))))
    end
    census = shadow_census(dmodel; cpu_grid = setup.cpu_grid)
    print_shadow_census(census); flush(stdout)

    # Saved before the repeat evaluation, which can fail on its own (out of memory, time limit).
    results["J"] = J
    for (name, value) in pairs(g)
        results["gradient_$name"] = value
    end
    results["compile_seconds"] = compile_seconds; results["first_seconds"] = first_seconds
    results["census"] = census
    adjoint_trace === nothing || (results["adjoint_trace"] = copy(adjoint_trace.records))
    save_results()

    repeat_evaluation = get(P, "repeat_evaluation", true)
    J2, second_seconds = J, NaN
    if repeat_evaluation
        stamp("evaluating gradient 2...")
        t2 = time(); J2, g2, _ = evaluate!(); second_seconds = time() - t2
        stamp(@sprintf("gradient 2 in %.2f s: repeat |ΔJ| %.2e, max|Δg| %.2e", second_seconds, abs(J2 - J),
                       max(maximum(abs, g2.u .- g.u), maximum(abs, g2.v .- g.v))))
        results["second_seconds"] = second_seconds
        save_results()
    end
    @test finite
    @test abs(J2 - J) <= 1e-10 * max(abs(J), 1)

    if finite && haskey(P, "taylor_h0")
        stamp("Taylor remainder check...")
        taylor = taylor_remainder_check(model, evaluate!; h0 = P["taylor_h0"], nh = P["taylor_nh"])
        results["taylor"] = taylor
        save_results()
        @test taylor.verified
    end

    if finite && haskey(P, "eddy_taylor_h0")
        eddy_taylor = map((:κ_skew, :κ_symmetric)) do name
            stamp("Taylor remainder check of ", name, "...")
            check = eddy_diffusivity_taylor_check(model, evaluate!, name; h0 = P["eddy_taylor_h0"], nh = P["eddy_taylor_nh"])
            results["eddy_taylor_$name"] = check
            save_results()
            check
        end
        @test all(c -> c.verified, eddy_taylor)
    end
end

if :step_jacobian in stages
    t0 = time()
    jacobian = step_adjoint_growth(model, state; steps = P["jacobian_steps"], iterations = P["jacobian_iterations"], Δt,
                                   compile_options = options, while_unroll_threshold)
    for r in jacobian
        n = length(r.growth)
        stamp(@sprintf("step %d: growth per step %.4g after %d iterations (previous %.4g)", r.step, r.growth[n], n,
                       r.growth[max(n - 1, 1)]))
    end
    stamp("step Jacobian stage in ", elapsed(t0))
    results["step_jacobian"] = jacobian
    save_results()
end

if :step_dot in stages
    t0 = time()
    scales = Dict(Symbol(k) => Float64(v) for (k, v) in P["dot_scales"])
    epsilons = Tuple(Float64.(P["dot_epsilons"]))
    results["step_dot"] = step_dot_products(model, state; steps = P["dot_steps"], Δt, scales, names = Tuple(keys(scales)),
                                            epsilons, gc_every = get(P, "dot_gc_every", 20), compile_options = options,
                                            while_unroll_threshold)
    stamp("step dot-product stage in ", elapsed(t0))
    save_results()
end

stamp("saved ", output_path)
