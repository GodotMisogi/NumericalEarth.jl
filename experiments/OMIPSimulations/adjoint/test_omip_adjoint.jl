# %%
# ==============================================================================
# Differentiable OMIP ocean test: Reactant + Enzyme gradient of the AMOC at 26.5°N with respect to
# the 10-m winds for the ocean-only 2° OMIP model.
#
#   julia --project=adjoint adjoint/test_omip_adjoint.jl adjoint/configs/<config>.json
#
# Stages (config `stages`, comma separated):
#   cpu       build the model on CPU(), step it, record the host objective (the forward twin)
#   primal    compile the window objective on ReactantState(), compare with the CPU twin, check repeats
#   gradient  compile the gradient, evaluate twice, shadow census, Taylor remainder check
# Optional `excluded_passes`: pattern names left out of Reactant's rewrite lists, replacing
# adjoint_compile_options' default (["while_deadresult"]); [] runs every pattern.
# Optional `while_unroll_threshold` (default 0, every loop rolled): Reactant's unroll pattern acts on
# loops of at most this many iterations in every rewrite application, before and after Enzyme.
# Optional `unroll_steps` (default false): unroll the time-step loop at trace time instead of tracing
# it, so Enzyme's reverse pass is straight-line too (no loop-carried shadow state).
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
using Dates: now, format

ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"   # the WOA download goes through DataDeps, which otherwise prompts

config_path = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "configs", "twodegree_nz20_1step.json")
P = JSON.parsefile(config_path)
stages = Symbol.(strip.(split(P["stages"], ",")))
Nz, n_steps, Δt, substeps = P["Nz"], P["steps"], Float64(P["dt"]), P["substeps"]
n_slices = P["n_atmosphere_slices"]
n_slices * 3 * 3600 > n_steps * Δt || error("$(n_slices) three-hourly slices do not cover $(n_steps) steps of $(Δt) s")

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
    cpu = omip_adjoint_setup(CPU(); Nz, Δt, substeps, n_atmosphere_slices = n_slices)
    obj_cpu = OverturningStrength(cpu.cpu_grid)
    stamp("CPU model built in ", elapsed(t0), "; AMOC row ", obj_cpu.row, ", initial AMOC ",
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

reactant_stages = (:primal, :gradient)
if any(in(stages), reactant_stages)
    :NVPTX in Reactant.LLVM.backends() ||
        error("Reactant lowers Oceananigans kernels through NVPTX; this Julia's LLVM lacks it (macOS). Run on the GPU cluster.")
    Reactant.set_default_backend(P["backend"])
    roll_barotropic_substeps!(P["roll_barotropic_substeps"])
    options = haskey(P, "excluded_passes") ? adjoint_compile_options(; excluded_passes = String.(P["excluded_passes"])) :
                                             adjoint_compile_options()
    while_unroll_threshold = get(P, "while_unroll_threshold", 0)
    unroll_steps = get(P, "unroll_steps", false)

    t0 = time()
    setup = omip_adjoint_setup(ReactantState(); Nz, Δt, substeps, n_atmosphere_slices = n_slices)
    model = setup.model
    obj = OverturningStrength(setup.cpu_grid)
    state = snapshot_state(model)
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

if :gradient in stages
    t0 = time()
    evaluate! = build_gradient(model, state, obj; n_steps, Δt, compile_options = options, while_unroll_threshold, unroll_steps)
    compile_seconds = time() - t0
    stamp(@sprintf("COMPILED %d-step gradient in %.1f min", n_steps, compile_seconds / 60))

    stamp("evaluating gradient 1...")
    t1 = time(); J, g, dmodel = evaluate!(); first_seconds = time() - t1
    finite = all(isfinite, g.u) && all(isfinite, g.v)
    max_g = max(maximum(abs, g.u), maximum(abs, g.v))
    stamp(@sprintf("gradient 1 in %.1f s: AMOC %.4f Sv, max|g| %.3e Sv per m/s, finite %s", first_seconds, J, max_g, finite))
    census = shadow_census(dmodel; cpu_grid = setup.cpu_grid)
    print_shadow_census(census); flush(stdout)

    stamp("evaluating gradient 2...")
    t2 = time(); J2, g2, _ = evaluate!(); second_seconds = time() - t2
    stamp(@sprintf("gradient 2 in %.2f s: repeat |ΔJ| %.2e, max|Δg| %.2e", second_seconds, abs(J2 - J),
                   max(maximum(abs, g2.u .- g.u), maximum(abs, g2.v .- g.v))))
    @test finite
    @test abs(J2 - J) <= 1e-10 * max(abs(J), 1)

    results["J"] = J; results["gradient_u"] = g.u; results["gradient_v"] = g.v
    results["compile_seconds"] = compile_seconds; results["first_seconds"] = first_seconds
    results["second_seconds"] = second_seconds; results["census"] = census
    save_results()

    if finite
        stamp("Taylor remainder check...")
        taylor = taylor_remainder_check(model, evaluate!; h0 = P["taylor_h0"], nh = P["taylor_nh"])
        @test taylor.verified
        results["taylor"] = taylor
        save_results()
    end
end

stamp("saved ", output_path)
