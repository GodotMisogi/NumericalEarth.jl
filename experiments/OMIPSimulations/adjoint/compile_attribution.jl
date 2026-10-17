# %%
# ==============================================================================
# Where the OMIP adjoint's compile time goes. For each window variant and mode (primal, gradient)
# the compile is split into its two stages, timed separately on one MLIR module:
#   mlir  Julia tracing, Enzyme and the MLIR pass pipeline (Reactant.Compiler.compile_mlir)
#   xla   XLA backend code generation of that module (Reactant.XLA.compile; ptxas/LLVM)
# with the Julia method-compile time inside each stage and an op histogram of the optimized module.
# Two finer views of the MLIR stage, each switched on in the config:
#   pass_timing  MLIR's per-pass execution time report after every pass pipeline (on stderr)
#   dump_stages  the module after every pass pipeline (Reactant's dump directory, kept under logs/),
#                with an op count per stage, to see which pass grows the module
#   excluded_passes  pattern names left out of Reactant's rewrite lists, replacing adjoint_compile_options'
#                    default (["while_deadresult"]); [] runs every pattern
# Variants, each differentiated with respect to the whole model like the production gradient:
#   coupled  `steps` coupled `time_step!`s, then the AMOC (the production window)
#   ocean    `steps` ocean-only `time_step!`s (no interface fluxes), then the AMOC
#   fluxes   one coupled `update_state!` (exchange-grid interpolation and the bulk flux solver),
#            then the sum of squares of the net ocean surface fluxes
#
#   julia --project=adjoint adjoint/compile_attribution.jl adjoint/configs/compile_attribution_nz20_1step.json
#
# Uses Reactant 0.2.290 compiler internals (compile_mlir, __resolve_device_and_client); the stage
# split follows `Reactant.Compiler.compile_xla`.
# ==============================================================================

using OMIPAdjoint
using OMIPAdjoint: omip_adjoint_grid, omip_adjoint_ocean, omip_adjoint_forcing, ocean_model,
                   window_objective, with_while_unroll_threshold
using NumericalEarth: OceanOnlyModel
using NumericalEarth.EarthSystemModels: default_sea_ice
using NumericalEarth.EarthSystemModels.InterfaceComputations: ComponentInterfaces
using OMIPSimulations: ncar_atmosphere_ocean_fluxes
using Oceananigans
using Oceananigans.Architectures: ReactantState
using Oceananigans.TimeSteppers: time_step!, update_state!
using Reactant
using Reactant: @trace, MLIR, XLA
using Enzyme: autodiff, set_strong_zero, ReverseWithPrimal, Active, Duplicated, Const, make_zero
using JSON
using JLD2
using Printf
using Dates: now, format

ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

config_path = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "configs", "compile_attribution_nz20_1step.json")
P = JSON.parsefile(config_path)
Nz, n_steps, Δt, substeps = P["Nz"], P["steps"], Float64(P["dt"]), P["substeps"]
variants = Symbol.(P["variants"])
modes = Symbol.(P["modes"])

# Markers on stderr too, so ptxas spill warnings in the error log can be attributed to a stage.
stamp(msg...) = (line = string("[", format(now(), "HH:MM:SS"), "] ", msg...);
                 println(line); println(stderr, line); flush(stdout); flush(stderr))

Base.cumulative_compile_timing(true)
julia_compile_seconds() = first(Base.cumulative_compile_time_ns()) / 1e9

# Wall time and Julia method-compile time of `f()`.
function timed_stage(f)
    t0, c0 = time(), julia_compile_seconds()
    value = f()
    return value, time() - t0, julia_compile_seconds() - c0
end

results = Dict{String, Any}("config" => P)
output_path = joinpath(@__DIR__, P["output"])
save_results() = (mkpath(dirname(output_path)); jldsave(output_path; results); nothing)

# %%
##### Model, built stage by stage (the body of `omip_adjoint_setup`, timed)

Reactant.set_default_backend(P["backend"])
roll_barotropic_substeps!(P["roll_barotropic_substeps"])
compile_options = haskey(P, "excluded_passes") ? adjoint_compile_options(; excluded_passes = String.(P["excluded_passes"])) :
                                                 adjoint_compile_options()

build = Dict{String, Any}()
function timed_build(f, name)
    value, seconds, julia = timed_stage(f)
    build[name] = (; seconds, julia)
    stamp(@sprintf("build %-8s %8.1f s wall, %8.1f s Julia compile", name, seconds, julia))
    return value
end

grid, cpu_grid = timed_build(() -> omip_adjoint_grid(ReactantState(); Nz), "grid")
ocean = timed_build(() -> omip_adjoint_ocean(grid, cpu_grid; Δt, substeps), "ocean")
forcing = timed_build(() -> omip_adjoint_forcing(ReactantState(); n_slices = P["n_atmosphere_slices"]), "forcing")
model = timed_build("model") do
    interfaces = ComponentInterfaces(forcing.atmosphere, ocean, default_sea_ice();
                                     radiation = forcing.radiation,
                                     atmosphere_ocean_fluxes = ncar_atmosphere_ocean_fluxes(Float64))
    OceanOnlyModel(ocean; atmosphere = forcing.atmosphere, radiation = forcing.radiation, interfaces)
end
results["build"] = build
obj = OverturningStrength(cpu_grid)
save_results()

# %%
##### Window variants

function ocean_window_objective(model, obj, n_steps, Δt)
    ocean = ocean_model(model)
    @trace mincut = true checkpointing = true track_numbers = false for _ in 1:n_steps
        time_step!(ocean, Δt)
    end
    return obj(model)
end

function flux_objective(model, obj, n_steps, Δt)
    update_state!(model)
    fluxes = model.interfaces.net_fluxes.ocean
    return sum(abs2, interior(fluxes.u)) + sum(abs2, interior(fluxes.v)) + sum(abs2, interior(fluxes.T))
end

windows = (coupled = window_objective, ocean = ocean_window_objective, fluxes = flux_objective)

function variant_gradient!(window, model, dmodel, obj, n_steps, Δt)
    return autodiff(set_strong_zero(ReverseWithPrimal), window, Active,
                    Duplicated(model, dmodel), Const(obj), Const(n_steps), Const(Δt))
end

function compile_target(variant, mode)
    window = windows[variant]
    mode === :primal && return window, (model, obj, n_steps, Δt)
    return variant_gradient!, (window, model, make_zero(model), obj, n_steps, Δt)
end

# %%
##### Stage split

include("mlir_diagnostics.jl")
time_passes[] = get(P, "pass_timing", false)

# XLA code generation of an optimized module, as in `Reactant.Compiler.compile_xla`.
function xla_compile(mod, fn, compile_options)
    client, device = Reactant.Compiler.__resolve_device_and_client(XLA.default_backend(), fn.seen_args,
                                                                    fn.linear_args, fn.is_sharded)
    options = XLA.make_compile_options(; device_id = Int64(XLA.device_ordinal(device)),
                                       xla_compile_options = compile_options.xla_compile_options,
                                       num_replicas = fn.num_replicas, num_partitions = fn.num_partitions,
                                       mesh_ids = nothing,
                                       xla_debug_options = compile_options.xla_debug_options,
                                       xla_executable_build_options = merge(
                                           (; use_shardy_partitioner = fn.use_shardy_partitioner,
                                              use_spmd_partitioning = false),
                                           compile_options.xla_executable_build_options))
    return XLA.compile(client, mod; compile_options = options, num_outputs = length(fn.linear_results),
                       num_parameters = length(fn.linear_args), is_sharded = false,
                       num_replicas = fn.num_replicas, num_partitions = fn.num_partitions)
end

function attribute(variant, mode)
    f, args = compile_target(variant, mode)
    tag = "$(variant)_$(mode)_$(n_steps)step"
    module_path = joinpath(@__DIR__, "logs", "attribution_$tag.mlir")
    stage_dir = joinpath(@__DIR__, "logs", "attribution_stages_$tag")
    empty!(pass_seconds)
    if get(P, "dump_stages", false)
        MLIR.IR.DUMP_MLIR_DIR[] = stage_dir
        MLIR.IR.MLIR_DUMP_COUNTER[] = 0
        MLIR.IR.DUMP_MLIR_ALWAYS[] = true
    end
    MLIR.IR.@dispose ctx = Reactant.ReactantContext() begin
        (mod, fn), mlir_seconds, mlir_julia = with_while_unroll_threshold(0) do
            timed_stage(() -> Reactant.Compiler.compile_mlir(ctx, f, args; compile_options,
                                                             drop_unsupported_attributes = true))
        end
        MLIR.IR.DUMP_MLIR_ALWAYS[] = false
        stamp(@sprintf("%-7s %-8s mlir %8.1f s (Julia compile %7.1f s)", variant, mode, mlir_seconds, mlir_julia))
        passes = print_pipeline_times(stdout, mlir_seconds)
        stages = stage_census(stage_dir)
        print_stages(stdout, stages)

        MLIR.IR.activate(ctx)
        try
            open(io -> show(io, mod), module_path, "w")
            ops, tensors = module_census(module_path)
            print_module_summary(stdout, @sprintf("[%s] %-7s %-8s", format(now(), "HH:MM:SS"), variant, mode), module_path, ops, tensors)

            row = Dict{String, Any}("mlir_seconds" => mlir_seconds, "mlir_julia_seconds" => mlir_julia,
                                    "pass_seconds" => passes, "stages" => stages,
                                    "ops" => ops, "tensors" => tensors, "module_bytes" => filesize(module_path))
            if P["xla"]
                _, xla_seconds, xla_julia = timed_stage(() -> xla_compile(mod, fn, compile_options))
                stamp(@sprintf("%-7s %-8s xla  %8.1f s (Julia compile %7.1f s)", variant, mode, xla_seconds, xla_julia))
                row["xla_seconds"] = xla_seconds
            end
            MLIR.IR.dispose(mod)
            return row
        finally
            MLIR.IR.deactivate(ctx)
        end
    end
end

for variant in variants, mode in modes
    stamp("attributing $variant $mode...")
    results["$(variant)_$(mode)"] = attribute(variant, mode)
    save_results()
end

stamp("saved ", output_path)
