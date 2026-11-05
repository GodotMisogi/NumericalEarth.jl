# %%
# ==============================================================================
# Time Reactant's MLIR optimization of a dumped module one pass-pipeline segment at a time.
#
# Takes a module written by the stage dumps of compile_attribution.jl (the traced module, or the one
# after any segment of an earlier bisect) and runs the segments listed in the config on it in order,
# with the wall time, MLIR's per-pass timing report (stderr) and the op count after each. The module
# after each segment is written to logs/bisect_<tag>/<nn>_<name>.mlir, so a later run can start from
# it. No model build and no GPU work: the whole run is MLIR passes on the CPU of the node.
#
# A segment is {"name", "pipeline"} with a pass pipeline string in which these placeholders expand to
# the strings Reactant (0.2.290 or 0.2.303) builds in `compile_mlir!` for the adjoint's compile options:
#   {opt_passes} {opt_passes2} {kern} {raise_passes} {enzyme_pass} {lower_enzymexla_passes} {jit}
#   {main_patterns} {lower_patterns}   the two pattern-rewrite applications of opt_passes2
#   {opt_passes_down} {opt_passes_up}  the passes of `post_op_transpose_reshape` (down, up, down)
# or {"name", "patterns": [i, j]}: entries i:j of the main pattern list as one rewrite application,
# or {"name", "patterns_except": [i, j]}: the main pattern list without entries i:j,
# or {"name", "patterns_split": n}: the main pattern list in n consecutive chunks, one segment each.
# A segment with "module" starts from that dumped module (path relative to adjoint/) instead of the module
# left by the previous segment, so one run can try independent variants from the same starting point.
# Optional `while_unroll_threshold` (default 0) and `excluded_passes` (default none) set the pattern lists
# as in the compile being reproduced.
#
#   julia --project=adjoint adjoint/pass_bisect.jl adjoint/configs/pass_bisect_ocean_1step_all.json
# ==============================================================================

using Reactant
using Reactant: MLIR, XLA
using Reactant.Compiler: optimization_passes, enzyme_pass, run_pass_pipeline!
using JSON
using JLD2
using Printf
using Dates: now, format

config_path = ARGS[1]
P = JSON.parsefile(config_path)
module_path = joinpath(@__DIR__, P["module"])
isfile(module_path) || error("module not found: $module_path")
backend = P["backend"]
dump_dir = joinpath(@__DIR__, "logs", "bisect_$(P["tag"])")
mkpath(dump_dir)
output_path = joinpath(@__DIR__, P["output"])

include("mlir_diagnostics.jl")
time_passes[] = get(P, "pass_timing", true)

stamp(msg...) = (line = string("[", format(now(), "HH:MM:SS"), "] ", msg...);
                 println(line); println(stderr, line); flush(stdout); flush(stderr))

Reactant.set_default_backend(backend)
Reactant.Compiler.WHILE_UNROLL_THRESHOLD[] = get(P, "while_unroll_threshold", 0)
opts = Reactant.CompileOptions(; raise = true, raise_first = true, transpose_propagate = :up, reshape_propagate = :none,
                                 excluded_passes = String.(get(P, "excluded_passes", String[])))

# %%
##### Reactant's pipeline building blocks, as `compile_mlir!` assembles them for a raising GPU compile

# The pipeline strings are copied from `compile_mlir!` of these two versions; 0.2.303 dropped the `backend`
# keyword of `optimization_passes` and lowers kernels through `convert-polygeist-to-llvm`.
reactant_version = pkgversion(Reactant)
reactant_version in (v"0.2.290", v"0.2.303") ||
    error("pass_bisect.jl reproduces the pipelines of Reactant 0.2.290 and 0.2.303, not $reactant_version")
legacy_pipeline = reactant_version == v"0.2.290"

function pipeline_blocks(opts, backend)
    client = XLA.default_backend()
    platform = backend == "gpu" ? "GPU" : "cpu"
    platform_kw = legacy_pipeline ? (; backend = platform) : (;)
    opt_passes  = optimization_passes(opts; sroa = true,  platform_kw...)
    opt_passes2 = optimization_passes(opts; sroa = false, platform_kw...)
    post_op_kw = (; raise_shlo_to_blas_lapack = false, platform_kw...)
    opt_passes_down = optimization_passes(Reactant.__compile_options_with_reversed_propagation(opts); post_op_kw...)
    opt_passes_up   = optimization_passes(opts; post_op_kw...)
    speculate = opts.speculate_partial_ifs
    raise_passes = "canonicalize,llvm-to-memref-access,canonicalize,convert-llvm-to-cf,canonicalize,enzyme-lift-cf-to-scf,canonicalize,func.func(canonicalize-loops{speculate_partial_if=$speculate}),canonicalize-scf-for,canonicalize,libdevice-funcs-raise,canonicalize,affine-cfg,canonicalize,func.func(canonicalize-loops{speculate_partial_if=$speculate}),canonicalize,llvm-to-affine-access,canonicalize,delinearize-indexing,canonicalize,simplify-affine-exprs,affine-cfg,canonicalize,func.func(affine-loop-invariant-code-motion),canonicalize,sort-memory,func.func(kernelcast),raise-affine-to-stablehlo{strip_llvm_debuginfo=$(opts.strip_llvm_debuginfo) prefer_while_raising=false dump_failed_lockstep=$(Reactant.Compiler.DUMP_FAILED_LOCKSTEP[])},canonicalize,arith-raise{stablehlo=true}," * opt_passes2
    if Reactant.Compiler.DUS_TO_CONCAT[]
        raise_passes *= "," * optimization_passes(opts; sroa = false, dus_to_concat = true, backend = platform)
    end
    blas_int_width = sizeof(Reactant.Compiler.BlasInt) * 8
    C = Reactant.Compiler
    linalg_platform = legacy_pipeline ? platform : XLA.platform_name(client)
    lower_enzymexla_passes = join(["lower-enzymexla-linalg{backend=$linalg_platform blas_int_width=$blas_int_width}",
                                   "lower-enzymexla-blas{backend=$linalg_platform blas_int_width=$blas_int_width}",
                                   "lower-enzymexla-lapack{backend=$linalg_platform blas_int_width=$blas_int_width}",
                                   "lower-enzymexla-math",
                                   legacy_pipeline ? "lower-enzymexla-mpi{backend=$platform}" :
                                                     "lower-comm-to-jit{mpi_request_width=$(C.MPI_REQUEST_WIDTH[])}"], ",")
    if backend == "gpu"
        kern = "lower-kernel{backend=cpu},symbol-dce,canonicalize"
        properties = XLA.device_properties(XLA.default_device(client))
        chip = "sm_$(properties.major)$(properties.minor)"
        toolkit = XLA.CUDA_DATA_DIR[]
        jit = if legacy_pipeline
            "lower-jit{cuOptLevel=$(C.cuOptLevel[]) cubinFormat=$(C.cubinFormat[]) indexBitWidth=$(C.cuindexBitWidth[])  cubinChip=$chip cubinFeatures=$(C.cubinFeatures()) run_init=true toolkitPath=$toolkit},symbol-dce"
        else
            cuda_get_kernel = div(MLIR.API.ReactantHermeticCudaGetVersion(), 1000) == 12
            "convert-polygeist-to-llvm{backend=cuda preserve-xla-funcs=true cuda-get-kernel=$cuda_get_kernel},strip-gpu-info,nvvm-attach-target{chip=$chip features=$(C.cubinFeatures()) O=$(C.cuOptLevel[])},enzymexla-gpu-module-to-binary{format=$(C.cubinFormat[]) toolkit=$toolkit},lower-jit,symbol-dce"
        end
    else
        kern = "lower-kernel{backend=cpu},canonicalize"
        jit = legacy_pipeline ? "lower-jit{openmp=$(C.OpenMP[]) backend=cpu},symbol-dce" :
              "lower-affine,$(C.OpenMP[] ? "convert-scf-to-openmp" : "convert-scf-to-cf"),convert-polygeist-to-llvm{backend=cpu preserve-xla-funcs=true},lower-jit,symbol-dce"
    end
    # The two pattern-rewrite applications inside opt_passes2: the main set, then the lowering set.
    rewrites = [m.match for m in eachmatch(r"enzyme-hlo-generate-td\{patterns=[^}]*\},transform-interpreter,enzyme-hlo-remove-transform", opt_passes2)]
    length(rewrites) >= 2 || error("expected two pattern-rewrite applications in opt_passes2, found $(length(rewrites))")
    main_patterns, lower_patterns = rewrites[1], rewrites[2]
    pattern_list = split(match(r"patterns=([^}]*)\}", main_patterns)[1], ";"; keepempty = false)
    return (; opt_passes, opt_passes2, kern, raise_passes, enzyme_pass, lower_enzymexla_passes, jit,
              main_patterns, lower_patterns, pattern_list, opt_passes_down, opt_passes_up)
end

blocks = pipeline_blocks(opts, backend)
stamp("main pattern list: ", length(blocks.pattern_list), " patterns")

rewrite_of(patterns) = "enzyme-hlo-generate-td{patterns=" * join(patterns, ";") * "},transform-interpreter,enzyme-hlo-remove-transform"

function expand_segments(specs, blocks)
    segments = NamedTuple{(:name, :pipeline, :start), Tuple{String, String, Union{Nothing, String}}}[]
    for spec in specs
        start = haskey(spec, "module") ? joinpath(@__DIR__, spec["module"]) : nothing
        start === nothing || isfile(start) || error("module not found: $start")
        if haskey(spec, "pipeline")
            pipeline = spec["pipeline"]
            for key in (:opt_passes, :opt_passes2, :kern, :raise_passes, :enzyme_pass, :lower_enzymexla_passes,
                        :jit, :main_patterns, :lower_patterns, :opt_passes_down, :opt_passes_up)
                pipeline = replace(pipeline, "{$key}" => getproperty(blocks, key))
            end
            push!(segments, (; name = spec["name"], pipeline, start))
        elseif haskey(spec, "patterns")
            i, j = spec["patterns"]
            push!(segments, (; name = spec["name"], pipeline = rewrite_of(blocks.pattern_list[i:j]), start))
        elseif haskey(spec, "patterns_except")
            i, j = spec["patterns_except"]
            kept = blocks.pattern_list[setdiff(eachindex(blocks.pattern_list), i:j)]
            push!(segments, (; name = spec["name"], pipeline = rewrite_of(kept), start))
        elseif haskey(spec, "patterns_split")
            n = spec["patterns_split"]
            bounds = round.(Int, range(0, length(blocks.pattern_list); length = n + 1))
            for c in 1:n
                i, j = bounds[c] + 1, bounds[c + 1]
                push!(segments, (; name = "$(spec["name"])_$c", pipeline = rewrite_of(blocks.pattern_list[i:j]),
                                  start = c == 1 ? start : nothing))
            end
        else
            error("segment needs pipeline, patterns, patterns_except or patterns_split: $spec")
        end
    end
    return segments
end

segments = expand_segments(P["segments"], blocks)
for (k, s) in enumerate(segments)
    @printf("  %2d %-28s %s\n", k, s.name, length(s.pipeline) > 160 ? first(s.pipeline, 157) * "..." : s.pipeline)
end
results = Dict{String, Any}("config" => P, "segments" => NamedTuple[])
save_results() = (mkpath(dirname(output_path)); jldsave(output_path; results); nothing)

# %%
##### Run

MLIR.IR.@with_context Reactant.ReactantContext() begin
    t0 = time()
    mod = parse(MLIR.IR.Module, read(module_path, String))
    stamp(@sprintf("parsed %s (%.1f MB) in %.1f s", basename(module_path), filesize(module_path) / 2^20, time() - t0))
    ops, _ = module_census(module_path)
    @printf("  start: %d ops, %d while loops\n", n_ops(ops), get(ops, "stablehlo.while", 0))

    for (k, segment) in enumerate(segments)
        if segment.start !== nothing
            mod = parse(MLIR.IR.Module, read(segment.start, String))
            stamp("segment $k starts from ", basename(segment.start))
        end
        stamp("segment $k $(segment.name)...")
        empty!(pass_seconds)
        t0 = time()
        run_pass_pipeline!(mod, segment.pipeline, segment.name)
        seconds = time() - t0
        path = joinpath(dump_dir, @sprintf("%02d_%s.mlir", k, segment.name))
        open(io -> show(io, mod), path, "w")
        ops, tensors = module_census(path)
        stamp(@sprintf("segment %d %-28s %8.1f s: %d ops, %d while, %.1f MB",
                       k, segment.name, seconds, n_ops(ops), get(ops, "stablehlo.while", 0), filesize(path) / 2^20))
        push!(results["segments"], (; segment.name, segment.pipeline, seconds, n_ops = n_ops(ops), ops, tensors, path))
        save_results()
    end
    print_module_summary(stdout, "final", results["segments"][end].path, results["segments"][end].ops,
                         results["segments"][end].tensors)
end

stamp("saved ", output_path)
