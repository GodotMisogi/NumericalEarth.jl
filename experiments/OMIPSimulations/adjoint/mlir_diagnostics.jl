#####
##### MLIR-stage diagnostics shared by compile_attribution.jl and pass_bisect.jl
#####
# Uses Reactant 0.2.290 internals: `MLIR.IR.run!` (replaced by a timed copy below) and the dump
# facility (`DUMP_MLIR_DIR`, `DUMP_MLIR_ALWAYS`, `MLIR_DUMP_COUNTER`) in `mlir/IR/Pass.jl`.

using Reactant: Reactant, MLIR
using Printf: @printf

# Wall time per pass-pipeline key (Reactant's `run_pass_pipeline!` keys), recorded by a timed copy of
# Reactant 0.2.290's `MLIR.IR.run!`. With `time_passes[]` MLIR also prints its per-pass timing report
# to stderr when the pass manager is disposed, right after each pipeline.
const pass_seconds = Dict{String, Float64}()
const time_passes = Ref{Bool}(false)
record_pass(key, seconds) = (pass_seconds[key] = get(pass_seconds, key, 0.0) + seconds; nothing)

@eval Reactant.MLIR.IR function run!(pm::PassManager, operation, key::String="")
    $time_passes[] && API.mlirPassManagerEnableTiming(pm)
    t0 = time()
    DUMP_MLIR_ALWAYS[] && dump_mlir(operation, pm, isempty(key) ? "pre_pm" : "pre_$(key)_pm")
    status = LogicalResult(API.mlirPassManagerRunOnOp(pm, operation))
    failed = isfailure(status)
    if failed || DUMP_MLIR_ALWAYS[]
        dump_mlir(operation, pm, isempty(key) ? "post_pm" : "post_$(key)_pm"; failed)
    end
    $record_pass(isempty(key) ? "(unnamed)" : key, time() - t0)
    failed && throw("failed to run pass manager on module")
    return operation
end

function print_pipeline_times(io, total_seconds)
    passes = sort(collect(pass_seconds); by = last, rev = true)
    inside = sum(last, passes; init = 0.0)
    @printf(io, "  pass pipelines %.1f s, outside them (tracing) %.1f s; slowest:\n", inside, total_seconds - inside)
    foreach(((key, s),) -> @printf(io, "    %9.1f s  %s\n", s, key), first(passes, 10))
    return Dict(passes)
end

const element_bytes = Dict("f64" => 8, "f32" => 4, "i64" => 8, "i32" => 4, "i1" => 1)

# Op name → count and ranked tensor type → occurrences over a printed module, streamed so a large
# module never sits in memory as one string. The largest tensor types show what the reverse pass caches.
function module_census(path)
    ops = Dict{String, Int}()
    tensors = Dict{String, Int}()
    for line in eachline(path)
        m = match(r"^\s*(?:%[^=]*=\s*)?\"?([a-z_]+\.[a-z_][A-Za-z_0-9.]*)", line)
        m === nothing || (ops[m[1]] = get(ops, m[1], 0) + 1)
        for t in eachmatch(r"tensor<([0-9x]+)x([a-z]+[0-9]+)>", line)
            tensors[t.match] = get(tensors, t.match, 0) + 1
        end
    end
    return ops, tensors
end

function tensor_bytes(type)
    t = match(r"tensor<([0-9x]+)x([a-z]+[0-9]+)>", type)
    return prod(parse.(Int, split(t[1], "x"))) * get(element_bytes, t[2], 8)
end

n_ops(ops) = sum(values(ops); init = 0)

function print_module_summary(io, label, path, ops, tensors)
    top = first(sort(collect(ops); by = last, rev = true), 12)
    println(io, label, @sprintf(" module %d ops, %.1f MB; ", n_ops(ops), filesize(path) / 2^20),
            "while ", get(ops, "stablehlo.while", 0), ", kernel_call ", get(ops, "enzymexla.kernel_call", 0),
            ", llvm.func ", get(ops, "llvm.func", 0), "; top: ", join(("$k $v" for (k, v) in top), ", "))
    largest = first(sort(collect(keys(tensors)); by = tensor_bytes, rev = true), 15)
    println(io, "  largest tensor types (MiB, occurrences):")
    foreach(t -> @printf(io, "    %10.1f  %6d  %s\n", tensor_bytes(t) / 2^20, tensors[t], t), largest)
    return nothing
end

# Reactant dumps the module before and after every pass pipeline into `dir` (file names carry a
# counter and the pipeline key). Returns the op count per stage, in pipeline order: the post-pipeline
# modules plus the first pre-pipeline one, the module as traced.
function stage_census(dir)
    isdir(dir) || return NamedTuple[]
    files = sort(filter(f -> endswith(f, ".mlir") && (!occursin("_pre_", f) || startswith(f, "module_000_")), readdir(dir)))
    stages = NamedTuple[]
    for file in files
        ops, _ = module_census(joinpath(dir, file))
        key = replace(file, r"^module_\d+_.*?(?=pre_|post_)" => "", r"_pm\.mlir$" => "", r"\.mlir$" => "")
        push!(stages, (; key, n_ops = n_ops(ops), n_while = get(ops, "stablehlo.while", 0),
                        bytes = filesize(joinpath(dir, file))))
    end
    return stages
end

function print_stages(io, stages)
    isempty(stages) && return nothing
    println(io, "  module after each pass pipeline (ops, while loops, MB):")
    foreach(s -> @printf(io, "    %9d  %4d  %8.1f  %s\n", s.n_ops, s.n_while, s.bytes / 2^20, s.key), stages)
    return nothing
end
