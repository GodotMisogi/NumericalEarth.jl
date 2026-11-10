# %%
# ==============================================================================
# Reduce gradient results files (test_omip_adjoint.jl output, hundreds of MB at long windows) to the small
# summaries the plots and the writeup need. Runs on a cluster CPU node next to the results:
#
#   SCRIPT=adjoint/reduce_gradients.jl CONFIG=adjoint/configs/reduce_<name>.json \
#     sbatch -p development -t 00:30:00 adjoint/omip_adjoint.slurm
#
# Config keys (paths relative to adjoint/):
#   gradients  {label: results file}; each gives J, the uniform-wind sums, max|g|, ‖g‖ and the
#              slice-summed sensitivity in 4 × 4 blocks of atmosphere cells (the map plot_amoc_sensitivity.jl draws)
#   pairs      optional [[reference label, other label], ...]: cosine, norm ratio and each gradient's
#              slope along both normalized directions
#   fd_scans   optional {label: results file holding `fd_scan`}, copied through
#   output     the reduced file
# ==============================================================================

using JLD2
using JSON
using LinearAlgebra
using Printf

config_path = length(ARGS) >= 1 ? ARGS[1] : error("usage: reduce_gradients.jl <config.json>")
P = JSON.parsefile(config_path)
path(p) = joinpath(@__DIR__, p)

# Each 2° ocean cell reads its 4 neighboring 0.5625° atmosphere cells, so per-cell values are dotted;
# 4 × 4 block sums give the response to a uniform perturbation over each 2.25° box.
block_sum(g, b) = [sum(view(g, (i-1)*b+1:i*b, (j-1)*b+1:j*b)) for i in 1:size(g, 1)÷b, j in 1:size(g, 2)÷b]
slice_sum(g) = dropdims(sum(g; dims = (3, 4)); dims = (3, 4))

# %%
##### Per-gradient summaries

reduced = Dict{String, Any}("config" => P)
directions = Dict{String, Vector{Float64}}()
for (label, file) in P["gradients"]
    summary = jldopen(path(file)) do f
        gu, gv = Float64.(f["gradient_u"]), Float64.(f["gradient_v"])
        g = vcat(vec(gu), vec(gv))
        directions[label] = g
        (J = f["J"], sum_u = sum(gu), sum_v = sum(gv), max_abs = maximum(abs, g), norm = norm(g),
         block_u = block_sum(slice_sum(gu), 4), block_v = block_sum(slice_sum(gv), 4),
         taylor = haskey(f, "taylor") ? f["taylor"] : nothing)
    end
    reduced["gradient/" * label] = summary
    @printf("%-12s J %+.6f Sv, Σ∂J/∂u %+.4f, Σ∂J/∂v %+.4f, max|g| %.3e, ‖g‖ %.4e\n",
            label, summary.J, summary.sum_u, summary.sum_v, summary.max_abs, summary.norm)
end

# %%
##### Pair comparisons

for (reference, other) in get(P, "pairs", [])
    a, b = directions[reference], directions[other]
    da, db = a ./ norm(a), b ./ norm(b)
    pair = (cosine = dot(a, b) / (norm(a) * norm(b)), norm_ratio = norm(b) / norm(a),
            reference_along_reference = dot(a, da), other_along_reference = dot(b, da),
            reference_along_other = dot(a, db), other_along_other = dot(b, db))
    reduced["pair/$reference/$other"] = pair
    @printf("%s vs %s: cosine %.6f, norm ratio %.4f; along d_%s: %.6e vs %.6e; along d_%s: %.6e vs %.6e\n",
            reference, other, pair.cosine, pair.norm_ratio, reference, pair.reference_along_reference,
            pair.other_along_reference, other, pair.reference_along_other, pair.other_along_other)
end

for (label, file) in get(P, "fd_scans", Dict())
    reduced["fd_scan/" * label] = jldopen(f -> f["fd_scan"], path(file))
end

output_path = path(P["output"])
mkpath(dirname(output_path))
jldopen(output_path, "w") do f
    for (key, value) in reduced
        f[key] = value
    end
end
println("saved ", output_path, " (", round(filesize(output_path) / 2^20, digits = 2), " MiB)")
