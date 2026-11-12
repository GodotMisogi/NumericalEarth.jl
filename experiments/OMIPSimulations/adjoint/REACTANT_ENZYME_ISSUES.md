# Reactant and Enzyme issues found in the OMIP adjoint

Problems found in Reactant.jl, Reactant_jll and Enzyme-JAX while differentiating the 2° OMIP ocean
(`OMIPAdjoint`, this directory). Only problems in those tools are listed. Problems that come from how
Oceananigans or NumericalEarth meet Reactant are listed at the end and kept out of the main list.

Versions, unless stated otherwise: Julia 1.12.7, Reactant.jl 0.2.303, Reactant_jll 0.0.436, Enzyme_jll
0.0.303+1, Oceananigans `ss/for-omip` (0.113.5). Hardware: TACC Lonestar6 A100 (x86_64) and Vista GH200
(aarch64). Job numbers are Slurm jobs; the detailed record is `PROGRESS.md` and the repository-root
`LOGBOOK.md`.

| # | Component | Issue | Effect | Reported | Workaround here |
|---|---|---|---|---|---|
| 1 | Reactant_jll raise + Enzyme-JAX | Empty kernel raised to an operand-free `enzymexla.jit_call` that Enzyme cannot differentiate | Gradient fails to compile | Yes: EnzymeAD/Reactant.jl#3636 | `cd417288` (Oceananigans fix #6396 is in v0.117.1+) |
| 2 | Enzyme-JAX (Enzyme-MLIR) | Zero-in/zero-out loop-carried shadows in reverse `while` loops | 72.9 GiB of loop state; rolled time loop does not fit on a GPU | No (draft in `PROGRESS.md`) | Steps unrolled at trace time, then Revolve checkpointing |
| 3 | Enzyme-JAX pattern `while_deadresult` | 25 of 26 min of one rewrite application, removing 250 of 430k ops | Gradient compile 122 → 34 min without it | No | Excluded by default |
| 4 | Enzyme-JAX pattern `cse_reduce_transposed` (new in 0.2.303) | Quadratic candidate search; never fires on our modules | 2-step compile 53 → 100 min | No | Excluded by default |
| 5 | Reactant `WHILE_UNROLL_THRESHOLD` / `enzyme_hlo_unroll` | Unrolls after differentiation, not before | One rewrite application ran > 4 h 47 min | No | Unroll at trace time instead |
| 6 | Reactant on aarch64 (LLVM) | `fixup value out of range` on the first call of a large compiled thunk | Gradients cannot run on Vista GH200 | No | Run on x86_64 (LS6) |
| 7 | Reactant thunk call path (suspected) | First call of compiled thunks costs 13–40 min of host time | Fixed overhead per job | No (unconfirmed) | None |

## 1. Empty kernel becomes an undifferentiable `enzymexla.jit_call`

- **Symptom.** After updating to Reactant 0.2.303 / Reactant_jll 0.0.436 (from 0.2.290 / 0.0.413), the gradient
  failed to compile with `could not compute the adjoint for this operation "enzymexla.jit_call"`, 15 times.
  Every occurrence was `gpu__compute_y_bcs` for a (Face, Center, Center) tendency field with a `Flux{Nothing}`
  south boundary and a `Zipper` (tripolar fold) north boundary (jobs 3543330/1).
- **Cause.** That kernel writes nothing. The new raise pipeline turns it into an operand-free,
  side-effect-free `jit_call`. The forward pipeline deletes such a call, but the Enzyme pass meets it first and
  has no adjoint rule for it. Reactant's raise pipeline string is the same in both versions; the change is in
  Reactant_jll.
- **Expected.** An operand-free, side-effect-free `jit_call` has a zero adjoint. It should be differentiable
  as a no-op, or be removed before differentiation.
- **Status.** Reported by G. Wagner as EnzymeAD/Reactant.jl#3636, with a minimal reproducer. The Oceananigans
  side avoids launching the kernel (CliMA/Oceananigans.jl#6396, in v0.117.1 and later). OMIPAdjoint carries the
  same dispatch (`compute_y_bcs!(…, ::NotFluxBC, ::ZBC, …) = nothing`, `cd417288`) until its Oceananigans pin
  includes #6396.

## 2. Zero-in/zero-out loop-carried shadows in reverse `while` loops

- **Symptom.** The 2-step gradient with the time loop traced as one `@trace` loop did not fit on a 40 GB A100:
  XLA asked for 145.6 GiB (job 3511683, "only reduced to 144 GiB from 145.59 GiB"). The 1-step gradient fits.
- **Diagnosis** (HLO dumps, job 3516453). Enzyme's reverse step loop (`while.24`, trip count 2, a 294k-instruction
  body) carries 64,134 iteration arguments, 72.9 GiB (34,369 × `f64[20,60,180]`, 4,749 × `f64[20,62,182]`, …).
  Of these, about 62,700 are initialized with `broadcast(0)` and yielded as `broadcast(0)`. Each is the shadow
  of one forward intermediate: read once by an `add`, consumed, and reset to zero for the next iteration. The
  enzyme-ops remover promotes these body-local gradients (`enzyme.set`/`get` inside the loop) to loop-carried
  values. The checkpoint loop around it carries the same tuple, and XLA holds both the operand and the
  result, giving 145.8 GiB. At trip count 1, XLA inlines the loop before buffer assignment, which is why one
  step fits.
- **What does not remove them.** Enzyme-JAX's `while_simplify` and `while_deadresult`, and XLA's
  `WhileLoopSimplifier`.
- **Suggested fix.** A `while` canonicalization that replaces an iteration argument whose init and yield are
  the same splat constant with that constant inside the body (`while_deadresult` then drops it). Or an
  enzyme-ops remover that keeps a gradient first written by `+=` on a zero local as a local.
- **Evidence.** The dumped module `logs/attribution_coupled_gradient_2step.mlir` (47.9 MB, LS6). The report
  text is drafted in `PROGRESS.md` under "Ways out", item 2.
- **Consequence here.** A rolled loop cannot be differentiated at any useful length, so the gradient now
  compiles one step and drives it on the host with Revolve checkpointing (`src/checkpointed_gradient.jl`).

## 3. `while_deadresult` dominates the first post-Enzyme rewrite

- On the 1-step coupled gradient (430k ops, 14 loops after Enzyme), the first post-Enzyme application of the
  main pattern list took 1570 s. Applying the list minus pattern ranges, then single patterns, isolated
  `while_deadresult`: 147 s alone, removing 137 ops. It was also the ~100 s in every later fixed-point
  application (pattern applications 624 s → 88 s without it). Removing it changes 250 ops of 430k.
- Compile time without it: 122 → 34 min at 1 step (jobs 3512868 → 3516091/3516163). The coupled-versus-ocean
  2× gap in the MLIR stage was this pattern scaling worse on the coupled module.
- Tools: `pass_bisect.jl`, `compile_attribution.jl`, `mlir_diagnostics.jl` (this directory).
- Workaround: `CompileOptions(excluded_passes = ["while_deadresult", …])` in `adjoint_compile_options`.

## 4. `cse_reduce_transposed` is quadratic and never fires (new in 0.2.303)

- **Symptom.** The 2-step trace-time-unrolled gradient compiles in 99.6 min on 0.2.303 (job 3543332) against
  53.2 min on 0.2.290 (3536598), with bit-identical results. Every pattern application after Enzyme is 6–14×
  slower; tracing, Enzyme and XLA are the same or faster (attribution 3555512 against 3555525).
- **Bisect.** Stage dumps (3557933/4), then `pass_bisect.jl` on LS6 and locally on CPU, with the same
  Reactant_jll builds. The cost is in the pattern code, not the module: 0.2.303 patterns take 1189–1503 s on
  either version's module, 0.2.290 patterns 151–154 s. Removing the 44 patterns new in 0.2.303, then groups,
  then single patterns: only excluding `cse_reduce_transposed` restores the time (24 s against 164–166 s). The
  output is identical (461,900 ops), so the pattern never fires here.
- **Cause, from the source** (`CSEReduceTransposed`, `EnzymeHLOOpt.cpp`). For every `stablehlo.reduce`, it
  collects as candidates all reduces in the block that share its init value (one splat zero for the whole
  module). It then walks the input's elementwise computation and each candidate's. That is quadratic in the
  number of reduces times the size of their computations, paid on every application and every driver
  iteration even when nothing matches. The unrolled 2-step reverse pass has ~520–580 reduces over large fused
  stencil computations.
- **Workaround.** Excluded by default (`fb295782`); confirmed by job 3559769 (47.3 min).

## 5. `WHILE_UNROLL_THRESHOLD` unrolls after differentiation

- **Intent.** Unroll the 2-step forward loop before Enzyme, so the reverse pass is straight-line (a way around
  issue 2).
- **Observed** (bisect 3530023, twin 3530024). The threshold is a field of
  `EnzymeXLATransformPassesOptions`, so `enzyme_hlo_unroll` sits in every pattern set. Before Enzyme it left
  the forward `@trace` loop rolled: the module going into Enzyme equals the rolled 2-step one (72.6k ops, 4
  loops). After Enzyme, `enzyme_hlo_unroll(2)` (index 257 of 481) met the trip-count-2 reverse loops, which have
  64k iteration arguments and 294k-op bodies. That first post-Enzyme application ran 1 h 46 min (dev) and
  4 h 47 min (twin) without finishing; at 1 step it takes 90 s. The gradient compile with threshold 2 timed out
  at 5 h (3516665).
- **Workaround.** Unroll at trace time (a plain Julia `for`, `unroll_steps`), now superseded by checkpointing.

## 6. aarch64: `fixup value out of range` on the first call

- On Vista (GH200, aarch64), the 1-step gradient compiled in 96.2 min. The first call then ran 88 min at 0% GPU
  and exited with `error: <unknown>:0: fixup value out of range`, with no Julia stack trace (job 1068827). The
  primal of the same model runs.
- The first call is host-side compilation of the thunk glue. The gradient thunk unpacks the whole model and its
  shadow, and AArch64 branch and literal fixups reach only ±1 MB (x86_64 displacements ±2 GB). ReactOcean hit
  the same error on aarch64 when its atmosphere sat in the forcing parameters
  (`../omip_adjoint_handoff/bulk_forcing_handoff.md`).
- Workaround: gradients run on LS6 (x86_64) only. A smaller thunk (fewer untraced leaves; only controls and
  state as arguments) is the likely fix.

## 7. Suspected: first-call cost of compiled thunks (unconfirmed)

- With the checkpointed gradient, five compiled thunks take the model (and its shadow) as arguments. Their
  first calls cost 2241–2368 s on the JRA55-forced model, independent of window length (2 to 360 steps), and
  753–840 s on the smaller ECCO4-forced model. The repeat calls take 0.8–104 s for the same work. XLA logs
  nothing during the first call, and the GPU is idle.
- Most likely Julia compiling the thunks' call wrappers for the very large argument types (the same glue as
  issue 6). Not yet confirmed with `Base.cumulative_compile_time_ns`; not reported.

## Not included: Oceananigans/NumericalEarth meeting Reactant

These were fixed or worked around in model code. They are not Reactant or Enzyme defects:

- Host `Clock{Float64}` on the prescribed atmosphere and radiation under `ReactantState`, read as a trace-time
  constant. Fixed with `Clock(grid)` (`afcb086f`).
- Oceananigans' time interpolation `ifelse(n₁ == n₂, ψ₁, ψ̃)` returns a Float32/Float64 union for Float32
  slices, which Reactant cannot raise. Fixed with Float64 in-memory forcing (`1da8855f`).
- Oceananigans' generic `find_time_index` reads `times[n]` at a traced index for unevenly spaced (vector)
  times, which hits the scalar-indexing guard. Fixed with `@allowscalar` in OMIPAdjoint (`48db0af3`); belongs
  in OceananigansReactantExt.
- NaN adjoint entries on a few tripolar fold-seam columns. They do not reach the controls; not yet attributed.
