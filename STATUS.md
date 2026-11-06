# OMIP adjoint: status as of 2026-10-30

Branch `ajs/omip-adjoint` (fork GodotMisogi), tip `1b8969ff`, pulled on LS6. Records: `LOGBOOK.md` (decisions,
parameters, dead ends), `experiments/OMIPSimulations/adjoint/PROGRESS.md` (full detail). Both uncommitted.

## Where we are

| Rung (Nz = 20, Δt = 7200 s) | State |
|---|---|
| Primal, 1 and 2 steps | verified against the CPU twin: −1.3433 Sv and −5.3233 Sv, relative difference < 5e-9, repeat drift 0 |
| 1-step gradient | runs end to end on LS6 A100; exactly zero by construction (the window's fluxes come from the reset), NaN in the `u` and `S` state shadows; Taylor check reports MISMATCH as intended |
| 1-step gradient compile | 30 min on Reactant 0.2.303 (34 on 0.2.290; 123 before the `while_deadresult` exclusion, `b697ddf7`) |
| 2-step gradient, rolled loop | does not fit: Enzyme's reverse step loop carries 72.9 GiB of zero-in/zero-out shadow state, counted twice by XLA (145.6 GiB). Diagnosed from XLA's HLO dump, job 3516453 |
| 2-step gradient, `while_unroll_threshold = 2` | **Dead end as implemented.** Job 3516665 timed out at 5 h compiling; the bisect (3530023) shows the threshold never unrolled the forward loop (module into Enzyme = the rolled 2-step one, 4 loops) and instead the first post-Enzyme pattern application, with `enzyme_hlo_unroll(2)` (pattern 257) meeting Enzyme's trip-count-2 reverse loops (64k iteration args, 294k-op bodies), ran > 4 h 47 min without finishing (twin 3530024; 90 s at 1 step) |
| 2-step gradient, steps unrolled at trace time | **fits and runs** (job 3536598, 10-30): compile 53.2 min, first call 965 s, second 0.45 s, 31.6 GiB held, gradient finite and non-zero (max 4.97e-3 Sv per m/s), repeat drift 0, same NaN census as at 1 step (seam columns, not reaching the wind gradient). **Taylor MISMATCH**: FD slope 2.872e-2 vs adjoint 1.911e-2, residual first order |

## What the mismatch means

`evaluate!()` = `reset!` (restore snapshot, then the separately compiled `refresh_diagnostics!`, which computes
the step-1 fluxes from the winds at call time) followed by the differentiated window. The Taylor check sets
the winds before `evaluate!()`, so the finite difference includes the step-1 flux path and the adjoint does
not. The 1-step zero gradient was the same defect. Everything about the unrolled 2-step machinery itself
(memory, compile, finiteness, reproducibility) passed.

## Running / waiting on

Nothing queued. **The 0.2.303 2-step regression is `cse_reduce_transposed`** (new Enzyme-JAX pattern): LS6
bisect 3558732/3 (pattern code, not module: 0.2.303 patterns 1189–1503 s on either module, 0.2.290 151–154 s),
then local CPU bisects on the dumped modules: excluding that one pattern restores 0.2.290's times with identical
output (it never fires; quadratic candidate search over reduces sharing one init). Next: add it to
`adjoint_compile_options`' `excluded_passes`, confirm on LS6, report upstream.
2-step regression attributed (3555512 vs 3555525): post-Enzyme pattern application 297 → 2549 s
(the `InterpreterPass`es after `DifferentiatePass` in `all` and in `post_op_transpose_reshape`); Enzyme 666 → 266 s
and XLA 1692 → 1538 s got faster. 0.2.290 worktree kept at
`/scratch/09603/ajseth/NumericalEarth-0290` for the comparison. Reactant 0.2.303 / Reactant_jll 0.0.436 on LS6 since 10-30, with the OMIPAdjoint `Zipper`
override (`cd417288`; EnzymeAD/Reactant.jl#3636, CliMA/Oceananigans.jl#6396). Compile times against 0.2.290:

| | 1 step (3544007/8 vs 3516163/3516091) | 2 steps unrolled (3543332 vs 3536598) |
|---|---|---|
| model build | 842 s (≈ 19 min) | 929 s (1246 s) |
| primal compile | 466 s (≈ 8–9 min) | 824 s (925 s) |
| gradient compile | 29.9 min (33.8) | **99.6 min (53.2)**; 81.7 min gradient-only (3548971) |
| first gradient call | 778 s (903) | 818 s (965) |

All numbers (primal, gradient, census, Taylor values) are identical. The 2-step gradient-compile regression is
not attributed yet. Container: `$WORK/containers/julia_1.12_ls6.sif` links to a `/scratch` copy (mounting from
`/work` timed out on 11-03).

## Next

1. Move `refresh_diagnostics!` into the window so every step's fluxes are differentiated; `reset!` only
   restores state. Then rerun the 1-step rung on `gpu-a100-dev` (≈ 19 min build + 34 min compile + one
   `update_state!` + 15 min first call) for the first meaningful Taylor check, then the 2-step unrolled rung
   (8 h on `gpu-a100`).
2. Optional confirmation for the upstream report: segment-6 sub-bisect on
   `logs/bisect_coupled_2step_unroll2_all/05_post_enzyme_inline.mlir`, main list except 257 and 257
   alone (`patterns_except: [257, 257]`, `patterns: [257, 257]`), dev queue.
3. Report upstream to Enzyme-JAX: `while_deadresult`'s cost (25 min for 250 ops on a 430k-op module) and the
   loop-carried zero shadows (a while canonicalization for iteration arguments with equal splat init and
   yield). Until fixed, every rung runs unrolled, so compile time grows with the step count (34 → 53 min).
4. Compile time left: XLA 19 min (unattributed), model build 19–21 min (Julia compile; sysimage or
   PrecompileTools), first call 15–16 min (thunk size; also the aarch64 failure).
5. Then Nz = 70, then 8 steps, with times at each rung.
6. Local environment (Kaimon, Julia 1.13.3) resolves cleanly; the only incompatible compat entry is the root
   package's `LibCURL = "0.6.4"` (ignored by Pkg on 1.13; `"0.6.4, 1"` fixes it).

## Constraints to remember

- LS6 only for the gradient (aarch64 Vista fails in the first call). `gpu-a100-dev` ≤ 2 h, ~3 queued jobs per
  user; `gpu-a100` for longer runs. Push via the ssh remote, `git pull --ff-only fork ajs/omip-adjoint` on LS6,
  submit from `experiments/OMIPSimulations` with `SCRIPT=`/`CONFIG=` and `adjoint/omip_adjoint.slurm`.
- No environment variables for the Julia scripts; everything comes from the JSON configs.
- Keep `PROGRESS.md`, `LOGBOOK.md`, `STATUS.md`, `experiments/OMIPSimulations/Manifest.toml` and
  `omip_adjoint_handoff/` uncommitted.
