# OMIP adjoint logbook

Reverse-mode gradient (Reactant + Enzyme) of the AMOC at 26.5°N with respect to the JRA55 10-m winds, for the
2° ocean-only OMIP model, package `OMIPAdjoint` in `experiments/OMIPSimulations/adjoint/` (all paths below are relative to it), branch `ajs/omip-adjoint`. `PROGRESS.md`
holds the detailed record and tables; this logbook lists the decisions, the parameter values, what was tried
and dropped, and what is open. Dates are 2026.

## Decisions

| Date | Decision | Why |
|---|---|---|
| 10-12 | Move the ReactOcean AMOC-inversion work into NumericalEarth as `experiments/OMIPSimulations/adjoint/` with its own `Project.toml` | Differentiate through NumericalEarth's own coupling (`OceanOnlyModel`, `PrescribedAtmosphere`) instead of ReactOcean's bare ocean with a frozen atmosphere; ReactOcean's 1-step gradient was NaN |
| 10-12 | Ocean-only with real data (tripolar 2°, ETOPO, WOA, JRA55 repeat-year), no sea ice | User's scope choice |
| 10-12 | Pin Reactant to 0.2.290; select the CUDA 12.9 `Reactant_jll` build | The CUDA 13.1 build fails to load on the GH200 nodes; 12.9 also runs on LS6's 570 driver |
| 10-12 | Hold the JRA55 forcing in memory on a regular 640×320 latitude-longitude grid, 16 three-hourly slices | No file I/O and no Gaussian-latitude search inside the traced step |
| 10-12 | Close the Central American isthmus (λ ∈ [276, 284], φ ∈ [7, 12]) and define the AMOC section by longitude span 80°W–14°W | The 2° bathymetry leaves Panama open; `atlantic_ocean_basin` does not cut this grid (longitudes run 73°–433°, the Cape Agulhas barrier never lands) |
| 10-12 | Reset also runs the ocean model's `update_state!` (`433b2d12`) | SplitRK3 stage 1 reads tendencies that are not in the prognostic snapshot; without it the first primal gave −1.92 Sv and the repeat −1.38 Sv |
| 10-12 | Compile options `raise = true, raise_first = true, reshape_propagate = :none, transpose_propagate = :up` ("noreshape"), `WHILE_UNROLL_THRESHOLD = 0`, barotropic substeps rolled into one traced loop | Inherited from ReactOcean: the gradient compiled only with `reshape_propagate = :none`; `:none`/`:down` for both propagation options miscompiles the forward step; rolling the substeps halves the graph |
| 10-13 | Run on Vista GH200 and LS6 A100 from one worktree shared through `/work`, one Slurm script, per-cluster plain `julia:1.12.7` containers | Same code and depot on both clusters |
| 10-14 | Move the gradient rung to LS6 (x86_64) | The first gradient call dies on aarch64 with LLVM `fixup value out of range` after 88 min at 0% GPU (see dead ends) |
| 10-15 | A zero `gᵀd` is no longer reported as "GRADIENT VERIFIED" (`1535df3a`) | The 1-step Taylor check passed vacuously with g = 0 |
| 10-15 | Build diagnostics before optimizing: `compile_attribution.jl` (MLIR vs XLA stage per window variant), then per-pass timing and stage dumps, then `pass_bisect.jl` on a dumped module | The 122-min gradient compile had no measured breakdown; the bisect runs the pipeline in segments without a model build |
| 10-17 | Exclude Reactant's `while_deadresult` rewrite pattern by default in `adjoint_compile_options` (`b697ddf7`) | One pattern cost 25 of the 26 min of the first post-Enzyme rewrite and ~100 s in every later fixed-point application, to remove ~250 of 430k ops; the module changes by 0.06% without it |
| 10-17 | Diagnose the 2-step memory blow-up from XLA's HLO dump rather than from the MLIR module | The MLIR tensor census showed nothing above 20 MiB; the estimate is XLA's live-buffer view |
| 10-17 | Run the 2-step rung with the time-step loop unrolled before differentiation (`while_unroll_threshold = 2`) | Rolled, Enzyme's reverse step loop carries 72.9 GiB of zero-in/zero-out shadow state (see findings); unrolled, those are locals, as at 1 step |
| 10-27 | Unroll the window at trace time (`unroll_steps`, a plain Julia loop) instead of through Reactant's threshold | The threshold never unrolled the forward loop and blew up the post-Enzyme rewrite (dead ends); a trace-time unroll makes both the forward and the reverse code straight-line by construction |
| 10-30 | Update to Reactant 0.2.303 (Reactant_jll 0.0.436, EnzymeCore 0.8.26, Enzyme_jll 0.0.303+1; Enzyme already the newest, 0.13.216) and re-time the compile (`017f6230`) | User's request. Only the Reactant/Enzyme packages move (`Pkg.update(...; preserve = PRESERVE_ALL)`): the Oceananigans `ss/for-omip` branch is now at 0.117.0 and a fresh resolve fails on OMIPSimulations' compat. The 12.9 CUDA build and `while_deadresult` (526 patterns, 525 excluded) are still there |
| standing | `PROGRESS.md`, `LOGBOOK.md`, `../Manifest.toml` and `../omip_adjoint_handoff/` stay uncommitted | Local records and a local manifest |
| standing | Scripts take a JSON config, never environment variables | Project convention |

## Parameter values

Current rung (`configs/twodegree_nz20_*.json`):

| Parameter | Value |
|---|---|
| Grid | tripolar 180 × 60 × Nz, Nz = 20 (target 70) |
| Δt | 7200 s |
| Barotropic substeps | 50, rolled into one traced loop |
| Steps in the window | 1 (first rung), 2 (second rung; smallest window with a non-zero wind gradient) |
| Atmosphere slices in memory | 16 three-hourly JRA55 slices (covers 48 h) |
| Closure | CATKE, GM/Redi 800, biharmonic 50 days, Henyey background |
| Advection / time stepping | split WENO, explicit vertical, split-explicit free surface, SplitRK3 |
| Initial state | WOA annual T/S converted to TEOS-10 |
| Objective | AMOC at row 41 (26.4°N, 31 wet columns), 1000 m, 80°W–14°W |
| Enzyme mode | `set_strong_zero(ReverseWithPrimal)`, `Duplicated(model, dmodel)`, wind gradient from the 10-m velocity slices |
| `@trace` loop flags | `mincut = true, checkpointing = true, track_numbers = false` |
| Taylor check | `h0 = 1.0`, `nh = 6`, along `d = g/|g|` on interior wind entries |
| Reactant compile options | `raise`, `raise_first`, `reshape_propagate = :none`, `transpose_propagate = :up`, `excluded_passes = ["while_deadresult"]` |
| While-unroll threshold | 0 for the 1-step rung; 2 (= steps) for the unrolled 2-step run |
| Cluster | LS6 `gpu-a100-dev` (≤ 2 h, ~3 jobs queued per user) or `gpu-a100`; allocation `Transfer-of-ECCO-glo`; GPU 0 only |

Reference numbers (LS6 A100, Nz = 20):

| Quantity | Value |
|---|---|
| AMOC after 1 step, CPU twin and Reactant primal | −1.3433 Sv (relative difference 3.47e-9, repeat drift 0) |
| AMOC after 2 steps | −5.3233 Sv (4.71e-9) |
| Model build | ~1120 s, ~92% Julia method compile (interfaces + `OceanOnlyModel` 629 s) |
| Primal compile + first call | 524 + 461 s |
| 1-step gradient compile | 122.8 min → 33.8 min after the `while_deadresult` exclusion |
| 1-step gradient first / second call | 903 s / 1.73 s |
| 1-step coupled gradient module | 439k StableHLO ops, 14 while loops, largest tensor 20 MiB |
| 2-step coupled gradient MLIR stage | 6320 s (old default) → 961 s |

## Dead ends and false leads

- **Vista (aarch64) for the gradient.** Compiles in 96 min, then the first call dies with LLVM
  `fixup value out of range` (job 1068827): the thunk that unpacks the whole model and its shadow exceeds
  AArch64's ±1 MB fixup range. Not fixed; the lever is a smaller thunk (pass only controls and state).
- **ReactOcean's compile fixes as the explanation of the 20× compile gap.** `WHILE_UNROLL_THRESHOLD = 0` and
  the rolled substeps were already in; the gap was elsewhere.
- **"A dead trailing `update_state!` inflates the coupled module."** The coupled gradient's MLIR stage took
  twice the ocean-only one for a module 2% larger; the cause was `while_deadresult` scaling worse on the
  coupled module. Without it both take ~900 s.
- **"XLA code generation dominates the compile."** ReactOcean's earlier diagnosis; here the MLIR pattern
  rewriting was 75% of the MLIR stage and 51 of 69 min of the ocean gradient compile. After the exclusion,
  XLA (19 min) is the larger half and remains unattributed.
- **Chunking the pattern list to find the slow pattern.** Applying the 482 patterns in 8 chunks made the
  following lowering set take 78 min instead of 2.5: the cost sat in the convergence chain between pattern
  groups, so chunk timings alone misled. Control runs (whole list minus a range) located it: minus chunk 3
  (121:181) 41 s instead of 1570 s; minus 159:166 55 s; `while_deadresult` alone 147 s.
- **Vacuous Taylor check.** With g = 0 the direction is 0 and Δf = 0 = h gᵀd; "GRADIENT VERIFIED" meant nothing.
- **Finding the 145 GiB in the MLIR module's tensor types.** The 2-step modules (ocean-only 429,595 ops,
  coupled 438,775) have the same tensor types as at 1 step, all ≤ 20 MiB; the per-iteration caches (leading
  dimension 2) sum to ~1.5 GiB. The static census cannot see loop-state growth.
- **Disabling XLA's `rematerialization` pass to reach buffer assignment.** With it off, XLA still ran past
  the 2 h limit after autotuning (job 3516453); the HLO dumps before optimization and before config
  assignment were enough instead.
- **`--xla_dump_to` via `XLA_FLAGS`.** Not used (no environment variables); the attribution script maps a
  config's `xla_debug_options` onto Reactant's `CompileOptions(xla_debug_options=...)` → `DebugOptions` proto.
- **`while_unroll_threshold = steps` as "unroll before differentiation".** The pattern never touched the
  forward `@trace` loop (bisect 3530023: the module into Enzyme equals the rolled one) and instead fired
  post-Enzyme on the trip-count-2 reverse loops, where the first pattern application ran > 1 h 46 min
  (90 s with threshold 0). Jobs 3516665, 3520928, 3520929, 3530023 all timed out there.
- Jobs cancelled as redundant: `gpu-a100` twins of dev runs (3515268, 3515590, 3515917, 3516059, 3516090,
  3516092) and the 1-step per-pass run 3515042 once its question was answered.

## Findings

1. **The 1-step wind gradient is exactly zero by construction.** `time_step!(::EarthSystemModel)` steps the
   ocean with the fluxes already in place, then computes the fluxes for the next step; a 1-step window uses
   fluxes from the reset, outside the differentiated region. Two steps is the smallest window that can see the
   winds. The ocean-state shadows carry NaN (40 `u` entries over all levels, 116 `S` entries over 8 levels);
   the census prints their `(i, j)` columns to separate the tripolar fold, the closed isthmus and coasts.
2. **Compile time.** Enzyme's output is 7× the primal (71k → 513k ops, 18 loops). The greedy rewriter costs
   ~100 s per application on the 430k-op module even at a fixed point, and `while_deadresult` drove the first
   application to 1570 s. Remaining after the exclusion (coupled 1-step): Enzyme 310 s, tracing ~70–210 s,
   Julia compile ~180 s, XLA 19 min, model build 19 min.
3. **The 2-step memory blow-up** (job 3511683: "only reduced to 144 GiB from 145.59 GiB"). In the HLO,
   Enzyme's reverse step loop (`while.24`, trip count 2, 294k-instruction body) carries 64,134 state
   elements, 72.9 GiB: 34,369 × `f64[20,60,180]`, 4,749 × `f64[20,62,182]`, ... About 62,700 are initialized
   with `broadcast(0)` and yielded as `broadcast(0)`: the shadow of one forward intermediate each, read once
   by an `add`, consumed, and reset. These are body-local gradients promoted to iteration arguments by the
   enzyme-ops remover; their value at every iteration boundary is zero. The checkpoint loop around it carries
   the same tuple (72.6 GiB). XLA holds the while operand and result: 145.8 GiB. At 1 step the loop has trip
   count 1 and XLA inlines it before buffer assignment, which is why that gradient fit in 40 GB. Neither
   Enzyme's `while_simplify`/`while_deadresult` nor XLA's `WhileLoopSimplifier` removes a zero-in/zero-out
   element. Candidate upstream fix: a while canonicalization replacing an iteration argument whose init and
   yield are the same splat constant by that constant in the body.

4. **The Taylor check and the adjoint differentiate different functions (job 3536598).** `evaluate!()` runs
   `reset!` (restore the snapshot, then the separately compiled `refresh_diagnostics!`, i.e. the coupled
   `update_state!`, which computes the interface fluxes from the *current* winds) and then the differentiated
   window. The check sets the winds before `evaluate!()`, so the finite difference sees the winds through the
   step-1 fluxes (computed in the reset) and the step-2 fluxes (computed inside the window), while the adjoint
   sees only the latter. At 2 steps: FD slope 2.872e-2, adjoint 1.911e-2, residual |Δf − h gᵀd| exactly
   first order in h (ratio 2.00 per halving), Δf/h constant to 1e-4. The 1-step zero (finding 1) is the same
   defect at the limit. Everything else about the unrolled 2-step gradient is as hoped: it fits in 40 GB,
   compiles in 53.2 min (34 at 1 step), first call 965 s, second 0.45 s, gradient finite, non-zero, repeat
   drift 0; the NaN census is unchanged (40 `u`, 116 `S`), with the NaN columns at i ∈ {1, 90, 91, 180},
   j ∈ {59, 60, 61} (the seam columns), and they do not reach the wind gradient.

## Tooling added

| File | Purpose |
|---|---|
| `compile_attribution.jl` + `mlir_diagnostics.jl` | Model build by stage; per window variant (`fluxes`, `ocean`, `coupled`) and mode, the MLIR stage vs XLA stage, Julia compile inside each, op histogram and largest tensor types; switches `pass_timing`, `dump_stages`, `excluded_passes`, `xla_debug_options` |
| `pass_bisect.jl` | Runs Reactant's `all` pipeline on a dumped module in named segments (`pipeline`, `patterns: [i, j]`, `patterns_except`, `patterns_split`, per-segment `module` restart) with time and op count per segment |
| `test_omip_adjoint.jl` | Stages `cpu`, `primal`, `gradient`; config keys `excluded_passes`, `while_unroll_threshold` |
| `omip_adjoint.slurm` | One script for Vista and LS6; `SCRIPT=` and `CONFIG=` select the run |
| Session scratch (not in the repo) | `hlo_census.py`, `while_state.py`, `while_sample.py`: computation output bytes, while-state histograms and body classification on an XLA HLO text dump |

## Jobs (LS6 unless noted)

| Job | What | Outcome |
|---|---|---|
| Vista 1068543 / 1068827 | primal / gradient | primal passes; gradient first call dies (aarch64 fixup) |
| 3506971 | primal | −1.3433 Sv |
| 3509300, 3510442 | 1-step gradient | runs; gradient exactly zero; compile 122.8 min |
| 3511683 | 2-step gradient (rolled) | out of memory in XLA (145.6 GiB), 5 h limit |
| 3512868 | compile attribution, 1 step | MLIR 3093 / 6272 s, XLA ~1050 s |
| 3515041 | per-pass timing, 1-step ocean | post-Enzyme rewriting 1940 s of 3338 s |
| 3515386, 3515623, 3515916, 3516058 | pass bisect: segments, chunks, chunk 3, singles | `while_deadresult` |
| 3516091 | attribution without `while_deadresult` | MLIR 883 / 909 s |
| 3516163 | 1-step gradient test, new default | compile 33.8 min, same zero gradient and census; Taylor MISMATCH as intended |
| 3514047 | 2-step MLIR-only attribution (old default) | ocean 3370 s, coupled 6320 s; no oversized tensor type |
| 3516453 | 2-step XLA stage, remat off, HLO dump | timed out in XLA; dumps gave finding 3 |
| 3516665 | 2-step gradient, loop unrolled (`gpu-a100`, 5 h) | primal −5.3233 Sv; gradient compile still running at the 5 h limit (4 h 18 min in, 0% GPU, 31.6 GiB held = XLA's reservation); no module dump (time limit, not a failure) |
| 3520928 / 3520929 | MLIR-stage attribution of the unrolled coupled 2-step gradient, per-pass timing + stage dumps (`gpu-a100-dev` 2 h / `gpu-a100` 5 h twin) | both TIMEOUT inside the `all` pipeline: tracing 4 min, then > 4 h 29 min in `all` (732 s at 1 step, 961 s rolled 2-step); traced module dumped, identical in size to the 1-step coupled one |
| 3536598 | 2-step gradient, steps unrolled at trace time (`unroll_steps`; `gpu-a100`, 8 h) | ran 10-30 11:11–13:13 (c310-003): **fits** (31.6 GiB held = XLA's reservation), gradient compile 53.2 min, first call 965 s, second 0.45 s, primal −5.3233 Sv, gradient finite and non-zero (max 4.97e-3 Sv per m/s), repeat drift 0; Taylor MISMATCH: FD slope 2.872e-2 vs adjoint 1.911e-2 (ratio 1.503, residual first order). Cause: the check's perturbation also enters the reset's flux computation, outside the window (finding 4) |
| 3543329 | update the LS6 adjoint manifest to Reactant 0.2.303 (`gpu-a100-dev`, 45 min; one-off script `/scratch/09603/ajseth/omip_adjoint_update/update_reactant_enzyme.jl`; old manifest kept as `logs/Manifest.before_reactant_0.2.303.toml`) | FAILED in 5 min: LS6's registry knows Reactant only up to 0.2.295. Julia 1.12's `Pkg.update` refreshes registries in the first depot only (`depots = [depots1()]`), and General lives in the second, Vista's, depot (packed, last refreshed 10-18). Manifest unchanged |
| 3543579 | the update again, with `Pkg.Registry.update(; depots = <depots holding General>)` first | COMPLETED 10-30 23:02–23:11 (9 min, c301-001): Reactant 0.2.303, ReactantCore 0.1.26, Reactant_jll 0.0.436+0, Enzyme 0.13.216, EnzymeCore 0.8.26, Enzyme_jll 0.0.303+1; Oceananigans stays 0.113.5; OMIPAdjoint precompiled (43 packages, 220 s) |
| 3543330 | 1-step test on Reactant 0.2.303 (vs 3516163) | FAILED 10-30 23:37, 49 min. CPU twin −1.3433 Sv; Reactant model built in 809.5 s; primal compiled in 476.8 s, first call 428.8 s, −1.3433 Sv (rel. diff 1.58e-7 vs CPU), repeat drift 0. Gradient compile: `could not compute the adjoint for this operation "enzymexla.jit_call"` (15×, all one kernel: `gpu__compute_y_bcs` for the (Face, Center, Center) tendency, south `Flux{Nothing}`, north `Zipper`) |
| 3543331 | 1-step attribution on 0.2.303 (vs 3516091) | FAILED 10-31 03:17, 27 min: build grid 70.7 s, ocean 222.6 s, forcing 53.8 s, model 512.2 s; same `jit_call` error on the ocean gradient |
| 3543332 | 2-step unrolled on 0.2.303 (vs 3536598, `gpu-a100` 8 h) | ran 11-04 13:07–15:46 on c317-010, exit 1 at the Taylor `@test` only. Environment 5.8 min, model build 928.6 s (1246), primal compile 823.5 s (925), primal first call 430.1 s (457), −5.3233 Sv, **gradient compile 99.6 min (53.2)**, first call 817.9 s (965), second 0.49 s. Gradient, census (40 `u`, 116 `S`), gᵀd 1.91137470e-2 and FD slope 2.87207311e-2 identical to 3536598 |
| 3548754 | 2-step unrolled, gradient stage only (`twodegree_nz20_2step_unrolled_gradient.json`, `87499032`), `gpu-a100-dev` 2 h | FAILED 11-03 04:26 in 17 s on c301-004: apptainer `squashfuse_ll failed to mount` the container (node fault) |
| 3548900 | 3548754 again with `--exclude=c301-004` | FAILED 11-03 06:14 in 15 s on c301-001, the same `squashfuse_ll failed to mount … in 10s`; so not one node. Image unchanged since 10-13; copying it off `/work` ran at ~5 MB/s (327 MB in 64 s), so `/work` is slow |
| 3548971 | again with `SIF=/scratch/09603/ajseth/containers/julia_1.12_ls6.sif` (copy, `cmp`-verified) | TIMEOUT 11-03 06:52–08:52 on c301-001 (mount fine): environment 7.3 min, Reactant model built in 913.1 s, 2-step gradient compiled in **81.7 min** (0.2.290, after the primal compile: 53.2), killed in the first call |
| 3555512 | 2-step `unroll_steps` coupled-gradient attribution on 0.2.303 (`gpu-a100-dev`) | COMPLETED 11-05, 110 min: MLIR 3361.7 s (`all` 1846.4, `post_op_transpose_reshape` 1331.4, tracing 178.5) + XLA 1538.3 s = 81.7 min; 464,435 ops, 12 while loops |
| 3555511 | the same on `gpu-a100`, 4 h | cancelled 11-05 (3555512 completed) |
| 3555525 | the same on 0.2.290 (worktree `/scratch/09603/ajseth/NumericalEarth-0290`, `gpu-a100-dev`) | COMPLETED 11-05, 84 min: MLIR 1550.4 s (`all` 1152.9, `post_op_transpose_reshape` 180.4, tracing 211.5) + XLA 1691.9 s = 54.0 min; 460,379 ops, 12 while loops |
| 3544007 | 1-step test, 0.2.303 + `cd417288` (vs 3516163) | exit 1 at the Taylor `@test` only, as on 0.2.290. Reactant model build 841.7 s, primal compile 466.1 s, primal first call 429.6 s, −1.3433 Sv (rel. diff 1.58e-7), gradient compile **29.9 min** (33.8), first call **778.1 s** (903), second 0.42 s, gradient zero (by construction), repeat drift 0; census unchanged (40 `u`, 116 `S` NaN, seam columns) |
| 3544008 | 1-step attribution, 0.2.303 + `cd417288` (vs 3516091) | COMPLETED, 77.5 min (88). Ocean MLIR 724.5 s (883) + XLA 918.0 s (1092). Coupled MLIR 806.9 s (909) + XLA 959.6 s (1124); inside: `all` 615.7 s (732), `post_op_transpose_reshape` 119.0 s (97), tracing 59.6 s (69). Module 438,905 ops (438,915), 14 while loops, no kernel_call |
| 3530023 / 3530024 | `all` bisect on the dumped unrolled 2-step traced module, 10 segments, threshold 2, `while_deadresult` excluded (`gpu-a100-dev` 2 h / `gpu-a100` 5 h twin) | dev run TIMEOUT: segments 1–5 in 7 min with the 1-step op counts and loop counts (the forward loop was **not** unrolled), then segment 6 `post_enzyme_main_patterns` ran 1 h 46 min to the cut (90 s at 1 step). Twin (10-25, c305-002) TIMEOUT too: segments 1–5 identical, segment 6 ran 4 h 47 min without finishing |
| 3557933 / 3557934 | 2-step `unroll_steps` MLIR stage with stage dumps, no XLA (`compile_attribution_nz20_2step_unroll_steps_dump.json`), 0.2.290 worktree / 0.2.303 | COMPLETED 11-05/06, 63 / 85 min: MLIR 1635.8 / 3487.8 s; modules `module_002_8Gmb_pre_all`, `module_004_T6dv_pre_post_op_transpose_reshape` under `logs/attribution_stages_coupled_gradient_2step_unrolled/` |
| 3558732 / 3558733 | `pass_bisect` on those (`pass_bisect_coupled_2step_unrolled_{290,303}.json`, `1ec855b9`) | COMPLETED 11-06, 40 / 88 min. `post_op_transpose_reshape` own module 150.6 / 1188.9 s, other version's module 154.1 / 1503.4 s; pre-Enzyme 280.3 / 297.8 s; Enzyme 532.2 / 247.5 s; post-Enzyme `opt_passes2` 107.6 / 821.4 s, again 52.0 / 401.5 s |
| local (Mac CPU, 11-06) | single main-list applications on the dumped modules, 0.2.290 and 0.2.303 envs | **`cse_reduce_transposed`** is the regression: excluding it alone 24.1 s vs 161.8 s full (0.2.290 7.9 s), identical output; whole `post_op_transpose_reshape` 68.6 s without vs 574.9 s with; post-Enzyme `opt_passes2` 43.6 vs 313.4 s |

## Open

- The differentiated window misses the first step's fluxes (finding 4): the reset computes them from the
  controls outside the window, so the gradient is wrong by construction (zero at 1 step, two thirds of the
  finite-difference slope at 2 steps). Next: move `refresh_diagnostics!` (the coupled `update_state!`) into
  the window so every step's fluxes are differentiated, and rerun the 1-step rung (cheapest: 34 min compile
  plus one `update_state!`) for the first meaningful Taylor check. The alternative, perturbing the winds
  after the reset in the check, would only verify the window's adjoint, not the gradient wanted.
- Optional confirmation of pattern 257 with a segment-6 sub-bisect (main list except 257, and 257 alone,
  on `logs/bisect_coupled_2step_unroll2_all/05_post_enzyme_inline.mlir`).
- Report upstream (Enzyme-JAX): `while_deadresult`'s cost; zero-in/zero-out loop-carried shadows;
  `cse_reduce_transposed`'s quadratic candidate search (the 0.2.303 2-step regression, 11-06).
- Exclude `cse_reduce_transposed` in `adjoint_compile_options` and confirm the 2-step compile returns to ~54 min.
- Attribute XLA's 19 min; cut the 19-min model build (sysimage or PrecompileTools); shrink the gradient thunk
  (first call ~15 min; Vista failure).
- Fix `atlantic_ocean_basin` for this tripolar grid in NumericalEarth.
- Then Nz = 70 and 8 steps, with compile and evaluation times at each rung.

## Entries

### 2026-10-17, 23:35

- Collected job 3514047 (2-step MLIR-only attribution, old default): ocean-only 3370 s / 429,595 ops, coupled
  6320 s / 438,775 ops; every tensor type ≤ 20 MiB in both, so the MLIR census could not locate the 145.6 GiB.
- Added `xla_debug_options` to `compile_attribution.jl` (`7109a76b`) and ran the 2-step coupled XLA stage with
  `rematerialization` disabled and `xla_dump_to` on (job 3516453, dev queue). MLIR 961 s with the new default.
  XLA passed autotuning at 10:51 and was still compiling at the 2 h limit; buffer assignment never ran.
- Censused the two HLO dumps (`before_optimizations`, `before_config_assignment`): the reverse step loop's
  state is 72.9 GiB (64,134 elements, ~62,700 of them zero-in/zero-out shadow slots of forward intermediates),
  held twice by XLA = 145.8 GiB. Recorded in `PROGRESS.md` ("The 2-step gradient") and in Findings above.
- Added `while_unroll_threshold` to `test_omip_adjoint.jl` and `configs/twodegree_nz20_2step_unrolled.json`
  (`bb7d80d2`); submitted as job 3516665 on `gpu-a100`, 5 h, stages `primal,gradient`. Still pending at
  23:33 (reason Priority; 253 jobs pending in the partition). The dev queue was not used: the unrolled
  2-step compile plus the first call is expected to exceed 2 h.
- Nothing reported upstream yet; the Enzyme-JAX report text is drafted in `PROGRESS.md` (ways out, item 2).

### 2026-10-19, 16:20

- Collected job 3516665 (unrolled 2-step gradient, `gpu-a100`, ran 2026-10-18 20:02 to 01:02 on c302-004,
  TIMEOUT). Model build 1180 s, primal compile 494 s, primal 1 in 468 s: −5.3233 Sv, repeat drift 0 (same as
  the rolled run 3511683). The gradient `@compile` started at 20:43 and had not finished at the 5 h limit
  (01:02): 4 h 18 min, against 34 min for the 1-step gradient and the expected ~2×. GPU at 0% throughout,
  31.6 GiB held (XLA's 29.6 GiB reservation plus the model), so no sign of the 145 GiB blow-up, and no
  memory error. No Reactant dump (the job was killed, not failed). Logs: `adjoint/logs/OMIPUnroll2.{o,e}3516665`.
- Where the time went is not visible from the logs (see Open). `WHILE_UNROLL_THRESHOLD` enters every
  Reactant pattern set (`EnzymeXLATransformPassesOptions`), so the unroll runs inside the greedy rewriter,
  pre- and post-Enzyme, on a module that then doubles to ~860k ops; the fixed-point rewriter costs and
  Enzyme's `DifferentiatePass` scale with that, but nothing measured says which stage blew up.
- `compile_attribution.jl` takes `while_unroll_threshold` from the config (`2c8ff3df`; the tag of the kept
  module and stage dumps gains `_unroll<n>` so the rolled 2-step module stays). Config
  `compile_attribution_nz20_2step_unrolled.json`: coupled 2-step gradient, MLIR stage only, per-pass
  timing and stage dumps. Submitted 16:35 as 3520928 (`gpu-a100-dev`, 2 h) and 3520929 (`gpu-a100`, 5 h
  twin), both pending.

### 2026-10-22

- Collected 3520928 (ran 10-19 19:39 to 21:39, c301-001) and 3520929 (10-22 03:32 to 08:32, c315-010): both
  TIMEOUT. Model build 627–631 s; tracing done ~4 min after "attributing coupled gradient" (stage dumps
  `pre_raise_triton`, `post_raise_triton`, `pre_all` written at 04:03 in the twin); then the `all`
  pipeline ran until the limit, 4 h 29 min in the twin, without writing its `post_all` dump. The only
  per-pass report on stderr is the 11.8 s `raise_triton` pipeline. So the whole excess is inside `all`
  (pre-Enzyme patterns including the unroll, Enzyme, post-Enzyme patterns), which took 732 s at 1 step and
  961 s for the rolled 2-step. GPU 0% throughout, 30.8 GiB held (reservation only).
- The traced module `logs/attribution_stages_coupled_gradient_2step_unroll2/module_002_8Gmb_pre_all_pm.mlir`
  (75.3 MB, 244,600 assignments, 1 `enzyme.autodiff`) is the same size as the traced 1-step coupled module
  (75.3 MB, 244,600): the unroll happens inside `all`, not at trace time, so the rolled 2-step and the
  unrolled 2-step start from the same module and differ only in the unroll threshold seen by the patterns.
- Next: `pass_bisect.jl` on that dumped module with the `all` pipeline in segments (as
  `pass_bisect_ocean_1step_all.json`), to see whether the pre-Enzyme unroll, Enzyme's `DifferentiatePass`
  on the doubled module, or the post-Enzyme rewriter is the cost. The script hard-codes
  `WHILE_UNROLL_THRESHOLD[] = 0` and builds `CompileOptions` without the `while_deadresult` exclusion, so
  both need config keys first.
- `pass_bisect.jl` takes `while_unroll_threshold` and `excluded_passes` from the config (`4165f876`).
  Config `pass_bisect_coupled_2step_unroll2_all.json`: the 10 segments of the 1-step `all` bisect on
  `module_002_8Gmb_pre_all_pm.mlir` with threshold 2 and `while_deadresult` excluded. Submitted as
  3530023 (`gpu-a100-dev`, 2 h) and 3530024 (`gpu-a100`, 5 h twin), both pending.

### 2026-10-24

- Collected 3530023 (ran 10-23 04:00 to 06:00, c301-001, TIMEOUT). Parsed the traced module (173,479
  ops, 4 while loops) in 2.7 s. Segments, with the 1-step ocean bisect (job 3515386) for comparison:

  | Segment | Time | Ops after | While loops | 1-step: time, ops |
  |---|---|---|---|---|
  | `memory_effects_opt` | 33 s | 66,425 | 4 | 33 s, 64,241 |
  | `kernel_raise` | 73 s | 72,734 | 4 | 71 s, 71,048 |
  | `batch_opt2` | 3 s | 72,625 | 4 | 3 s, 70,944 |
  | `enzyme` | 285 s | 523,574 | 18 | 282 s, 512,567 |
  | `post_enzyme_inline` | 8 s | 451,030 | 14 | 8 s, 441,697 |
  | `post_enzyme_main_patterns` | **> 1 h 46 min, cut** | | | 1570 s with `while_deadresult`, ~90 s without |

  So `while_unroll_threshold = 2` did **not** unroll the forward time-step loop: the module entering
  Enzyme has the same 4 loops and 2% more ops than at 1 step, exactly the rolled 2-step module. The
  whole excess is the first post-Enzyme pattern application with the threshold at 2.
- Reactant's main pattern list (481 entries, built locally with threshold 2) holds `enzyme_hlo_unroll(2)`
  at index 257, in both the pre- and the post-Enzyme applications. Pre-Enzyme it left the `@trace`
  loop alone (reason not yet checked: the loop's form under `checkpointing = true`, or its trip count
  not visible to the pattern). Post-Enzyme it meets the reverse loops with trip count 2, 64k iteration
  arguments and 294k-op bodies (finding 3), and the greedy driver does not return in 1 h 46 min.
- Conclusion: "unroll before differentiation" through `WHILE_UNROLL_THRESHOLD` does not do that here; it
  unrolls (or tries to) after differentiation, at the worst possible place. Unrolling at trace time
  instead (a plain Julia `for` in `window_objective` in place of `@trace for`, so the forward code is
  straight-line and Enzyme's reverse pass is too) is the way to test the unrolled rung. The twin 3530024
  is left in the queue to see whether segment 6 ever finishes in 5 h and how large its output is.

### 2026-10-26

- Collected the twin 3530024 (ran 10-25 11:18 to 16:18, c305-002, TIMEOUT): segments 1–5 reproduce the
  dev run to the op (enzyme 274 s), then segment 6 `post_enzyme_main_patterns` ran **4 h 47 min** without
  finishing. No module after segment 6, so its size is unknown. Nothing of ours left in the queue.

### 2026-10-27

- `unroll_steps` (`1b8969ff`): `unrolled_window_objective` is the window with a plain Julia `for`
  (unrolled while tracing); `window_function(unroll_steps)` picks it or the `@trace` one in
  `build_primal`/`build_gradient`, and `gradient_core!` takes the window as its first argument.
  `test_omip_adjoint.jl` reads `unroll_steps` from the config. Loads locally; the selector and
  signatures checked. `twodegree_nz20_2step_unrolled.json` now uses `unroll_steps = true` with the
  threshold at 0 (same output path as before).
- Submitted as job 3536598 (`gpu-a100`, 8 h rather than 5 so the gradient compile cannot be cut),
  stages primal + gradient. Pending (Resources).

### 2026-10-30

- Job 3536598 collected (ran 10-30 11:11 to 13:13 CDT on c310-003, exit 1 from the Taylor `@test`): model
  build 1246 s, primal compile 925 s, primal −5.3233 Sv (first call 457 s, repeat 0.1 s, drift 0),
  gradient compile **53.2 min**, first call 965 s, second 0.45 s, repeat |ΔJ| = |Δg| = 0, max |g|
  4.973e-3 Sv per m/s, all finite; peak device memory 31.6 GiB (the reservation). Census as at 1 step
  (40 `u`, 116 `S` NaN; columns listed in the log), now with finite non-zero shadows in the atmosphere
  fields and radiation too. Taylor: f0 −5.3233, gᵀd +1.9114e-2, FD slope +2.8721e-2, rel_err 0.503,
  MISMATCH. Logs `logs/OMIPUnroll2T.{o,e}3536598`, results
  `results/omip_adjoint_twodegree_nz20_2step_unrolled.jld2` (on LS6).
- Diagnosis (finding 4): the residual is first order and Δf/h is constant, so the primal is smooth and the
  adjoint is short by a fixed part of the sensitivity, the one through the step-1 fluxes that `reset!`
  computes from the perturbed winds outside the window. The fix is to differentiate the flux computation:
  `refresh_diagnostics!` moves into the window. Not applied yet.
- Local environment check (Kaimon session, Julia 1.13.3): `Pkg.resolve()` on a scratch copy of
  `../Project.toml` succeeds with no package change beyond stdlib JLL bumps; the "project has changed"
  warning is only the 1.12 → 1.13 project-hash scheme. One ignored compat entry in the root package:
  `LibCURL = "0.6.4"` (needs `"0.6.4, 1"` on 1.13).
- Reactant 0.2.290 → 0.2.303 (`017f6230`): `Reactant = "=0.2.303"`, `Enzyme = "0.13.216"` in
  `Project.toml`. Locally the targeted update moves Reactant, Reactant_jll (0.0.413 → 0.0.436), EnzymeCore
  (0.8.23 → 0.8.26) and Enzyme_jll (0.0.303+1) plus stdlib JLLs; OMIPAdjoint loads, every Reactant
  internal the scripts use still exists, the default `WHILE_UNROLL_THRESHOLD` is now 4 (ours sets 0).
  `Pkg.resolve()` alone refuses ("empty intersection between Reactant@0.2.290 and project compatibility"),
  so the LS6 manifest is moved by job 3543329, and the three timing jobs wait on it with `afterok`.
- Update job 3543329 failed: LS6 resolved against Vista's General registry from 10-18 (Reactant ≤ 0.2.295),
  because `Pkg.update` only refreshes registries in the first depot and LS6's has none. The one-off script now
  refreshes the depot that holds General first (Vista's shared registry becomes newer too; harmless, manifests
  pin versions). Resubmitted as 3543579; the three timing jobs retargeted to it with `scontrol update`.
- Update 3543579 completed in 9 min; the LS6 manifest is on Reactant 0.2.303 / Reactant_jll 0.0.436 /
  Enzyme_jll 0.0.303+1 with Oceananigans unchanged at 0.113.5. The three timing jobs are released.
- 0.2.303 regression (jobs 3543330/1): the y-boundary flux kernel of a field with a `Zipper` north and no-flux
  south boundary writes nothing; raised, it becomes an operand-free side-effect-free `enzymexla.jit_call` that
  the forward pipeline deletes but the Enzyme pass cannot differentiate. Reported upstream the same day by
  glwagner: EnzymeAD/Reactant.jl#3636 (MWE) and CliMA/Oceananigans.jl#6396 (adds `ZBC` to `NotFluxBC`). Reactant's
  own raise pipeline string is identical between 0.2.290 and 0.2.303; the change is in Reactant_jll 0.0.413 → 0.0.436.
  Fix `cd417288`: OMIPAdjoint adds `compute_y_bcs!(…, ::NotFluxBC, ::ZBC, …) = nothing` (same dispatch as #6396,
  checked locally: u, v and c on a CPU TripolarGrid all dispatch to it, no ambiguities), with a TODO to drop it
  once the pinned Oceananigans includes #6396. Primal on 0.2.303 already shows: model build 809.5 s, primal
  compile 476.8 s, first primal call 428.8 s.
- Reactant 0.2.303 compile times (1 step, Nz = 20, with `cd417288`): gradient compile 33.8 → 29.9 min (−12 %),
  coupled MLIR 909 → 807 s (`all` 732 → 616 s, `post_op_transpose_reshape` 97 → 119 s), coupled XLA 1124 → 960 s,
  first gradient call 903 → 778 s, primal compile ≈ 8–9 → 7.8 min. Numbers (primal, census, zero gradient) are
  unchanged. 2-step unrolled rerun 3543332 still queued on `gpu-a100`.
- 11-03: 3543332 still queued after 3.5 days (400 pending on `gpu-a100`). Job 3536598 ran 2 h 02 min of which the
  primal stage was 23 min, so a gradient-only 2-step run fits the dev queue on 0.2.303: 3548754. 3543332 stays
  queued as a fallback; cancel it if 3548754 completes.
- 11-03: the container mount failures (3548754 on c301-004, 3548900 on c301-001) are the image on `/work`:
  from `/scratch`, c301-001 mounted it (3548971). `$WORK/containers/julia_1.12_ls6.sif` is now a symlink to the
  `/scratch` copy; the original is `julia_1.12_ls6.work.sif` beside it. Undo with `mv`. `/scratch` purges files
  unread for 10 days.
- 2-step unrolled gradient compile on 0.2.303: 81.7 min gradient-only (3548971) against 53.2 min on 0.2.290
  after a primal compile (3536598). Not yet a like-for-like comparison: in 3536598 the primal compile (925 s) came
  first and may have done Julia/tracing work that the gradient compile then reused. The environment step took
  the same time in both (7.3 vs 7.75 min) and the model build was faster (913 vs 1246 s), so slow `/work` does
  not explain it. 3543332 (same config as 3536598, `gpu-a100`) gives the like-for-like number; it stays queued
  and now picks up the `/scratch` container through the symlink.
- 11-04, 2-step unrolled like for like (3543332 vs 3536598): Reactant 0.2.303 nearly doubles the gradient compile,
  53.2 → 99.6 min (81.7 min in the gradient-only run 3548971, so ±10 min between runs), while every other stage
  is faster (build −25 %, primal compile −11 %, first gradient call −15 %) and all numbers are bit-identical. At
  1 step the gradient compile fell 12 %. So the regression scales with the unrolled module, not a fixed cost.
  Which stage (tracing, the `all` rewrite pipeline, `post_op_transpose_reshape`, XLA) is not known yet; a 2-step
  `unroll_steps` attribution run would say, but exceeds the dev queue's 2 h.
- 11-04, attributing the 2-step regression (user chose option 1): `compile_attribution.jl` takes `unroll_steps`
  (`c408639b`). The stage split runs on both versions from the same commit; 0.2.290 from a detached worktree on
  `/scratch` with the pre-update manifest (never committed), so the two differ only in packages. Remove it with
  `git worktree remove /scratch/09603/ajseth/NumericalEarth-0290` when done.
- 11-05, the 2-step regression attributed (3555525 on 0.2.290 vs 3555512 on 0.2.303, same commit; totals match the
  test runs, 54.0 / 81.7 min). All of it is post-Enzyme pattern application (MLIR `InterpreterPass`, the
  Enzyme-JAX rewrite lists):

  | | 0.2.290 | 0.2.303 |
  |---|---|---|
  | `all`: pattern applications before Enzyme | 133 s | 110 s |
  | `all`: Enzyme (`DifferentiatePass`) | 666 s | 266 s |
  | `all`: the four pattern applications after Enzyme | 83 + 17 + 27 + 17 = 145 s | 600 + 238 + 169 + 238 = 1244 s |
  | `post_op_transpose_reshape`: six pattern applications | 17–39 s each, 152 s | 167–246 s each, 1305 s |
  | tracing | 212 s | 179 s |
  | XLA | 1692 s | 1538 s |

  Pattern work on the reverse-pass module is ~8.5× slower (297 → 2549 s); Enzyme and XLA got faster. At 1 step
  the same pipelines did not regress (`all` 732 → 616 s, `post_op` 97 → 119 s) with a module of similar size
  (439k ops), so it is specific to the unrolled 2-step reverse module, not to op count. Next: dump the
  post-Enzyme module on 0.2.303 and bisect the pattern list on it (`pass_bisect.jl`) to name the pattern(s),
  then report upstream with the module.
