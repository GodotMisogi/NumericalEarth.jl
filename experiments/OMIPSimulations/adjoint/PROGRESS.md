# Differentiable OMIP test: progress (2026-10-13, updated after the LS6 port)

Goal: a verified reverse-mode gradient of the AMOC at 26.5°N with respect to the JRA55 10-m winds, for the 2°
ocean-only OMIP model, differentiated with Reactant + Enzyme through NumericalEarth's own coupling
(`ocean_simulation` + `PrescribedAtmosphere` + `OceanOnlyModel`). This continues the ReactOcean work recorded in
`../omip_adjoint_handoff/` (branch `feat/omip-amoc-inversion`), whose 1-step gradient compiled but was NaN.

Scope chosen by the user: ocean-only with real OMIP data (tripolar 2° grid, ETOPO bathymetry, WOA initial
state, JRA55 repeat-year forcing), no sea ice.

## Status

| Stage | State |
|---|---|
| CPU forward twin (Mac and Vista) | passes, 1 step: AMOC −1.3433 Sv |
| Reactant primal on Vista GH200 (job 1068543) | **passes**: −1.3433 Sv, 3.5e-9 relative to CPU, repeat drift 0 |
| Reactant primal on LS6 A100 40GB (job 3506971) | **passes**: −1.3433 Sv, 3.47e-9 relative to CPU, repeat drift 0 |
| Reactant gradient on Vista (job 1068827) | **fails on aarch64**: compiled in 96.2 min, then the first call died after 88 min with LLVM `fixup value out of range` |
| Reactant gradient on LS6 A100 (job 3509300) | **first call runs on x86_64**: primal −1.3433 Sv (3.47e-9), gradient compiled in 122.8 min, first call ~15 min, then the harness died reading `g.u` from a plain tuple (fixed in `d47daaf4`) |
| Reactant gradient on LS6 A100, rerun (job 3510442) | **runs end to end, but the 1-step wind gradient is exactly zero**; the "GRADIENT VERIFIED" it printed was vacuous (see below); NaN in the `velocities.u` and `tracers.S` shadows |
| 2-step gradient on LS6 A100 (job 3511683) | **does not fit in device memory**: XLA's rematerialization estimate is 145.6 GiB. **Cause found (job 3516453's HLO dump)**: Enzyme's reverse step loop carries a zero-in/zero-out shadow slot for every forward intermediate as loop state, 72.9 GiB, counted twice by XLA (see below). **Unrolled 2-step test (job 3516665): primal −5.3233 Sv, then the gradient compile ran 4 h 18 min without finishing (5 h limit); 0% GPU, 31.6 GiB held, no error, no dump** |
| 2-step gradient, steps unrolled at trace time (job 3536598) | **fits and runs**: compile 53.2 min, first call 965 s, second 0.45 s, gradient finite and non-zero, repeat drift 0; **Taylor MISMATCH** (FD slope 1.50 × adjoint, first-order residual) because the check's wind perturbation also enters the reset's flux computation outside the window (see "The 2-step unrolled gradient") |
| Gradient compile time (jobs 3512868 → 3516091, 3516163) | **cut from 122 to 34 min** by excluding Reactant's `while_deadresult` pattern, found by the pass bisect (see "Compile time vs ReactOcean"); the default since commit b697ddf7; the 1-step test reproduces 3510442 (zero gradient, same NaN census) at the new time |

Timings (Nz = 20, 1 step, Δt = 7200 s):

| Step | Vista GH200 (1068543) | LS6 A100 (3506971) |
|---|---|---|
| Environment precompile (fresh depot) | — | 480 s |
| CPU model build | 117 s | 139 s |
| CPU twin, 1 step | 101 s | 141 s |
| Reactant model build | 967 s | 1120 s |
| Reset + primal compile | 377 s | 520 s |
| First primal call | 431 s | 458 s |
| Second primal call | < 0.1 s | 0.1 s |

The GPU memory in `logs/gpu.<jobid>.csv` is XLA's up-front 75% reservation, not the model's footprint.

## Branch and environments

- Branch `ajs/omip-adjoint` on the fork `git@github.com:GodotMisogi/NumericalEarth.jl.git`, tip `09a77f33`.
- Vista worktree: `$WORK/.julia/dev/NumericalEarth-adjoint` (remote `fork`). The main checkout
  `$WORK/.julia/dev/NumericalEarth.jl` is untouched. LS6 runs the same worktree through `/work`
  (`/work/09603/ajseth/vista/.julia/dev/NumericalEarth-adjoint`); its depot `/work2/09603/ajseth/ls6/.julia` is
  stacked over Vista's, with the dataset scratchspaces symlinked to Vista's.
- Containers: `$WORK/containers/julia_1.12_<vista|ls6>.sif` (plain `julia:1.12.7` images, built by
  `$WORK/containers/build_julia_<cluster>.slurm`). The older `ReactOcean/vi_julia.sif` is no longer used.
- Left uncommitted on purpose: `../omip_adjoint_handoff/` and the modified `../Manifest.toml`.

## What was built (`experiments/OMIPSimulations/adjoint/`, package `OMIPAdjoint`)

- `Project.toml`: own environment. Reactant pinned to `=0.2.290`. Oceananigans and ClimaSeaIce from
  `ss/for-omip`, SeawaterPolynomials from `ss/conversion-functions`, OMIPSimulations and NumericalEarth by path.
- `src/ocean_only_model.jl`
  - `omip_adjoint_grid`: 180×60×Nz tripolar grid, ETOPO bathymetry regridded on a CPU twin and copied to the
    device. `close_central_american_isthmus!` lands λ ∈ [276, 284], φ ∈ [7, 12], because the 2° bathymetry
    leaves Panama open.
  - `omip_adjoint_ocean`: `omip_closure` (CATKE, GM/Redi 800, biharmonic 50 days, Henyey background),
    split WENO advection with explicit vertical time discretization, split-explicit free surface (50 substeps),
    SplitRK3, WOA annual T/S converted to TEOS-10.
  - `omip_adjoint_forcing`: JRA55 repeat-year, 16 three-hourly slices, interpolated onto a regular 640×320
    latitude-longitude grid and held in memory, so the step does no file I/O and no Gaussian-latitude search.
  - `omip_adjoint_setup`: `OceanOnlyModel` with the NCAR (Large & Yeager) fixed-iteration bulk fluxes.
- `src/overturning.jl`: `OverturningStrength`, the AMOC at the row nearest 26.5°N (row 41, 26.4°N, 31 wet
  columns) and 1000 m, over the RAPID longitude span 80°W–14°W. The traced functor is pure reductions.
- `src/rolled_substeps.jl`: Reactant overload of `iterate_split_explicit!` that rolls the barotropic substeps
  into one traced loop (ported from ReactOcean; halves the graph).
- `src/gradient.jl`: noreshape compile options (`raise=true, raise_first=true, reshape_propagate=:none`),
  `WHILE_UNROLL_THRESHOLD[] = 0`, host snapshot/restore of the prognostic state and clocks, a separately compiled
  reset, the `@trace` window objective, the Enzyme `ReverseWithPrimal` gradient core, the shadow census
  (NaN/Inf/nonzero counts per shadow field) and a Taylor remainder check on the interior wind entries.
- `test_omip_adjoint.jl`: stages `cpu`, `primal`, `gradient`, driven by a JSON config in `configs/`.
- `prepare_data.jl`: login-node download of ETOPO, WOA and the JRA55 window.
- `omip_adjoint.slurm` (was `omip_adjoint_vista.slurm`): one script for Vista and LS6. Container
  `$WORK/containers/julia_1.12_$TACC_SYSTEM.sif`, `Transfer-of-ECCO-glo` (valid on both), default partition `gh`
  (pass `-p gpu-a100`/`gpu-h100` on LS6), GPU 0 only, the CUDA 12.9 `Reactant_jll` preference (loads on LS6's
  570 driver too), `nvidia-smi` sampling, MLIR dumps copied to `logs/` on failure, e-mail on every state change.

## Fixes outside `adjoint/`

Each was needed for the coupled model to lower under Reactant on the GPU.

| File | Change | Symptom it fixed |
|---|---|---|
| `ext/NumericalEarthReactantExt.jl` | `reconcile_state!` compiles with `raise=true raise_first=true` | CUDA codegen rejected the exchanger kernel |
| `src/EarthSystemModels/InterfaceComputations/InterfaceComputations.jl` | `clamp_fractional_index` uses `min(max(…))` in the grid's float type | `Base.clamp`'s dynamic promotion gave invalid IR |
| `src/Oceans/assemble_net_ocean_fluxes.jl` | index the 3-D ocean temperature at `kᴺ` inside the kernel; salt flux `ifelse` branches share one float type | `inttoptr` from a boxed Int zero mixed with a Float branch |
| `experiments/OMIPSimulations/src/omip_simulation.jl` | Henyey diffusivity uses `sin(deg2rad(y))` instead of `sind(y)` | `llvm.intr.trap` from `sind`'s domain check |

NumericalEarth's `test/test_ocean_only_model.jl` passes locally after these changes.

## Bugs found in the harness

- **Stale tendencies after reset** (fixed in `433b2d12`). The coupled `update_state!` refreshes only the
  interface fluxes. SplitRK3 stage 1 reads the tendencies from the previous ocean `update_state!`, and those are
  not in the prognostic snapshot. Before the fix the first primal gave −1.92 Sv and the repeat −1.38 Sv. The
  reset now also runs the ocean model's `update_state!`, and both calls match the CPU twin.

## The 1-step gradient (job 3510442)

Compile 122.6 min, first call 887 s, second call 1.73 s with zero drift in J and g. Output:

```
gradient 1 in 887.2 s: AMOC -1.3433 Sv, max|g| 0.000e+00 Sv per m/s, finite true
  velocities.u   NaN   40  Inf 0  nonzero 117635  of 536256  levels 1:20 (20 of them)
  tracers.S      NaN  116  Inf 0  nonzero 106081  of 536256  levels 0:21 (8 of them)
  shadow census: 2 of 14 fields hold NaN/Inf; finite non-zero fields: velocities.v, tracers.T, tracers.e, free_surface.η
```

- **The zero comes from the window structure.** `time_step!(::EarthSystemModel)` steps the ocean with the fluxes
  already in place, then calls `update_state!`, which computes the fluxes for the *next* step. The fluxes a
  1-step window uses come from `reset!`, outside the differentiated window, so J cannot see the winds. A 2-step
  window (step 2 uses fluxes computed inside the window) is the smallest one with a non-zero wind gradient.
- **The Taylor check passed vacuously.** It steps along `d = g/|g|`; with g = 0, d = 0, so Δf = 0 = h gᵀd. Fixed
  in `1535df3a`: a zero `gᵀd` is no longer reported as verified.
- **NaN in the ocean-state shadows** (dJ/d of the initial state, not the controls): 40 `u` entries over every
  level, 116 `S` entries over 8 levels including halos. They will reach the wind gradient if they sit at surface
  cells that feed the end-of-step-1 fluxes. The census now prints the `(i, j)` columns holding NaN, to tell the
  tripolar fold, the closed isthmus, and coast-adjacent cells apart.

## The 2-step gradient (job 3511683): does not fit in device memory

- Primal passed: −5.3233 Sv vs CPU twin −5.3233 Sv (4.71e-9), repeat drift 0. Primal compile 486 s, first call
  459 s, Reactant build 1124 s: the same as at 1 step.
- The gradient compile ran 4 h 14 min and hit the 5 h limit. XLA's log: "Very slow compile?" alarm, then
  `hlo_rematerialization`: "Can't reduce memory use below 29.92GiB by rematerialization; only reduced to
  144.07GiB, down from 145.59GiB". The 1-step gradient fit in 40 GB, so more wall time won't help: even an
  80 GB H100 or the 96 GB GH200 is too small. Something cached for the reverse pass grows hugely once step 2
  depends on step 1 (likely a rolled loop's per-iteration cache carrying 3-D state, unconfirmed).
- `configs/compile_attribution_nz20_2step.json` (job 3514047, `gpu-a100`, started 2026-10-17 06:09, before the
  `while_deadresult` default) runs the MLIR stage only for the 2-step ocean-only and coupled gradients and
  prints the largest tensor types in each module, to find that cache.
- **2-step ocean-only gradient (07:30): MLIR 3370 s, module 429,595 ops, the same size as at 1 step** (the
  time-step loop is rolled, so a second step only adds a leading dimension of 2 to the loop-carried caches).
  Largest tensors: 20 MiB `tensor<2x36x16x4x16x12x3xi64>` (6), 13.5 MiB, 12.9 MiB `tensor<16x1x326x646xf32>`
  (18, atmosphere slices), then 8 MiB `tensor<2x37x76x196xf64>` (21) and `tensor<2x36x76x196xf64>` (41):
  nothing near 145 GiB in the ocean-only variant.
- **2-step coupled gradient (09:16): MLIR 6320 s, module 438,775 ops, 14 while loops**, the same tensor
  types with the same sizes; the only difference from ocean-only is the 12.9 MiB atmosphere tensor
  `tensor<16x1x326x646xf32>` appearing 325 times instead of 18 (the interface fluxes index the slices).
  The static census cannot see the 145 GiB: it is XLA's live-buffer estimate after scheduling, and every
  loop-carried cache (the `tensor<2x...>` types) sums to about 1.5 GiB.
- `configs/compile_attribution_nz20_2step_xla.json` (job 3516453, `gpu-a100-dev`, commit 7109a76b; new config
  key `xla_debug_options` = DebugOptions fields by name) ran the XLA stage of the coupled 2-step gradient with
  the `rematerialization` pass disabled and `xla_dump_to` on. MLIR 961 s (6320 s before the `while_deadresult`
  exclusion). XLA dumped `before_optimizations` (10:24) and `before_config_assignment` (10:42), finished
  autotuning at 10:51, and was still compiling at the 2 h limit (11:42): rematerialization is not the only
  slow stage; buffer assignment was never reached. The HLO dumps are in `logs/xla_dump_nz20_2step/`.

**Found (HLO census of the dump, `hlo_census.py` / `while_state.py` / `while_sample.py` in the session
scratchpad): the 145.6 GiB is the loop state of Enzyme's reverse time-step loop, counted twice.**

| while (before optimizations) | role | trip count | state | body |
|---|---|---|---|---|
| `while.26` ⊃ `while.7` | forward: checkpoint loop ⊃ step loop | 1 ⊃ 2 | 0.74 GiB ⊃ 0.65 GiB (1287 / 1214 elements) | 69k instr |
| `while.27` | reverse: checkpoint loop | 1 | **72.6 GiB** (64,148 elements) | 279 instr |
| `while.23` in `while.27` | recompute the 2 steps from the checkpoint | 2 | 1.37 GiB (1324 elements; caches with leading dim 2) | 69k instr |
| `while.24` in `while.27` | reverse step loop | 2 | **72.9 GiB** (64,134 elements) | 294k instr |

`while.24`'s state: 34,369 × `f64[20,60,180]` (55.3 GiB), 4,749 × `f64[20,62,182]` (8.0 GiB), 1,832 ×
`f64[19,60,180]`, 11,960 × `f64[60,180]`, ... Of its 64,134 elements, 987 pass through unchanged, about 450
are real accumulators (`add`/`subtract`), 18 constants, and **about 62,700 are yielded as `broadcast(0)` and
initialized with `broadcast(0)`**: each is the shadow of one forward intermediate of the step, read once
(`add(%arg#k, slice)`), consumed, and reset to zero for the next iteration. These are Enzyme-MLIR's
body-local gradients promoted to loop-carried values (the enzyme-ops remover turns `enzyme.set`/`get`
inside a loop into iteration arguments); their value at every iteration boundary is zero, so they could be
locals, which is what they are in straight-line code. XLA's `main` holds the tuple twice (while operand +
result): 145.8 GiB, the 145.6 GiB of job 3511683. At 1 step the loop has trip count 1 and XLA's
while-loop simplifier inlines it before buffer assignment, which is why the 1-step gradient fit in 40 GB.
Neither Enzyme's `while_simplify`/`while_deadresult` nor XLA's `WhileLoopSimplifier` removes a
zero-in/zero-out state element. After XLA's optimizations (`before_config_assignment`) the trip-count-1
loops are gone and `while.85` (= `while.24`) still carries 72.9 GiB.

Ways out:
1. **Unroll the time-step loop before differentiation** (`while_unroll_threshold = n_steps`; Reactant's
   `enzyme_hlo_unroll` pattern runs in the pre-Enzyme pipeline): the shadows become locals that XLA
   schedules. Cost: the module grows with `n_steps` (430k ops per step), so compile time grows with it.
   `configs/twodegree_nz20_2step_unrolled.json` (job 3516665, `gpu-a100`, 5 h, commit bb7d80d2; new test
   config key `while_unroll_threshold`) runs the 2-step rung this way: primal (reference −5.3233 Sv from
   3511683), gradient, census, Taylor check. **Result (2026-10-18 20:02 to 01:02, c302-004, TIMEOUT)**:
   model build 1180 s, primal compile 494 s, primal −5.3233 Sv in 468 s, repeat drift 0; the gradient
   `@compile` started at 20:43 and was still running at the 5 h limit, 4 h 18 min later (1-step: 34 min).
   GPU 0% throughout and 31.6 GiB held (XLA's reservation), so it is compile time, not memory. The logs
   cannot place it: the test prints nothing between "primal 2" and "COMPILED", and XLA's stderr shows only
   ptxas warnings at the end of its codegen (job 3516163 logged no "JIT compilation" line for the gradient
   either). Note `WHILE_UNROLL_THRESHOLD` is a field of `EnzymeXLATransformPassesOptions`, so the unroll
   pattern sits in every pattern set, pre- and post-Enzyme. `compile_attribution.jl` now takes
   `while_unroll_threshold` from the config (`2c8ff3df`); `configs/compile_attribution_nz20_2step_unrolled.json`
   runs the coupled 2-step gradient's MLIR stage alone with `pass_timing` and `dump_stages`, as jobs
   3520928 (`gpu-a100-dev`, 2 h) and 3520929 (`gpu-a100`, 5 h twin), submitted 2026-10-19 16:35.
   **Both timed out inside the `all` pipeline**: tracing finished ~4 min after the model build (stage
   dumps `pre_raise_triton`, `post_raise_triton`, `pre_all`), then `all` ran 4 h 29 min in the twin
   without finishing (732 s at 1 step, 961 s for the rolled 2-step). The traced module
   `logs/attribution_stages_coupled_gradient_2step_unroll2/module_002_8Gmb_pre_all_pm.mlir` (75.3 MB,
   244,600 assignments) is the same size as the traced 1-step coupled module: the unroll is a pattern
   inside `all`, so the rolled and unrolled 2-step compiles start from one module. `pass_bisect.jl` now
   takes `while_unroll_threshold` and `excluded_passes` from the config (`4165f876`);
   `configs/pass_bisect_coupled_2step_unroll2_all.json` runs the 10 `all` segments on that module with
   threshold 2 and `while_deadresult` excluded, as jobs 3530023 (`gpu-a100-dev`, 2 h) and 3530024
   (`gpu-a100`, 5 h twin), submitted 2026-10-22. Modules after each segment go to
   `logs/bisect_coupled_2step_unroll2_all/`.
   **Result (3530023, 2026-10-23, TIMEOUT at 2 h)**: segments 1–5 in 7 min with the 1-step module sizes
   (66,425 → 72,734 → 72,625 ops with 4 loops into Enzyme; 523,574 ops and 18 loops out; 451,030 and 14
   after inlining), so the threshold did **not** unroll the forward time-step loop; the module Enzyme
   saw is the rolled 2-step one. Segment 6, the first post-Enzyme main-pattern application, then ran
   1 h 46 min to the cut (90 s at 1 step with `while_deadresult` excluded). The main list (481 patterns
   with threshold 2) holds `enzyme_hlo_unroll(2)` at index 257 in both the pre- and post-Enzyme
   applications; post-Enzyme it meets the reverse loops of trip count 2 with 64k iteration arguments and
   294k-op bodies. So `WHILE_UNROLL_THRESHOLD` unrolls after differentiation here, not before, and this
   way out is closed as implemented. The forward unroll has to happen at trace time: a plain Julia `for`
   in `window_objective` instead of `@trace for`, which makes both the forward and Enzyme's reverse code
   straight-line. The twin 3530024 (10-25, 5 h) reproduced segments 1–5 and then ran segment 6 for
   4 h 47 min without completing it.
   **Trace-time unroll** (`1b8969ff`): `unrolled_window_objective` steps with a plain Julia `for`,
   selected by `unroll_steps` in `build_primal`/`build_gradient` and the test config;
   `gradient_core!` takes the window function as its first argument. `twodegree_nz20_2step_unrolled.json`
   ran this way (threshold 0) as job 3536598 (`gpu-a100`, 8 h, 2026-10-30): it fits, the gradient compiles
   in 53.2 min and is finite and non-zero; the Taylor check fails for a reason in the harness (next section).
2. **Upstream (Enzyme-JAX)**: a while canonicalization that replaces an iteration argument whose init and
   yield are the same splat constant by that constant inside the body (then `while_deadresult` drops it),
   or an enzyme-ops-remover that keeps gradients first written by `+=` on a zero local. Report with the
   numbers above; the dumped module is `logs/attribution_coupled_gradient_2step.mlir` (47.9 MB).

## The 2-step unrolled gradient (job 3536598): fits, non-zero, Taylor check fails by construction

Ran 2026-10-30 11:11–13:13 CDT on c310-003 (`gpu-a100`, exit 1 from the Taylor `@test`). Logs
`logs/OMIPUnroll2T.{o,e}3536598`, results `results/omip_adjoint_twodegree_nz20_2step_unrolled.jld2` (LS6).

| Stage | Time / value |
|---|---|
| Reactant model build | 1246 s |
| Primal compile / first call / repeat | 925 s / 457 s / 0.1 s; −5.3233 Sv, drift 0 |
| Gradient compile | 3192 s (53.2 min; 34 min at 1 step) |
| Gradient first / second call | 965 s / 0.45 s; repeat ΔJ = Δg = 0 |
| max abs g | 4.973e-3 Sv per m/s, all finite |
| Device memory held | 31.6 GiB (XLA's reservation; fits in 40 GB) |

Census: 40 `velocities.u` and 116 `tracers.S` NaN as at 1 step, columns i ∈ {1, 90, 91, 180}, j ∈ {59, 60, 61}
(the seam), not reaching the wind gradient; `atmosphere.{u,v,T,q,p}` and `radiation.{sw,lw}` shadows are
now finite and non-zero.

Taylor check along d = g/|g|: f0 −5.32327890, gᵀd +1.91137477e-2, FD slope +2.87208178e-2, rel_err 0.503.
Δf/h is constant to 1e-4 over h = 1 … 1/32 and |Δf − h gᵀd| halves with h (ratio 2.00): the primal is smooth
and the adjoint is short by a fixed fraction of the directional derivative, not noisy.

**Cause.** `evaluate!()` runs `reset!` and then the differentiated window. `reset!` restores the snapshot and
runs the separately compiled `refresh_diagnostics!` (the coupled `update_state!` then the ocean's), which
computes the step-1 interface fluxes from the winds *as they are at call time*. `taylor_remainder_check`
sets the winds before `evaluate!()`, so the finite difference sees the winds through the step-1 fluxes
(reset, outside the window) and the step-2 fluxes (inside the window); the adjoint sees only the latter.
The 1-step gradient being zero is the same defect at the limit. So the gradient computed so far is the
derivative with the first step's fluxes held fixed, not the gradient wanted.

**Fix (not applied yet).** Move `refresh_diagnostics!` into the window (`window_objective` and
`unrolled_window_objective` call it before the steps) so every step's fluxes are differentiated, and let
`reset!` only restore the state. Cost: one more `update_state!` in the forward and reverse code. The 1-step
rung then has a non-zero gradient and becomes the cheap verification rung (34 min compile plus that
`update_state!`). The alternative, perturbing the winds after the reset inside the check, would verify the
window's adjoint alone.

## Compile time vs ReactOcean

ReactOcean compiles forward + adjoint in 7–8 min. Here, on an LS6 A100 at 2°, Nz = 20, 1 step (jobs 3506971 and
3510442), the first call is part of the compile (the second calls take 0.1 s and 1.73 s):

| Stage | Time |
|---|---|
| Reactant model build | 1116 s |
| primal `@compile` + first call | 524 s + 461 s ≈ 16 min |
| gradient `@compile` + first call | 7356 s + 887 s ≈ 137 min |

ReactOcean's two compile fixes are already in (`WHILE_UNROLL_THRESHOLD = 0`, rolled barotropic substeps), so
they don't explain the gap. The candidates, unmeasured: the differentiated region is the whole
`time_step!(::EarthSystemModel)` (JRA55 time interpolation, exchange grid, iterative similarity-theory flux
solve, CATKE) where ReactOcean steps a bare ocean with its own bulk kernel and a frozen atmosphere; and the
thunk unpacks the whole model and its shadow (first-call cost, and the Vista fixup error).

`compile_attribution.jl` (LS6 job 3512868, config `configs/compile_attribution_nz20_1step.json`) splits each
compile into the MLIR stage (tracing, Enzyme, passes) and XLA code generation, for three windows, primal and
gradient: `fluxes` (one coupled `update_state!`, objective Σ of squared net ocean fluxes), `ocean` (ocean-only
`time_step!`, AMOC) and `coupled` (the production window). It also times the model build by stage and prints an
op histogram of each optimized module (modules kept in `logs/attribution_<variant>_<mode>.mlir`). Both new
windows ran on a local CPU model (flux objective 2.09e-4; ocean-only AMOC −1.3433 Sv, equal to the coupled
1-step value as expected).

### Attribution results (job 3512868, 1 step, A100)

Model build (1120 s) is ~92% Julia compile: grid 62 s (42 s compile), ocean 375 s (352 s), forcing 55 s (14 s),
interfaces + `OceanOnlyModel` 629 s (617 s).

| Window | MLIR stage | of which Julia compile | XLA | Module |
|---|---|---|---|---|
| fluxes primal | 5.3 s | 2.4 s | 1.8 s | 1389 ops, no loops |
| fluxes gradient | 42.8 s | 38.6 s | 8.5 s | 4455 ops, no loops |
| ocean primal | 286.5 s | 133.5 s | 157.2 s | 70.9k ops, 4 loops |
| ocean gradient | **3093.3 s** | 1.7 s | 1060.9 s | 430k ops, 14 loops |
| coupled primal | 234.8 s | 37.7 s | 174.8 s | 72.5k ops, 4 loops |
| coupled gradient | **6271.8 s** | 1.7 s | 1035.9 s | 439k ops, 14 loops |

- The flux machinery is cheap; the ocean step's reverse pass is the cost.
- The ocean gradient's compile is dominated by Reactant's MLIR pass pipeline (51.6 min), not by Julia and not
  by XLA (17.7 min). This differs from ReactOcean's earlier diagnosis, where XLA code generation dominated.
- The coupled gradient's MLIR stage takes twice the ocean-only one (104.5 vs 51.6 min) for a module only 2%
  larger, with the same XLA time. MLIR + XLA = 121.8 min, matching the 122.6 min `@compile` of job 3510442.
  Either the 1-step window's trailing `update_state!` (dead with respect to the AMOC) inflates the
  intermediate modules before the optimizer drops it, or a pass scales badly with module size.
- No large stacked cache in the 1-step modules: the largest tensors are 10–13 MB.
- Reactant runs almost all of the optimization as one pipeline keyed `all`, so the per-pipeline timer says
  little. Two finer views, both config switches of `compile_attribution.jl`:
  - `pass_timing`: MLIR's per-pass execution time report after every pipeline, on stderr (`OMIPPasses.e<id>`,
    one report per pipeline, interleaved with the `[HH:MM:SS]` markers);
  - `dump_stages`: Reactant's own dump of the module before/after every pipeline, kept under
    `logs/attribution_stages_<variant>_<mode>_<n>step/`, with a table of ops / while loops / MB per stage, to
    see which pass grows the module (and whether the coupled window starts out much bigger than the ocean one).
- Queued with both switches: `configs/compile_attribution_nz20_1step_passes.json` (ocean and coupled 1-step
  gradients, no XLA) as jobs 3515041 (`gpu-a100-dev`, 2 h: the ocean variant fits, the coupled one will be cut)
  and 3515042 (`gpu-a100`, 5 h); and the 2-step config as job 3514047 (8 h, estimated start 2026-10-17 10:38).

### Per-pass results (job 3515041, 1-step ocean gradient, MLIR stage 3338 s with dumps on)

| Where | Time | Note |
|---|---|---|
| tracing (outside pipelines) | 211 s | |
| pre-Enzyme optimization + kernel raising | ~110 s | affine simplification 21 s, AffineCFG 11 s, patterns 25 s |
| Enzyme `DifferentiatePass` | 311 s | |
| post-Enzyme pattern rewriting (`transform-interpreter`) | **~1940 s** | first application 1578 s, then 153, 104, 103 s |
| `post_op_transpose_reshape` pipeline | 624 s | three down/up/down rounds of the same pattern sets, 190 s each, removing 360 of 429,942 ops |
| the other pipelines | ~70 s | ~10 s each, mostly the verifier |

The greedy pattern rewriter on the differentiated module is 75% of the MLIR stage; Enzyme itself is 9%.
Module sizes: 166k ops as traced (kernels as LLVM dialect, 4 loops) → 430k StableHLO ops after `all` (14 loops);
nothing after that changes it. Dumps: `logs/attribution_stages_ocean_gradient_1step/` (1.1 GB).

### Pass bisect on the dumped module (`pass_bisect.jl`)

Parses a stage-dump module (the traced 1-step ocean gradient, 61 MB, with its `enzyme.autodiff` op) and
runs Reactant's `all` pipeline on it in named segments, with time, MLIR per-pass report and op count after
each, keeping the module after every segment under `logs/bisect_<tag>/`. No model build, so a run starts
in minutes. Segments take placeholders for Reactant's building blocks (`{opt_passes}`, `{enzyme_pass}`,
`{main_patterns}`, ...), a `patterns: [i, j]` slice of the 482-pattern main list, or `patterns_split: n`.
- `configs/pass_bisect_ocean_1step_all.json`: the `all` pipeline in 10 segments. Ran as job 3515386
  (`gpu-a100-dev`, started 22:13 on 2026-10-16; its `gpu-a100` twin 3515268 was cancelled). Submission:
  `SCRIPT=adjoint/pass_bisect.jl CONFIG=<this> sbatch -J OMIPBisect -p gpu-a100-dev -t 02:00:00 adjoint/omip_adjoint.slurm`.
  Completed in 44 min; parse of the 58 MB module 2.5 s; the final module (429,942 ops) is identical in size
  to the attribution run's, so the standalone pipeline reproduces the in-compile one.

  | Segment | Time | Ops after | Note |
  |---|---|---|---|
  | traced | | 166,266 | 4 while loops, kernels as LLVM dialect |
  | `memory_effects_opt` | 33 s | 64,241 | |
  | `kernel_raise` | 71 s | 71,048 | affine simplification 21 s, AffineCFG 11 s, patterns 20 s |
  | `batch_opt2` | 3 s | 70,944 | |
  | `enzyme` | 282 s | 512,567 | 18 while loops: Enzyme's output is 7× the primal |
  | `post_enzyme_inline` | 8 s | 441,697 | 14 loops from here on |
  | `post_enzyme_main_patterns` | **1570 s** | 427,883 | removes 3% of the ops |
  | `post_enzyme_lower_patterns` | 154 s | 430,078 | adds ops |
  | `enzyme_cleanup` | 4 s | 430,078 | |
  | `opt2_again` | 215 s | 429,957 | two applications of ~103 s each, changing 121 ops |
  | `lower_jit` | 12 s | 429,942 | |

  So one greedy pass over the 430k-op differentiated module costs ~100 s even at a fixed point (the two
  `opt2_again` applications, and the 6 applications of `post_op_transpose_reshape`), and the first
  application after Enzyme costs 15 of those because it iterates to convergence. The 5 fixed-point-ish
  applications after the first one (154 + 215 + 624 s) change under 0.1% of the ops for 16 min.
- `configs/pass_bisect_ocean_1step_patterns.json`: the 1578 s post-Enzyme rewrite split into 8 pattern
  chunks, starting from the module after segment 5 of the run above. Ran as job 3515623 (`gpu-a100-dev`,
  2026-10-17 01:15; twin 3515590 cancelled). Chunk 3 (patterns 121:181) took **246 s**; the other seven took
  4 to 14 s each. The chunks together take 290 s against 1570 s for the whole list, so the convergence loop
  of the full application is driven by chunk 3's patterns interacting with the rest. Chunk 3 holds the
  `while_*` patterns (`while_simplify`, `while_deadresult`, `while_dus`, `while_induction_reduction`, ...),
  the `dus_*` ones (`dus_dus`, `dus_concat`, `sink_dus`, `hoist_slice`), `slice_concat`/`concat_slice`,
  `common_compare_expression_rewrite` and `associative_common_mul_op_reordering`.
  The lowering pattern set that followed the eight chunks then took **4707 s** (78 min) to reach 430,064 ops,
  against 154 s after the converged main application and essentially the same op count (430,078). So the
  cost is not one pass: the first application of a full pattern set on the differentiated module iterates
  through a long chain of op-count-neutral rewrites, and applying the main set in chunks first made that chain
  three times longer. The work is in the interaction between pattern groups, which the chunk-3 control run
  (whole main list minus chunk 3) tests directly.
- `configs/pass_bisect_ocean_1step_chunk3.json`: chunk 3 split in 8 (on the module after chunk 2) and the
  control run of the whole main list without chunk 3 (on the module after `post_enzyme_inline`, where the
  whole list took 1570 s). Uses the new per-segment `module` and `patterns_except` keys. Ran as job 3515916
  (`gpu-a100-dev`, 2026-10-17 02:52; twin 3515917 cancelled). Sub-chunk **159:166 took 147 s**; the other
  seven 3 to 4 s. The control, **the whole main list without chunk 3, took 41 s instead of 1570 s**, ending at
  436,681 ops against 427,883 with it (2% more ops). Patterns 159:166: `abs_positive_simplify`,
  `transpose_elementwise_transpose`, `select_comp_iota_const_simplify`, `sign_abs_simplify`,
  `broadcastindim_is_reshape`, `reduce_window_wrap`, `slice_reduce_window`, `while_deadresult`.
- `configs/pass_bisect_ocean_1step_p159_166.json`: those eight, one per segment, on the module after
  sub-chunk 151:158; then the control without 159:166 on the post-inline module, followed by the lowering
  set, to see whether the convergence chain moves into the next application (as it did after the chunked
  run). Ran as job 3516058 (`gpu-a100-dev`, 2026-10-17 03:09; twin 3516059 cancelled).
  **`while_deadresult` alone: 147 s** (removing 137 ops); each of the other seven 3 s and no change. The
  control, **the whole main list without 159:166: 55 s to 428,131 ops**, against 1570 s to 427,883 ops with
  them. So one pattern costs 25 min of the 26 to remove about 250 ops of 430k. The lowering set after that
  control took 254 s (154 s after the converged run), ending at 430,074 ops: it still contains
  `while_deadresult`, which did part of its work there. Reactant's
  `CompileOptions(excluded_passes=["while_deadresult"])` drops a pattern by name from every generated list
  (checked locally: 2 occurrences → 0 in both the sroa and non-sroa pipelines).

### Testing the fix: `excluded_passes` (commit 76c91036)

`compile_attribution.jl` and `test_omip_adjoint.jl` take `excluded_passes` from the config and pass it to
`adjoint_compile_options`. Two configs, submitted 2026-10-17 03:20:
- `configs/compile_attribution_nz20_1step_no_deadresult.json`: ocean and coupled 1-step gradients, MLIR +
  XLA, per-pass timing, against job 3512868 (ocean 3093 + 1061 s, coupled 6272 + 1036 s). Jobs 3516090
  (`gpu-a100`, 5 h) and 3516091 (`gpu-a100-dev`, 2 h: the ocean variant fits); cancel the twin.
- `configs/twodegree_nz20_1step_no_deadresult.json`: the first rung again, to check gᵀd against job 3510442
  and "GRADIENT VERIFIED" with the faster compile. Job 3516092 (`gpu-a100`, 5 h).

Result from 3516091 (dev twin, 2026-10-17 04:14): **ocean gradient MLIR stage 883 s, from 3093 s.** The
post-Enzyme pattern applications went 1578/153/104/103 s → 90/11/10/8 s and `post_op_transpose_reshape`
624 s → 88 s: `while_deadresult` was also the ~100 s in every fixed-point application. What remains is
Enzyme's `DifferentiatePass` 310 s, tracing ~210 s and ~180 s of Julia compile. The module is 429,840 ops
(258 more than with the pattern, 0.06%) and XLA took 1092 s (1061 s before): the pattern buys nothing here.
Ocean gradient compile total: 883 + 1092 s = 33 min, from 69 min. **Coupled gradient MLIR stage 909 s, from
6272 s** (04:47): the coupled/ocean 2× gap was `while_deadresult` scaling worse on the coupled module, not a
dead trailing `update_state!`; without it the two cost the same. Coupled XLA 1124 s (1036 s before), module
438,915 ops. Job 3516091 completed in 88 min; twin 3516090 cancelled.

| 1-step gradient, A100 | MLIR before | MLIR now | XLA before | XLA now | compile total |
|---|---|---|---|---|---|
| ocean-only | 3093 s | 883 s | 1061 s | 1092 s | 69 → 33 min |
| coupled (production) | 6272 s | 909 s | 1036 s | 1124 s | 122 → 34 min |

Inside the 909 s coupled MLIR stage: `all` 732 s (Enzyme ~310 s of it), `post_op_transpose_reshape` 97 s,
tracing 69 s. The XLA stage, 19 min, is now the larger half and was never attributed (`--xla_dump_to`).
The gradient test with the exclusion, job 3516163 (`gpu-a100-dev`, 05:22 to 06:55; twin 3516092 cancelled),
reproduces job 3510442 exactly: primal −1.3433 Sv (3.47e-9 vs CPU), **gradient compiled in 33.8 min
(122.8 min before)**, first call 903 s, gradient exactly zero with the same shadow census (`velocities.u`
40 NaN, `tracers.S` 116 NaN). The Taylor check now reports MISMATCH for the zero gradient and the test
fails there, as the fix for the vacuous pass intends. The exclusion is the default in
`adjoint_compile_options` from this point; a config's `excluded_passes` replaces it (`[]` runs every pattern).
Shared helpers (pass timer, census, stage table) live in `mlir_diagnostics.jl`.

## Known issues left open

- **Gradient first call on aarch64 (Vista).** Job 1068827 passed the primal again with the new container
  (−1.3433 Sv, 3.47e-9), compiled the 1-step gradient in 96.2 min, then spent 88 min in the first call at 0% GPU
  and exited with `error: <unknown>:0: fixup value out of range` (no Julia stack trace). The first call of a
  compiled thunk is Julia JIT of the thunk glue (the primal's took ~430 s, also at 0% GPU); the gradient thunk
  unpacks the whole `OceanOnlyModel` and its shadow. ReactOcean hit the same error on aarch64 when the atmosphere
  sat in the forcing parameters (`../omip_adjoint_handoff/bulk_forcing_handoff.md`). AArch64 branch and literal
  fixups reach ±1 MB, x86_64 displacements ±2 GB, so the rung moved to LS6, where the same first call finishes
  in ~15 min (job 3509300). Shrinking the thunk (fewer
  untraced leaves in the arguments, or passing only the controls and state) is the lever for Vista, and it would
  also cut the first-call cost on both clusters.

- `atlantic_ocean_basin` does not cut this tripolar grid. The grid's longitudes run 73°–433°, so the Cape
  Agulhas barrier at 20° never lands. The objective sidesteps it by defining the section by longitude span. The
  basin mask should be fixed in NumericalEarth.

## Next steps

1. Differentiate the flux computation: `refresh_diagnostics!` into the window, `reset!` restores state only.
   Rerun the 1-step rung on the dev queue (expected ≈ 19 min build + 34 min compile + the `update_state!`
   share + 15 min first call) for the first meaningful Taylor check, then the 2-step unrolled rung.
2. Report `while_deadresult`'s cost upstream (Enzyme-JAX: 25 min on a 14-loop, 430k-op differentiated
   module for 250 ops). Compile time is now 34 min for the 1-step coupled gradient: XLA 19 min (never
   attributed; `--xla_dump_to`), Enzyme 5 min, tracing + Julia compile 5 min, model build 19 min (Julia
   compile of ComponentInterfaces + OceanOnlyModel; a sysimage or PrecompileTools workload would remove it).
3. Rolled time-step loops need the Enzyme-JAX fix for the zero-in/zero-out shadow state (see "The 2-step
   gradient"); until then every rung runs with the steps unrolled at trace time (`unroll_steps`), whose
   compile time grows with the step count (34 min at 1 step, 53 at 2). Report upstream together with
   `while_deadresult`'s cost.
4. Then Nz = 70, then 8 steps, recording compile and evaluation times at each rung. Pass criteria: finite
   shadow census and "GRADIENT VERIFIED" from the Taylor check.

## Reactant 0.2.303 (jobs 3543579, 3543330/1, 3544007/8)

Updated the adjoint environment to Reactant 0.2.303 / Reactant_jll 0.0.436+0 / Enzyme 0.13.216 / EnzymeCore 0.8.26 /
Enzyme_jll 0.0.303+1 (`017f6230`), Oceananigans unchanged at 0.113.5. On LS6 the registry refresh must name the
depot holding General (Julia 1.12's `Pkg.Registry.update` defaults to the first depot, which has none).

The first runs failed to compile the gradient: `could not compute the adjoint for this operation
"enzymexla.jit_call"`, 15 times, all `gpu__compute_y_bcs` for the (Face, Center, Center) tendency with a
`Flux{Nothing}` south and `Zipper` north boundary. That kernel writes nothing; the new Enzyme-JAX raises it to an
operand-free side-effect-free `jit_call` which the forward pipeline deletes and the Enzyme pass cannot
differentiate (EnzymeAD/Reactant.jl#3636). `cd417288` adds `compute_y_bcs!(…, ::NotFluxBC, ::ZBC, …) = nothing`
to OMIPAdjoint, the same change as CliMA/Oceananigans.jl#6396, to be removed when the pin includes it.

| 1 step, Nz = 20 | 0.2.290 | 0.2.303 |
|---|---|---|
| gradient compile (test) | 33.8 min (3516163) | 29.9 min (3544007) |
| first gradient call | 903 s | 778.1 s |
| primal compile / first call | ≈ 8–9 min / – | 466.1 s / 429.6 s |
| ocean MLIR + XLA | 883 + 1092 s | 724.5 + 918.0 s |
| coupled MLIR + XLA | 909 + 1124 s | 806.9 + 959.6 s |
| inside coupled MLIR: `all` / `post_op_transpose_reshape` / tracing | 732 / 97 / 69 s | 615.7 / 119.0 / 59.6 s |
| coupled module | 438,915 ops | 438,905 ops |

Primal (−1.3433 Sv, 1.58e-7 from the CPU twin), repeat drift 0, zero gradient and the NaN census (40 `u`, 116 `S`
on the seam columns) are identical to 0.2.290.

### The 2-step compile regression on 0.2.303 (jobs 3543332, 3555512, 3555525)

Like for like, the 2-step `unroll_steps` gradient compiles in 99.6 min on 0.2.303 (3543332) against 53.2 min on
0.2.290 (3536598); results are bit-identical. Attribution on both versions from the same commit (0.2.290 from a
worktree with the old manifest), coupled gradient, `compile_attribution_nz20_2step_unroll_steps.json`:

| | 0.2.290 (3555525) | 0.2.303 (3555512) |
|---|---|---|
| MLIR stage | 1550 s | 3362 s |
| tracing | 212 s | 179 s |
| pattern applications before Enzyme | 133 s | 110 s |
| Enzyme `DifferentiatePass` | 666 s | 266 s |
| pattern applications after Enzyme, in `all` | 145 s | 1244 s |
| `post_op_transpose_reshape` (six pattern applications) | 180 s | 1331 s |
| XLA | 1692 s | 1538 s |
| total | 54.0 min | 81.7 min |
| module | 460,379 ops | 464,435 ops |

Every pattern application after Enzyme is 6–14× slower; everything else is the same or faster. The 1-step
module (439k ops) shows no such regression, so a pattern in the newer Enzyme-JAX scales badly on the unrolled
2-step reverse pass specifically. To name it: dump the post-Enzyme module on 0.2.303 and bisect the pattern list.

**Named: `cse_reduce_transposed`** (new in 0.2.303; Enzyme-JAX `CSEReduceTransposed`, `EnzymeHLOOpt.cpp`).
Stage dumps on both versions (3557933/4) reproduce the MLIR stage (1636 / 3488 s). `pass_bisect.jl` (now building
0.2.303's pipelines too) on LS6, module × pattern code, `post_op_transpose_reshape`:

| | 0.2.290 module | 0.2.303 module |
|---|---|---|
| 0.2.290 patterns (3558732) | 150.6 s | 154.1 s |
| 0.2.303 patterns (3558733) | 1503.4 s | 1188.9 s |

so the pattern code, not the module Enzyme produces. After Enzyme (Bisect303): `opt_passes2` 821.4 s (107.6 on
0.2.290), again 401.5 s (52.0). The rest ran locally on CPU (Mac, ~3.5× faster than an LS6 node; same
Reactant_jll builds), one main-list application on the 0.2.290 post-`all` module: 0.2.290 7.9 s, 0.2.303 161.8 s;
without the 44 patterns new in 0.2.303 15.0 s; `max_reused_const_prop_size = 0` 119.2 s (not it); list halves
then the new patterns in four groups then one at a time: only excluding `cse_reduce_transposed` restores it
(24.1 s against 164–166 s for the other five of its group), with the **same** output (461,900 ops), i.e. the
pattern never fires here. Confirmations without it: whole `post_op_transpose_reshape` on the 0.2.303 module 68.6 s
(vs 574.9 s with it, 49.7 s on 0.2.290), output 464,435 ops as in the LS6 compile; post-Enzyme `opt_passes2`
43.6 s (313.4 s with it), output identical. Cause, from the source: for every `stablehlo.reduce` the pattern
collects as candidates all reduces in the block sharing its init value (one splat zero for the whole module) and
walks the input's elementwise computation and each candidate's, so its cost is quadratic in reduces times the
size of their computations, paid on every application and every driver iteration even when nothing matches.
The unrolled 2-step reverse pass has ~520–580 reduces over large fused stencil computations.

