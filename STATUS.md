# OMIP adjoint: status as of 2026-11-10

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

Queued on LS6 (11-07; results 11-08):

- **Grad1Section 3566763 VERIFIED** (11-08, dev, `1da8855f`, 1:25:52; resubmitted from `gpu-a100` 3561032,
  `twodegree_nz20_1step.json`): CPU twin J −14.3509 Sv, traced primal vs CPU twin 7.5e-13, rolled 1-step
  gradient compile 31.9 min, first call 847 s, repeat 0.47 s exact; gradient vs primal J 3.7e-16; Taylor
  rel_err 6.1e-5, second-order remainder. File `results/omip_adjoint_twodegree_nz20_1step.jld2`.
- **Grad1Frozen 3564718 done** (11-08, 57.7 min, 1 step unrolled, `frozen_mixing`, `b0418df6`, staircase
  section): build 916 s, compile 22.7 min (22.2 unfrozen, 3560476), first call 768 s, second 0.37 s, J −14.3509 Sv,
  max|g| 1.09e-2 Sv per m/s, repeat drift 0. Against the full model's finite difference: gᵀd 4.49911511e-2,
  FD slope 4.49959212e-2, rel_err 1.06e-4, **GRADIENT VERIFIED**, remainder second order (rate 4.0 → 3.7 until
  roundoff below h = 1/16). Freezing CATKE's κ costs 1e-4 at 1 step and saves no compile time here. Against
  the full-mixing gradient 3566763 (Float64 forcing; J differs 2.9e-7 from the forcing precision alone):
  cosine 0.99999968, norm ratio 0.99984, max|Δg| / max|g| 1.2e-3; the forcing precision accounts for ≲ 3e-5
  (from the 2-step pair), so this is the frozen-mixing error.
- **Grad2Ckpt 3565170 failed its J check** (11-08, 1:36:45, Revolve, 1 checkpoint): build 916 s, primal
  compile 484 s, primal J −28.563087893063496 (repeat exact), **gradient compile 27.1 min** (47.8 for the
  unrolled 2-step window), first call 2230 s, max|g| 1.11e-2, finite. Its J differs from the primal's by
  7.5e-8 (relative), the fused unrolled gradient (3561033) by 3.6e-12; my `@test rel < 1e-8` aborted the job
  before the census, repeat and Taylor check, so no gradient was saved. The check now warns and records
  instead (`211728c9`). The old file is `..._2step_checkpointed_3565170.jld2`. Rerun 3565514 failed at model
  build on `afcb086f` (below).
- **Grad2Ckpt 3565554 VERIFIED** (11-08, `1da8855f`, 1:39:02): primal J −28.563087573790 (repeat exact),
  gradient compile 27.1 min, first call 2368 s (the five Revolve thunks each pay a first-call cost), repeat
  0.82 s with max|Δg| 6e-36; gradient vs primal J 1.6e-15; Taylor rel_err 2.0e-7, second-order remainder down
  to h = 0.125. Against the unrolled 2-step gradient 3561033 (Float32 forcing): J 1.1e-8, cosine 0.9999999996,
  norm ratio 0.999996, max|Δg| / max|g| 2.9e-5, i.e. the forcing precision. NaN shadows on the same seam
  columns as before (82 u, 160 v, 8 S). File `results/omip_adjoint_twodegree_nz20_2step_checkpointed.jld2`.
- **Host clocks baked into compiled functions** (StepConsist 3565513, `step_consistency.jl`, log kept as
  `StepConsist.o3565513.before_clock_fix`): under ReactantState the `EarthSystemModel`, `PrescribedAtmosphere`
  and `PrescribedRadiation` clocks were host `Clock{Float64}` (NumericalEarthReactantExt's `same_time_type`
  accepts them beside the ocean's `ConcreteRNumber` clock). The atmospheric and radiative interpolation reads
  `coupled_model.clock.time`, so each compiled function used the forcing time it was traced at, and tracing
  ticked the host clocks (0 → 14400 s after compiling the 1- and 2-step windows). Separately compiled step vs
  window: 5.2e-3 at 1 step, 6.5e-4 at 2; atmosphere state, CATKE and fluxes differ O(1). A rolled window gives
  every iteration the forcing of its trace time. Earlier verified results stand: 1-step windows and the
  unrolled 2-step window were traced from a fresh clock (unrolling advances the host clock step by step), and
  the rolled 2-step primal's stale value only enters the fluxes after the last step, which J does not see.
  Fixed in OMIPAdjoint (`afcb086f`, `Clock(grid)` for all three). With the time traced, Oceananigans'
  `ifelse(n₁ == n₂, ψ₁, ψ̃)` returned `Union{Float32, Float64}` for the Float32 JRA55 slices and Reactant could
  not raise the atmospheric interpolation kernel (3565514, `reconcile_state!`); the in-memory forcing is now
  Float64 (`1da8855f`; CPU 2-step J −28.5630876 vs −28.5630879 with Float32). **Verified** by StepConsist
  3565553 (`1da8855f`; 3565543 cancelled, it had started on `afcb086f`): no host number changes during any
  compile; separately compiled refresh + step vs window, J relative difference 2.2e-15 at 1 step and 1.1e-15
  at 2 (window J −28.563087573790813, matching the CPU twin); every device array agrees to ≤ 6e-11 relative
  (CATKE κ, the implicit solver's scratch). Upstream TODO: convert ψ₁ in Oceananigans'
  `interpolate`/`time_interpolated_getindex`.
  Upstream: NumericalEarth's `default_earth_system_clock` and the prescribed components' default
  `Clock{Float64}` should be device clocks on ReactantState.
- Longer rollouts on `gpu-a100-dev` (11-09, `3c70bde4`; 3565171 cancelled after a day in `gpu-a100` Priority):
  Grad8Ckpt 3570013 (8 steps, 3 checkpoints), Grad84Ckpt 3570014 (1 week, 12 checkpoints, 60 slices) and
  Grad360Ckpt 3570015 (30 days, 30 checkpoints, 248 slices, gradient stage only to fit 2 h). All three ran
  end to end and saved their gradients (8: 52 MB local; 84: 197 MB and 360: 813 MB on LS6 only); all three
  failed the old Taylor verdict.

  | steps | build | gradient compile | first call | repeat | J (Sv) | grad vs primal J | small-h Taylor |
  |---|---|---|---|---|---|---|---|
  | 8 | 931 s | 28.4 min | 2350 s | 3.5 s | −11.28 | 3.6e-14 | second order; Richardson 2.6e-6 |
  | 84 | 1022 s | 27.3 min | 2241 s | 28 s | −15.97 | 2.1e-11 | noise ±2e-4 in Δf; agrees to ~3e-3 |
  | 360 | 1423 s | 29.1 min | 2367 s | 104 s | +16.63 | (no primal) | Δf ≈ 8e-4 + h gᵀd; 1e-2 |

  Compile and first call are flat in the step count; a full 30-day gradient costs 104 s once compiled.
  The first call's ~39 min is host-side: XLA logs nothing between "COMPILED" and the end of the first call
  (it does log every compile), so it is most likely Julia compiling the five thunks' call wrappers for the model
  type (ECCO4's smaller model: 753 s). The old verdict's six-point fit was dominated by h = 1 (8 steps: rel_err
  1.08e-3 though the remainder is O(h²) down to 6.5e-7); it now fits the three smallest h (`c1dcb8b1`;
  earlier runs: 2-step 1.9e-6, 1-step 2.7e-6, ECCO4 8.1e-6). For 84 and 360 steps the finite difference is
  not smooth: the remainder stops falling with h (84) or Δf keeps a ~8e-4 Sv jump as h → 0 (360), as if
  discrete switches (CATKE's convective/stable branches, `ifelse` limiters) flip under any perturbation.
  The adjoint differentiates the branch taken, so FD with h ≥ 1/32 m/s cannot check it there.
  FD scans (11-10): the first submission (3571246–8) failed to precompile OMIPSimulations, because the 11-09
  merge of upstream `ss/omip-prototype` (`ba7f0eeb`) brought `a44003cf`, which includes
  `src/extended_halo_checkpoint_patch.jl` that upstream committed only in `c6eede8f` (with a fold-halo patch
  and a `restore_prognostic_state!` override). At the user's choice the merge is reverted (`9f9c1f90`; the tree
  outside `adjoint/` and STATUS equals `6c0d6357`, the code of the saved gradients); merge upstream again after
  the scans and re-verify. Resubmitted as FDScan8 3571322, FDScan84 3571323, FDScan360 3571324 on dev, stages
  `primal,fd_scan`: forward and central differences of the compiled primal along each saved gradient for
  h = 1 … 6e-8 m/s (25 octaves, ~1–6 s per evaluation), 8 steps as the smooth control. The user expects the
  adjoint will in practice have to freeze the turbulence (`frozen_mixing`).
  **Results (11-10, all three completed; f0 equals the saved J to 4e-14, 2e-11, 4e-12):** the adjoint is the
  exact local derivative at every length; the objective is piecewise smooth with small jumps.

  | steps | gᵀd (Sv per m/s) | smooth neighborhood along ±d | FD vs gᵀd there | jumps in f | central diff. at h = 0.25–1 |
  |---|---|---|---|---|---|
  | 8 | 0.1842 | all h ≤ 1 | central 4e-9 – 2e-8 (h = 2e-3 – 2e-2), forward O(h) | none | 4e-4 |
  | 84 | 0.6766 | h ≤ 2.4e-4 | central 4e-9 (h = 1.2e-4), forward 2e-6 | 1e-6 – 3e-4 Sv for h ≥ 4.9e-4 | 5e-3 – 7e-3 |
  | 360 | 0.7346 | +d: h ≤ 1.2e-4; −d: h ≤ 4.8e-7 | forward 1e-6 – 2e-5 (h = 2e-6 – 3e-5) | 2.6e-7 Sv at h ∈ (4.8e-7, 9.5e-7) along −d; up to ~1e-3 Sv for h ≥ 6e-5 | 2e-3 (h = 0.5), 5e-4 (h = 0.25) |

  The jumps are discontinuities of f itself (constant offset over a range of h, e.g. −2.6e-7 Sv along −d at
  360 steps from h = 9.5e-7 to 3e-5), not kinks: discrete switches in the forward model (CATKE's convective
  and stable branches are the likely source). Their size grows with the window (none at 8 steps, ≤ 3e-4 Sv at
  84, ~1e-3 Sv at 360), so the old Taylor checks (same direction, h ≤ 1e-2) failed on the jumps, not on the
  gradient. At finite amplitude the jumps are negligible: over h = 0.25–1 m/s the secant slope matches gᵀd to
  0.05–2 % at 30 days. Along d = g/|g| the signal is largest; along other directions the same jumps weigh more.
  Freezing the mixing in the adjoint does not change f or its jumps; it changes what the gradient
  approximates, so its check is the large-h secant, not small-h Taylor.
  Files: `results/omip_adjoint_twodegree_nz20_{8,84,360}step_fd_scan.jld2`, logs `FDScan*.o357132{2,3,4}`.
- **Frozen mixing at 84 and 360 steps** (11-10, Frozen84 3572831, Frozen360 3572832, `8604de03`): both ran
  end to end; "FAILED" is the small-h Taylor verdict on the jumps (the Taylor rows were then not saved; fixed
  by saving before the `@test`). Reduced on a CPU node (ReduceGrad 3573710, `reduce_gradients.jl` via
  `omip_cpu.slurm`; 3573177 failed because `omip_adjoint.slurm`'s full precompile loads Reactant's CUDA
  extension, which fails without a GPU). Cosine(full, frozen) 0.99964 (84) and 0.99418 (360), norm ratio
  0.9963 and 0.9905. Against secants over h = 1/32 … 1: along d_full the full gᵀd is −0.6 % / −0.2 % and the
  frozen −1.0 % / −1.7 %; along d_frozen full −0.1 % / −1.0 %, frozen −0.4 % / −1.3 %. So the full adjoint
  predicts the finite response slightly better, and freezing saves no time (compile 29.0 / 30.1 min, first
  call 2675 / 2475 s, repeat 27 / 114 s). Uniform-wind sums differ more than the cosine suggests: at 30 days
  Σ∂J/∂v₁₀ is +0.32 (full) vs +0.76 (frozen) Sv per m/s, and the full map's Caribbean peak is absent from
  the frozen one. Next check: FD along uniform u₁₀ and v₁₀ at 30 days. Writeup with figures:
  `experiments/OMIPSimulations/adjoint/papers/omip_amoc_adjoint.tex`.
- **ECCO2Ckpt 3566764 VERIFIED** (11-08, dev, `1da8855f`, 1:04:54; resubmitted from `gpu-a100` 3565218):
  build 292 s, CPU twin J −34.9752 Sv, primal vs CPU twin 2.9e-12, 2-step checkpointed gradient wrt the stress
  compile 27.4 min, first call 753 s, repeat 0.54 s exact, gradient vs primal J 9.8e-15, max|g| 1.56 Sv per
  N m⁻² (the log's "Sv per m/s" label is JRA55's). Taylor (h0 = 0.1) rel_err 7.1e-4 from the averaged slope,
  which the h = 0.1 point dominates (remainder 6.5e-4, then 8e-6 at h = 0.05); at h = 3.1e-3 the remainder is
  2.3e-6 of h gᵀd. Same seam NaN shadows. File `results/ecco4_adjoint_nz20_2step_checkpointed.jld2`.
  Setup: `PrescribedFluxOcean`,
  ECCO4 1993-01 initial state and monthly stress, net heat and net freshwater fluxes, CPU twin, primal,
  2-step checkpointed gradient wrt the stress. Local CPU build checked (2 steps: J 0 → −35.0 Sv from rest,
  stress ≤ 0.4 N m⁻², heat flux −530…+230 W m⁻², freshwater −37…+200 mm/day). The Earthdata credentials
  (`ECCO_USERNAME`, `ECCO_WEBDAV_PASSWORD`) are now exported in LS6's `~/.bashrc` (copied from Vista);
  oceFWflx for all of 1993 is cached (job 3565274).

Checkpointing v0.13.0 was added to LS6's adjoint Manifest by a `development` job (3565146, `PRESERVE_ALL`)
before pulling, so pending jobs kept a consistent environment.

**Grad2Section 3561033** (11-07, 1:22:49, 2 unrolled steps on the staircase section, quadratic-fit verdict):
build 920 s, compile 47.8 min, first call 761 s, J −28.5631 Sv, max|g| 1.18e-2 Sv per m/s, **GRADIENT
VERIFIED** (rel_err 9.18e-7). Sensitivity sits on 26.5°N with peaks at the staircase steps (63°W, 28°W);
uniform 1 m/s: −0.257 Sv (u), +0.658 Sv (v). J of −28.6 Sv 4 h from rest is likely the spin-up transient
(row 41 gave −5.3); the CPU twins of 3566763 (1 step) and StepConsist 3565553 (2 steps) match the traced
objective, and ECCO4's initial state gives −35.0 Sv after 2 steps, so it is the spin-up from rest.

**Grad2Window 3560703** (11-07, 81 min, 2 unrolled steps, gradient only, `4cc69c3e`; 3559769's
result kept as `..._2step_unrolled_gradient_3559769.jld2`): build 923 s, compile 47.3 min, first call 734 s,
second 0.43 s, J −5.3233 Sv, max|g| 8.19e-3, same seam NaN census as at 1 step. Taylor reports MISMATCH
(rel_err 3.0e-3) but the gradient is right: |Δf − h gᵀd| falls 4.0× per halving (second order) and
(Δf − h gᵀd)/h² is constant (2.07–2.14e-4) over h = 1 … 1/32, so the check's averaged successive-difference
slope carries the curvature c(hᵢ + hᵢ₊₁), which predicts the whole 3.0e-3; a quadratic fit Δf = a + g h + q h²
gives rel 9.3e-5 (2.3e-4 at 1 step). The verdict needs that fit (or a remainder-order test).

**The objective is not at 26.5°N in the Atlantic.** On the 2° `TripolarGrid` (180 × 60) the rows already curve
at 26°N: `nearest_row` probes i = Nx ÷ 2 (111°W, Pacific) and picks row 41 (26.4°N there), which across the
RAPID span (280–346°E) runs from 28.4°N at Florida to 33.8°N. The wind sensitivity map (2-step, local plot)
follows that row: Ekman-signed ∂J/∂u < 0, ∂J/∂v > 0 along it; a uniform 1 m/s gives −0.35 Sv (u), +0.42 Sv (v);
> 99.999 % of Σ|g| in the North Atlantic. Fixed in `b0660e21`: a staircase of v- and u-faces along 26.5°N
(rows 38–40, within ±1.4° of the line); every AMOC value before it is row 41's.

**First verified gradient: Grad1Window 3560476** (11-07, 56 min, 1 unrolled step, gradient only,
`twodegree_nz20_1step_unrolled_gradient.json`, on `4cc69c3e`, which moves `refresh_diagnostics!` into the
differentiated window; `reset!` only restores the snapshot, `build_reset` removed): build 925 s, gradient
compile 22.2 min, first call 680 s, second 0.31 s, J −1.3433 Sv, max|g| 7.22e-3 Sv per m/s, finite, repeat
drift 0. Taylor: gᵀd 2.66807157e-2, FD slope 2.66993605e-2, rel_err 7.0e-4, **GRADIENT VERIFIED**; the
remainder |Δf − h gᵀd| falls 4.5× per halving of h (second order). Census: NaN in 82 `u`, 160 `v`, 8 `S`
state shadows, all on the fold seam (j = 60–61, i ∈ {0, 1, 2, 90, 91, 92, 180}), not reaching the winds.

Grad2NoCSE 3559769 (done 11-06, `fb295782`, `cse_reduce_transposed` excluded): gradient compile **47.3 min**
(99.6 with the pattern, 3543332; 53.2 on 0.2.290), build 939 s, first call 1073 s, second 0.43 s; J, gᵀd
1.91137470e-2 and FD slope 2.87207311e-2 identical to 3543332, Taylor MISMATCH as before (finding 4).
**The 0.2.303 2-step regression is `cse_reduce_transposed`** (new Enzyme-JAX pattern): LS6
bisect 3558732/3 (pattern code, not module: 0.2.303 patterns 1189–1503 s on either module, 0.2.290 151–154 s),
then local CPU bisects on the dumped modules: excluding that one pattern restores 0.2.290's times with identical
output (it never fires; quadratic candidate search over reduces sharing one init). Excluded by default
(`fb295782`), confirmed by 3559769; upstream report pending.
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

1. `refresh_diagnostics!` is in the window (`4cc69c3e`); 3560476 checks the 1-step rung. If the Taylor check
   passes, rerun the 2-step unrolled rung (gradient only fits `gpu-a100-dev`), then map the AMOC sensitivity
   to the winds.
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
