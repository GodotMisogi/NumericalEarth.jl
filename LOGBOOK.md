# OMIP adjoint logbook

Reverse-mode gradient (Reactant + Enzyme) of the AMOC at 26.5°N with respect to the JRA55 10-m winds, for the
2° ocean-only OMIP model, package `OMIPAdjoint` in `experiments/OMIPSimulations/adjoint/` (all paths below are relative to it), branch `ajs/omip-adjoint` (Oceananigans 0.113.5), since 11-12 `ajs/omip-adjoint-oc117` (0.117.2). `PROGRESS.md`
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
| 11-07 | Move `refresh_diagnostics!` into the differentiated window (`4cc69c3e`) | Finding 4; first verified gradient 3560476 |
| 11-07 | AMOC section as a staircase of faces along 26.5°N, rows 38–40 (`b0660e21`) | Row 41 curves from 28°N to 34°N across the Atlantic on the tripolar grid |
| 11-08 | Device clocks for the coupled model and the prescribed components (`afcb086f`); Float64 forcing in memory (`1da8855f`) | Host clocks were baked into the compiled functions (entry 11-07 to 11-10) |
| 11-08 | Checkpointed gradient: one compiled step and its reverse pass, run on Revolve's schedule (Checkpointing.jl) | Compile time flat in the window length; unrolled windows grow with the step count |
| 11-08 | ECCO4 prescribed-flux ocean (`PrescribedFluxOcean`) with monthly stress, net heat and net freshwater flux as controls | Pillar et al. (2016)'s controls; each monthly slice is one lead |
| 11-10 | `frozen_mixing`: CATKE's diffusivities enter the adjoint through `ignore_derivatives` | The full CATKE adjoint is NaN at 90 days; FD scans show a piecewise-smooth objective (CATKE's switches) |
| 11-11 | Objective averaged over the window's last 30 days; Pillar-style lead maps | Pillar et al. (2016) |
| 11-11 | LS6 jobs charge `ADCIRC` (user) | Transfer-of-ECCO-glo nearly spent |
| 11-12 | Merge upstream `ss/omip-prototype` (`f9d52dcb`): Oceananigans 0.117.2, branch `ajs/omip-adjoint-oc117` | `ss/for-omip` deleted upstream; same Oceananigans as ReactOcean |
| 11-14 | Every Slurm job mails ajseth@utexas.edu (user) | |
| 11-14 | Stay on Julia 1.12.7 | Reactant 0.2.303 leaves `arith.uitofp` at the XLA export under Julia 1.13.3 |
| 11-16 | Adjoint-only damping is the intended long-window remedy; forward-viscosity runs only as diagnostics (user) | Standard practice (Hoteit et al. 2005) |
| 11-17 | Test Reactant main (0.2.309, git `2bb32e28`, Reactant_jll 0.0.445) on its own branch and worktree `ajs/omip-adjoint-r309` | Leaves the jobs queued in the shared worktree on 0.2.303 |
| 11-19 | Move `ajs/omip-adjoint-oc117` to Reactant 0.2.309 (`8cfba671`); keep the `cse_reduce_transposed` exclusion | 0.2.309 reproduces 0.2.303's gradients and removes the first-call cost; dropping the exclusion needs its own compile-time check |
| 11-26 | Heat- and freshwater-flux controls through the smoothing S, `flux_smoothing_passes` (default 0; use 2) (user) | Their sensitivities are grid-scale but real; 2 passes bring them to the stress's smoothness (finding 28) |
| 11-26 | Accept the frozen-CATKE adjoint's bias in convecting columns (user) | Exact along stress and stable-column fluxes; the alternatives are a stable-column control mask or smoothed CATKE switches, a physics change |
| 11-26 | Pause sharding within a window; calibrate with windows over MPI (user) | Two Reactant gaps block the sharded build, and IFRT, the only runtime where sharding works, makes each gradient ~4× slower (finding 27) |
| 11-26 | Test the frozen κ gradient against the misfit surface J(κ) from forward windows (user) | Finding 22's unfrozen reference is wrong past ~120 steps (finding 24); forward runs need no adjoint |
| standing | Manifests and `../omip_adjoint_handoff/` stay uncommitted | Local manifests |
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
- **Static instability as the cause of the long-window blow-up (11-15/16).** All 20 largest heat-flux
  sensitivities of the 1-year adjoints lay in statically unstable columns (100 % against 14–17 % of wet columns,
  1145882–4), but convective adjustment at 10 m²/s (1146808) moved the blow-up to the Agulhas region (2e20)
  instead of removing it.
- **Grid-scale noise and horizontal dissipation.** Laplacian (1000 m²/s) and biharmonic (λ = 50 days) tracer
  diffusion cut the forward grid-scale noise but not the adjoint growth; forward Laplacian viscosity 10⁴ and
  5×10⁴ m²/s (1147584, 1147586) leave its rate unchanged (3e13 and 2e17, moving to the Southern Ocean);
  adjoint-only damping at 10⁴ and 5×10⁴ m²/s is NaN at 1 year (3588713/4) and only ~17× smaller at 90 days.
- **Time step, GM/Redi, WENO (on 0.113.5).** Δt = 1 h (3576780), no GM/Redi (3576463) and linear third-order
  upwind advection for momentum and tracers (3576772, 1.5e161) all blow up at 1 year.
- **Forward drift.** 60 days from ECCO's own 1993-11 state (1149976) blows up within a month.
- **"The 90-day window is healthy."** True only on 0.113.5 (3574422); on 0.117.2 it is 1e23 (3588711). The
  seasonal reading (January bounded, November not) compared runs on different Oceananigans versions.

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

5. **The adjoint is the exact local derivative (11-10 FD scans, 8/84/360 steps).** The objective is
   piecewise smooth: jumps of 1e-7 to 1e-3 Sv appear as h grows (none at 8 steps), from discrete switches in
   the forward model; small-h Taylor checks fail on the jumps, not on the gradient. Large-h secants match
   gᵀd to 0.05–2 % at 30 days.
6. **Long windows: the adjoint grows by about one e-fold per 17 h** and reaches 1e13–1e23 or NaN within one
   to three months of the window's start, near coasts and fronts (NE Atlantic, Grand Banks, Agulhas, the
   Antarctic margin), whatever the dissipation, Δt or advection scheme (dead ends above).
7. **The forward model does not have that growth (11-18, `perturbation_growth.jl`).** Twin runs of the
   compiled step with 1e-6 K noise in T: rms ΔT grows ×60 (from 1993-11, 1150762) and ×110 (from 1993-01,
   1150763) in 60 days, an e-fold of 12–15 days; 1e-3 K saturates (×4–9); the ε = 0 twins are bitwise
   identical. Single cells jump 100× within 3 days: the forward map has steep switches. So the adjoint
   explosion is an artifact of differentiating something steep, not physical sensitivity, and damping can
   only mask it.
8. **The Oceananigans 0.117.2 migration made it worse.** Same 90-day January window: J 16.71 → 17.91 Sv and
   max|∂J/∂τx| 3.9 → 1.2e23, hot spot at 50°W 42–48°N next to the coast. The migration changed WENO's
   near-land treatment (`boundary_scheme` → `minimum_buffer_upwind_order`, 3 for tracers, 1 for momentum),
   the Zipper (#6396) and the fold halos. The near-boundary advection is the prime suspect; the late-year
   blow-up predates the migration. **Refuted 11-19 (finding 10):** without WENO the window still explodes.
9. **Reactant 0.2.309 removes the first-call cost (1150765, Julia 1.12).** 2-step test: primal first call
   68.8 → 4.3 s, gradient first call 3095 s → 19.1 s; compiles unchanged (6.2 and 20.2 min); Taylor verified
   (2.2e-4). Under Julia 1.13 the same Reactant stalls in the first compile for > 3 h 49 min at 0 % GPU
   (1147598), so the stall is Julia 1.13's.
10. **The 90-day adjoint grows in bursts at two places, whatever the advection (11-19, `AdjointTrace`).** Same
   1993-01 window on 0.117.2 and Reactant 0.2.309, ocean shadows summarized daily in the reverse sweep. WENO
   (1153999; J 17.9094 Sv as 3588711): baseline about 2e4 until day 85 of the window, then 2e4 → 5e19 in about
   four days at the Kuroshio separation (145.7°E 39.2°N, top two levels; S and the v tendency lead), a plateau
   near 1e14 over days 77–57, and a second burst at the Grand Banks (310.7°E 43.4°N, near the surface) to 4e26
   by day 51. Linear third-order upwind (1154000; J 15.0931 Sv): bounded until day 69, then the same two
   places in the same order (Kuroshio Extension 160°E 42.5°N days 69–66, Grand Banks from day 64), ending at
   2.3e101. So the near-boundary WENO change is not needed for the 0.117.2 explosion; the hot spots are the
   wintertime western-boundary-current separations near the surface, with growth of several decades per
   day in short bursts rather than a steady e-fold. Both branches build GM/Redi with `FluxTapering(1e-2)`
   through `omip_closure`, so the 0.113.5 → 0.117.2 change is not in OMIPSimulations' closure setup.
11. **GM/Redi drives the 90-day explosion on 0.117.2, and one step's linearization is unstable at single cells
   (11-19).** The traced WENO window with GM/Redi off (1155069) stays bounded: the adjoint peaks at 1.25e5
   (day 70) and settles near 1e4, max|∂J/∂τx| 2.7 (with GM/Redi: 8.8e27 and 4.1e21); J 18.2888 Sv. Power
   iteration of one step's reverse pass at fixed states (`step_jacobian` stage, 1155076; noise seed in u, v, T,
   S; 60 iterations, converged by about 10) gives a leading growth of 16.2 per step at x₉₇₈ (inside the
   sweep's Kuroshio burst) and 3.99 per step at x₁₀₄₄ (where the sweep grows 1.2 per day). Both modes sit in
   one cell at level 18 of 20 (S leads, then the u tendency): the Barents Sea (35.2°E 69.7°N) and the Grand
   Banks (313.0°E 43.7°N); the shadow of Redi's `ϵκR₃₃` peaks in the same cells. The tangent-linear step
   has the same eigenvalue, so this is the linearization, not the reverse pass. Frozen mixing freezes only
   CATKE's κu, κc, κe: the isopycnal slopes in the GM/Redi fluxes and in `ϵκR₃₃` (stored for the implicit
   vertical solve) are still differentiated, and their derivative scales with 1/N², large in weakly
   stratified winter cells. Caveats: the fixed-state growth is not the sweep's (the sweep grows only where
   its adjoint already is, on a moving state), and the 1-year blow-up on 0.113.5 is NaN without GM/Redi too
   (3576463), so it has another cause as well.
12. **Holding the buoyancy gradients constant in derivatives bounds the 90-day adjoint (11-20).** Config key
   `frozen_buoyancy_gradients` (`7f69e3bd`) stores ∂xᵣb, ∂yᵣb, ∂zb in the `BuoyancyForce` and stop-gradients
   them after each `compute_buoyancy_gradients!`, so the triad slopes, tapers, stratification masks and N²
   are constant in the adjoint; the GM/Redi fluxes are still differentiated in the tracer they carry.
   One-step growth: x₉₇₈ 16.2 → 1.00 per step, x₁₀₄₄ 3.99 → 0.99 (1156065); with GM/Redi off instead,
   1.21 and 1.30 (1156053), u and v modes at level 17 near the equator at 319–321°E. The
   90-day traced window (1156066) stays bounded: prognostic adjoint ≤ 4.9e3 (GM/Redi off 1.25e5, on 8.8e27),
   max|g| 3.9; the u tendency shadow peaks at 3.2e7 for one sample (state 996, 50.6°E 13.8°S, level 18). Its
   gradients differ from the GM/Redi-off run's by 30–98 % of max, as expected for different physics; the
   frozen gradient is an approximation (it drops the sensitivity through the slopes), like frozen mixing.
   The J of 1156066 (17.9075 Sv, 1.1e-4 from 1153999) came from a forward-model change, not rounding:
   CATKE's TKE equation (`step_closure_prognostics!`) runs after each stage's tracer update and before
   `update_state!`, so with stored gradients it read the stage's starting N² (CPU, `buoyancy_gradient_check.jl`:
   e differs in every wet cell after stage 1, κc by up to 1.04 m²/s, T by 0.22 K after one step). `6df20f28`
   recomputes stored gradients before the TKE step; the CPU models then agree to 4.4e-16 after one step and
   1.4e-14 after four. Rerun 1157589: J 17.90934 Sv (2.1e-6 from 1153999's 17.90938), prognostic adjoint
   ≤ 2.8e3, max|g| 3.9; its flux gradients differ from 1156066's by 26–34 % of max (stress 2–4 %).
   Stored gradients still differ from inline ones in the vertical halos (k = 0, Nz + 1) and the fold's top halo,
   which nothing reads beyond rounding. The final shadow S also holds 4 NaN in the top halo at the fold
   (j = 61), next to the u, v fold-halo NaN already there with GM/Redi off; the gradient is finite.
13. **With frozen buoyancy gradients the 1-year adjoint is finite but still bursts (11-20).** 1157526 (1993-01
   start, 4380 steps, frozen mixing and gradients, traced daily, `6df20f28`): J 21.2056 Sv, gradient finite,
   max|∂J/∂τx| 7.6e5 at 129°E 1°S (month 5), freshwater flux 3.9e7 at 121°E 23°N; every earlier 1-year
   variant was NaN or ≥ 1e13. The prognostic adjoint is ≤ 7e3 from day 365 back to day 242 (one u spike of
   2.5e6 on day 240 at 36.5°E 27°S, level 14, gone the next sample), then S grows
   1.8e5 → 2.7e11 within days 206–204 in one column at 296–298°E 61°S (west of the Antarctic Peninsula),
   levels 6–8, and stays at 0.5–2e9 in that column back to day 0 (no decay, as for a tracer adjoint with
   little mixing). One-sample u, v spikes of 1e9–1.5e11 in the tropics (days 145, 137, 98, 74) do not persist.
14. **At 1° the barotropic substep sets stability (11-21).** Forward-only health runs, 90 days from 1993-01,
   360 × 180 × 20, ECCO4 fluxes, daily snapshots: Δt = 2 h with 50 substeps (144 s) is NaN by day 1
   (1158106), so 3591885 failed in its first day, not late in the window. Δt = 2 h with 100 substeps (72 s,
   1158107) and Δt = 1 h with 50 substeps (72 s, 1158108) both run 90 days and agree: J 15.35 and 15.40 Sv on
   day 90, SST rms 1.89 K for both, max vertical CFL 0.08–0.16 (2 h) and 0.04–0.08 (1 h) at 125°E 5°N, κc max 3.00 m²/s
   throughout. Both start with a transient, J 34 → 45 Sv over days 1–3, then settle. Halving the
   barotropic substep is enough, so Δt = 2 h with 100 substeps is the cheaper setting (3.1 min per 90 days,
   the same as 50 substeps; Δt = 1 h takes 4.2 min). A barotropic CFL limit on the finer grid would explain
   it; not checked.
15. **At 1° frozen buoyancy gradients do not bound the 90-day adjoint (11-21).** 1158947 (Δt = 2 h, 100
   substeps, frozen mixing and gradients, traced daily): J 16.3630 Sv, compile 16.6 min, gradient 585 s, every
   forcing gradient NaN. The prognostic adjoint is 4e1–4e3 over days 90–86, then 2.2e6 on day 85 (u at 36°E
   26°S, level 18), 4e7–1e15 over days 84–80 (S and u at 45°E 41°S, levels 7–12), 1e17–2e35 over days 78–71
   (u, v, S at 51°E 11°S, levels 5–20), and overflows by day 69: two to five decades per day, all in the
   southwest Indian Ocean (Mozambique Channel, Agulhas Return Current, north of Madagascar), not only near
   the surface.
16. **The 2° 90-day frozen-gradient adjoint matches finite-amplitude secants to 0.2–0.6 % (11-22).** 1159299
   (frozen mixing and buoyancy gradients, same window as 1157589; J 17.9093 Sv, max|g| 3.901): along the
   stress-gradient direction gᵀd = 18.8228; secants Δf/h = 18.9268, 18.9426, 18.8652 at h = 0.1, 0.05, 0.025
   (gᵀd lower by 0.55 %, 0.63 %, 0.22 %). The secants are not monotone in h (jumps in J), so the script's
   extrapolated slope 19.129 (rel. error 1.6 %) and its "MISMATCH" verdict are the small-h Taylor criterion,
   not a gradient error; the job is FAILED for that `@test` only. One direction tested.

17. **At 1° the one-step Jacobian is unstable in momentum, with or without GM/Redi (11-22).** `step_jacobian`
   at states 1002 and 1026 (days 83.5, 85.5) of the 1° 90-day window, 60 iterations, converged by iteration 10.
   Frozen mixing and buoyancy gradients (1160770, 1160771): 23.7 per step (42°E 3°S, level 15) and 19.2
   (45°E 42°S, level 12, the Agulhas Return Current site of 1158947's sweep). GM/Redi off, inline gradients
   (1160555): 10.3 (154°E 35°S, level 16) and 6.46 (346°E 51°S, level 11). Every mode is led by u, v and
   their tendencies, 87–90 % of its sum of squares in a 3×3×3 box. At 2° the same tests gave 1.00/0.99 and
   1.21/1.30 (finding 12), so 1° has another unstable term, not GM/Redi and not the buoyancy gradients;
   unidentified (untested: momentum advection, vertical friction, Δt = 1 h).
18. **At 1° WENO's weights carry the one-step growth (11-22).** Same window and model days as finding 17,
   frozen mixing and buoyancy gradients. Linear third-order upwind advection (1161247): 0.99 per step at
   both states, the mode spread over the domain (largest 3×3×3 box < 1 % of its sum of squares). WENO at
   Δt = 1 h, 50 substeps (1161248, states 2004 and 2052): 2.91 (358°E 44°S, level 16) and 4.66 (6°E 55°N,
   level 20, with η and the barotropic transports), i.e. 8.5 and 22 per 2 h against 23.7 and 19.2: not a
   time-step effect. Each variant has its own forward trajectory. Not yet run: the 1° 90-day sweep with
   upwind3 and frozen gradients. At 2° upwind3 alone did not bound the sweep (1154000), but that run
   differentiated the slopes.
19. **At 1° the 90-day adjoint is bounded with upwind3 advection (11-22).** 1161433: 1158947's window (Δt = 2 h,
   100 substeps, frozen mixing and buoyancy gradients, traced daily) with linear third-order upwind advection.
   J 14.5108 Sv (16.3630 with WENO), max|∂J/∂τx| 1.198, compile 12.8 min, gradient 437 s. Prognostic adjoint
   4.2e1 at day 90, then 3e2–7.2e2 over the whole sweep (peak 7.23e2 on day 60), led by S in the subtropical
   North Atlantic (343°E 26°N level 8–9; 282–287°E 26–35°N near the surface; 341°E 11°N). Only u, v and S at the
   fold row (j = 181) are NaN, as in every run. So at 1° WENO's weights are the remaining linearization
   instability; why they are benign at 2° is open.
20. **At 2° WENO is not benign in every month (11-22).** The January frozen-gradient window (1157589) repeated
   from 1993-04, 07 and 10 (1161899–1161901; J 17.61, 18.72, 23.05 Sv). All three burst in the momentum
   tendencies Gⁿ.u, Gⁿ.v, not in tracers: April to 3.3e8 on day 79 (32–34°E, 27–30°S, level 12), October to
   1.5e9 on day 43 (32°E 36°S, level 13), both decaying within two days to ≤ 1e5 (max|∂J/∂τx| 4.5 and 4.4);
   July to 8.5e14 on day 74 at 73°W 32°N at the surface, staying near 1e10 to day 0 (max|∂J/∂τx| 2.8e7). The
   1° mechanism (finding 18) therefore exists at 2° too, intermittently. With upwind3 (1162458–1162461) all four
   start months are bounded: adjoint ≤ 2.5e4 over the whole sweep (the seed's peak on day 88 and 1.8e4 on day 58,
   the same cells in every month), max|∂J/∂τx| 3.9–4.6, J 15.09, 14.75, 15.79, 19.96 Sv. Calibration windows use
   upwind3.
21. **The NaN beside the fold is 0 × NaN from zero metrics (11-23).** `TripolarGrid` sets Δxᶠᶜᵃ, Δyᶠᶜᵃ, Azᶠᶜᵃ to
   zero (inverses Inf) at the fold's singular u-points (1, Ny) and (Nx/2 + 1, Ny). Every cell there is land, so the
   forward model masks the NaN that forms in Gⁿ.u (all levels), ∂xᵣb and Gⁿ.T, S, e (levels 1 and Nz at i = 1, 90,
   91, 180), and the state stays finite; reverse mode multiplies zero adjoints by it. That is the NaN in the u, v,
   S shadows of every run, harmless for the forcing gradients, but ∂J/∂κ sums the κ shadow, NaN at exactly those
   16 cells (1162370's κ census). `regularize_fold_metrics!` (`d5421ded`) gives the singular points their eastern
   neighbor's metrics: on the CPU the 2° model is bit-identical after 3 steps and the interior NaN is gone. The
   southern halo rows (j ≤ 0) have zero metrics too (w is NaN there) but no shadow is NaN there.
22. **The frozen adjoint underestimates ∂J/∂κ over 90 days (11-23).** Twin calibration on the 2° 90-day upwind3
   window (1164314, frozen mixing and buoyancy gradients; misfit of the last-30-day mean T, S at all 112,082 wet
   cells, σ 0.1 K and 0.01 g kg⁻¹, truth κ = 1000, 1000): at κ = 600, 600 the adjoint ∂J/∂log κ is 0.31
   (κ_skew) and 0.76 (κ_symmetric) times the central difference (h = 0.05). The central difference is right:
   BFGS's first step lowered J by 4.7e4, more than the adjoint's linear prediction (2.1e4), which a J convex
   along the step forbids; the central difference predicts 6.8e4. Both switches only drop derivatives
   (`ignore_derivatives`), so the forward run is exact and the adjoint misses how κ's change of T and S feeds
   back through the isopycnal slopes and CATKE. In a twin both gradients vanish at the truth, so BFGS still
   recovers κ: 1000.5, 999.7 after 9 iterations (J 7.1e4 → 2.1). With real observations the zero of the frozen
   gradient is not the minimum. The 8-step Kap8 κ Taylor check (1164313) cannot test this: 8-step AMOC moves
   by ≤ 5e-4 Sv for Δκ ≤ 64 m² s⁻¹ and its differences change sign with h. The κ shadows themselves are finite.
   Attribution (11-24, ratio adjoint / central difference at h = 0.05, κ_skew and κ_symmetric): 90 days with
   CATKE differentiated, adjoint NaN (1165326); 14 days frozen 0.38 and −0.39 (1165327), slopes differentiated
   0.45 and −0.12 (1165328), nothing frozen 1.22 and 1.60 (1165329). So the freeze bias appears within 14 days
   (freezing changes GM's own response to T and S, and gets ∂J/∂κ_symmetric's sign wrong), and CATKE cannot be
   differentiated over 90 days. With nothing frozen the central differences do not converge (1166153): for κ_skew
   −8.8e3, −1.59e4, −1.54e4, −1.39e4, −1.24e4, −1.00e4 at h = 0.8 to 7.8e-4 (÷4 each), drifting away from the
   adjoint's −1.87e4; for κ_symmetric 475, 140, 154, 152, 165, 414 against 247. The 14-day misfit is rough in κ
   below h ≈ 0.2 (the forward map's steep switches, finding 9), so no single derivative exists to verify; the
   unfrozen adjoint is within 18 % of the h = 0.2 secant for κ_skew, the frozen ones are 1.4–2.7× smaller than
   the secants at h ≤ 0.2. κ_symmetric's gradient is 1 % of κ_skew's and is at the roughness level.
23. **The heat and freshwater flux sensitivities are grid-scale noise, frozen or not (11-24).** Measure: the
   energy share of the three alternating components (checker, stripes in i and j) of each 2×2 block, Atlantic
   260–380°E, by latitude band; white noise gives 0.75, a smooth field near 0. 1° 90-day frozen upwind3 map
   (1161433): heat flux 0.60–0.85 north of 45°N, rising to 0.63–0.92 at 0–22°N toward the objective time; stress
   0.06–0.45; 57 % of the last slice's checker energy in the box at 282°E 24°N (Florida Straits). 2° 90-day maps
   0.4–0.9, as noisy with GM/Redi off. 14-day 2° upwind3 gradients from 1993-01 (slice 1, then 2): nothing frozen
   0.63–0.79, 0.67–0.84 (1167067); CATKE frozen 0.58–0.94, 0.60–0.96 (1167068); CATKE and slopes frozen the same
   to two digits (1167069). Split by the forward run's convection (1167070, mean unstable interfaces per column
   over days 0–14): nothing frozen, 0.77 in convecting blocks and 0.95 in stable ones; CATKE frozen, 0.88 and
   0.75. The largest 100 heat-flux sensitivities lie in convecting columns (79–83 %, against 16 % of wet
   columns) in all three. So freezing CATKE moves the noise into the convecting columns but does not cause it:
   the exact adjoint is already at the white-noise level. The forward run's surface T has a grid-scale residual
   of 0.26 K rms on day 1, growing to 0.35 K by day 14 (max 13 K at 119°E 8°N).
24. **The CATKE-differentiated adjoint is wrong from about step 120; the frozen-CATKE adjoint is right (11-24 to
   11-26, `adjoint/handoffs/checkerboard.md`).** Central differences of the 14-day window along flux and stress
   directions (`fd_scan` over any forcing control, ChkFD2 1168469): the frozen-CATKE adjoint matches to 0.1–0.3 %;
   the unfrozen adjoint gives FD/adjoint 0.889 along stress and 0.16 for one heat-flux cell. Unfrozen, the
   end-of-window AMOC is exact at 8, 42 and 84 steps (0.99–1.01; ChkFD8 1168601, ChkB42 1168771, ChkB84 1168772)
   and wrong at 168 (0.07–0.82, ChkB168 1169004). A one-step dot-product test (`step_dot` stage) finds step 120:
   v-input adjoint 7.1e-5 against FD 4.8e-3 (ChkDot 1168959, ChkDot120 1169075; not GC, ChkDotGC7 1169296). One
   cell at the mixed-layer base where e is near its floor (314.7°E 46.7°N, level 16, e 6e-8 under 1e-4 m² s⁻²)
   gets a reverse-pass value 60× its neighbors' and cancels the step's inner product (ChkDotF120 1169321); the
   next anomalies have the same structure at the Kuroshio Extension, the sites of the long-window bursts.
   Holding e constant in derivatives after each TKE step (`frozen_tke`) removes the spike and is right along
   stress (1.015) and in stable columns (1.02), but along convecting directions gives −0.19 to 0.03 (ChkTke168
   1170952, ChkTke14 1170953); also freezing κ where κc ≥ 0.1 m² s⁻¹ (`frozen_mixing_above`) does not help
   (0.10, −1.7; ChkTkeA01 1171105). The rest is CATKE's convective and near-neutral switches; no stop-gradient
   mask fixes it. Both flags stay on `ajs/omip-adjoint-checkerboard` (`17ae1305`, `180dc3be`), not on `oc117`.
   Finding 22's unfrozen 14-day κ ratios (1165329) are therefore suspect too.
25. **The flux sensitivities' grid-scale part is real; the stress gradient is filtered by construction (11-25 to
   11-26).** 2° 14 days, frozen CATKE: FD/adjoint 0.997 along the grid-scale heat-flux part in never-convecting
   blocks. 1° 90 days (1161433's window, ChkFD1deg 1171121): 1.00 ± 0.03 along the same part, 1.003 along stress
   (converged); in convecting blocks the model responds about 2× the frozen adjoint (1.8–3.2, rough).
   `_prescribe_surface_fluxes!` interpolates the cell-center stress to faces (`ℑxᶠᵃᵃ`, `ℑyᵃᶠᵃ`), so ∂J/∂τx has
   no component alternating in i and ∂J/∂τy none alternating in j (0.00–0.04 in every run); the fluxes enter at
   centers and keep all three. The surviving stress stripe is as large as the flux's at 2° (0.27–0.45 against
   0.32–0.41) and smaller at 1° (0.11–0.25 against 0.14–0.33). At 1° the stable columns, not the convecting
   ones, are the most grid-scale near the objective (0.91 against 0.66; ChkH1deg 1171100), so convection does not
   carry it. Advection barely matters (2° 90-day upwind3 1162458 against WENO). Biharmonic tracer diffusion at
   50 days lowers the 14-day share only to 0.53–0.73 (ChkBh50 1168067); the forward surface-T residual 0.26 K is
   already in the ECCO4 initial state (ChkBh50H 1168467).
26. **Windows over MPI scale: one iteration of N windows costs one window (11-24 to 11-26,
   `adjoint/handoffs/parallelization.md`).** The 3× slowdown of 1166675 was MPICH_jll's collectives progressing
   only when a rank next calls MPI (ParTime4 1168099: reductions released exactly when rank 0 re-entered MPI);
   `MPIR_CVAR_ASYNC_PROGRESS=1` and `MPI.Init(threadlevel = :multiple)` fix it (ParTime4b 1168552: 2.2 min per
   4-window iteration, J identical to 1166675). Revolve's checkpoint slots allocated once and overwritten
   (`76710bdd`): bit-identical, gradient 120 → 78 s, GC 38 → 1–2 s, host allocation 110 → 0.6 GiB, iteration
   2.2 → 1.5 min (ParSlot1 1170932); 4 windows 1.55–1.73 min (ParSlot4 1170933). No cheap compile sharing: XLA's
   kernel cache saves nothing and reuses a wrong kernel (`CUDA_ERROR_ILLEGAL_INSTRUCTION`, ParKc2 1168843);
   Reactant's `persistent_kernel_cache_enabled` sets a removed XLA option (Reactant.jl #3845).
27. **Sharding needs Reactant's IFRT runtime (11-25).** The sharded 2° build fails in `set_bottom_height!` with
   a `TypeError` from a broadcast between views of sharded PJRT arrays (LS6 ParShard2 3612502), and
   `copyto!(::ConcretePJRTArray, ::Array)` writes a host array into the first shard's buffer only, corrupting
   memory; Oceananigans' `set!` and checkpoint restore make that call (Reactant.jl #3844). With
   `xla_runtime = "IFRT"` both are correct, including the failing build step (CPU, two virtual devices).
   Environment `adjoint/ifrt` (same packages, the preference added). Single-device 8-step reference on two A100s:
   J 24.1308164238 Sv, gradient compile 26.5 min. Under IFRT the sharded build still fails, and the gradient call
   is ~4× slower (11-26, `adjoint/handoffs/parallelization.md`). Two processes on two Vista nodes (ParShardV
   1171559): `copyto!(::ConcreteIFRTArray, ::Array)` puts the host array on the mesh's first device, which the
   other process cannot address (`regularize_fold_metrics!`). One process with two A100s (ParShardI 3615047): the
   first kernel on the sharded grid, `compute_numerical_heights!`, receives one field by two paths and needs a
   reshard, which Reactant does not implement (`codegen_unflatten_aliased_args!`, `TODO(#2234)`). Single device
   under IFRT: J and gradients identical to PJRT, compile times the same, gradient 82.8 s against 18.9 s (LS6)
   and 81.7 against 27 s (Vista).
28. **Smoothed flux controls make the flux gradients as smooth as the stress's (11-26,
   `adjoint/handoffs/checkerboard.md`).** Controls c with Q = Q₀ + S c, so ∂J/∂c = Sᵀ ∂J/∂Q; S is `passes`
   passes of a 1-2-1 average in i (periodic) and then in j, over wet–wet edges only, closed at the southern
   boundary and the fold (`smooth_flux_controls`, `flux_smoothing_passes`, `1bc70878`). Dot test
   ⟨S x, y⟩ = ⟨x, Sᵀ y⟩ to ≤ 5.5e-15 (CPU, 2° and 1° masks). Grid-scale share of the heat-flux sensitivity, raw /
   1 pass / 2 passes against stress-x: 2° 14 days (1167068) 0.58–0.96 / 0.24–0.56 / 0.12–0.31 against
   0.28–0.84; 1° 90 days (1161433) 0.17–0.92 / 0.07–0.52 / 0.04–0.29 against 0.06–0.45. GPU check, 1 pass
   (ChkSmooth 1171615): FD / adjoint 0.86 ± 0.02 over 6 octaves for one smoothed cell, whose S d reaches a row of
   convecting columns (the raw cell, in stable columns, gave 0.999); along white heat- and freshwater-flux
   directions the central differences do not converge (5× spread over h), so those two say nothing. The accuracy
   limit is the frozen-CATKE adjoint in convecting columns (findings 24–25), with or without S.
29. **J(κ) has a derivative at h ≥ 0.0125; the frozen adjoint is 0.25–0.6 of it in κ_skew (11-27).** Twin
   misfit surfaces (truth 1000, 1000; frozen mixing and buoyancy gradients, upwind3) on a 17 × 17 log grid from 250
   to 4000 m² s⁻¹ and on cuts through 600, 600 in log steps of 0.0125, by forward windows (`scan` stage of
   `calibrate.jl`; KapScan14 1172435, KapScan90 1172436). The grid minimum is the truth; the surface is a bowl,
   steeper above the truth than below. At 600, 600, central differences of ∂J/∂log κ_skew agree within ±10 % from
   h = 0.0125 to 0.35: −1.39e4 to −1.67e4 (14 days), −1.88e5 to −2.27e5 (90 days); one-step slopes scatter ±30 %
   (14 days) and ±50 % (90 days), so the roughness is below h ≈ 0.01, not 0.2 as finding 22 read it. Adjoint /
   central difference for κ_skew 0.35–0.42 (14 days) and 0.31–0.37 (90 days), steady over h. κ_symmetric: 90 days
   0.57–0.90 (settling at 0.82–0.90 for h ≥ 0.1); 14 days its slope is 1 % of κ_skew's and rough, with the
   adjoint's sign wrong. At the other points (against the nearest grid node, h = 0.17–0.35): κ_skew 0.25–0.6,
   κ_symmetric 0.6–0.9, signs right. The bias differs per component, so the frozen gradient points in a rotated
   direction and its zero is not J's minimum away from a twin. Cost per gradient: adjoint 17 s (14 days) and
   93 s (90 days); central differences with 4 forward windows 6 s and 36 s.
## Tooling added

| File | Purpose |
|---|---|
| `compile_attribution.jl` + `mlir_diagnostics.jl` | Model build by stage; per window variant (`fluxes`, `ocean`, `coupled`) and mode, the MLIR stage vs XLA stage, Julia compile inside each, op histogram and largest tensor types; switches `pass_timing`, `dump_stages`, `excluded_passes`, `xla_debug_options` |
| `pass_bisect.jl` | Runs Reactant's `all` pipeline on a dumped module in named segments (`pipeline`, `patterns: [i, j]`, `patterns_except`, `patterns_split`, per-segment `module` restart) with time and op count per segment |
| `test_omip_adjoint.jl` | Stages `cpu`, `primal`, `gradient`; config keys `excluded_passes`, `while_unroll_threshold` |
| `omip_adjoint.slurm` | One script for Vista and LS6; `SCRIPT=` and `CONFIG=` select the run |
| Session scratch (not in the repo) | `hlo_census.py`, `while_state.py`, `while_sample.py`: computation output bytes, while-state histograms and body classification on an XLA HLO text dump |
| `src/checkpointed_gradient.jl` | Revolve sweep of the compiled step and reverse-step thunks; objective averaged over the last `average_steps`; optional adjoint damping |
| `src/prescribed_flux_ocean.jl` | ECCO4 prescribed-flux ocean, `ecco_adjoint_setup` (`start_date`, `n_months`, closure and advection options) |
| `src/adjoint_damping.jl`, `test_adjoint_damping.jl` | `AdjointDamping`: forward damping step D on scratch fields, its transpose by Enzyme after each reverse step; forward check and dot-product test |
| `forward_health.jl`, `plot_forward_health.jl`, `plot_convection_vs_adjoint.jl` | Forward diagnostics of an adjoint window (CFL, grid-scale noise, unstable columns, κ, SST against ECCO4) and their maps |
| `perturbation_growth.jl` | Twin forward runs: growth of ε × noise added to T |
| `step_jacobian` stage (`src/step_jacobian.jl`, configs `jacobian_steps`, `jacobian_iterations`) | Power iteration of one step's reverse pass at fixed states of the window: leading growth per step, per-field summaries per iteration, final shadow u, v, T, S |
| `frozen_buoyancy_gradients` config key (`src/frozen_mixing.jl`, `freeze_buoyancy_gradients!`; `materialize_buoyancy_gradients` keyword of the setups) | Stores the buoyancy gradients and holds them constant in derivatives: GM/Redi slopes, tapers and N² frozen like CATKE's diffusivities |
| `buoyancy_gradient_check.jl` (config `buoyancy_gradient_check_jan1993.json`) | CPU: stored against inline buoyancy gradients at every stencil index (interior and halos), then both models stepped side by side with per-step u, v, T, S, η differences |
| `AdjointTrace` (`src/adjoint_trace.jl`, config `adjoint_trace_every`) | Per-field summaries of the ocean adjoint (largest finite entry and its indices, root sum of squares, non-finite count) every N reversed steps, saved as `adjoint_trace` |
| `reduce_gradients.jl`, `omip_cpu.slurm`, `plots/` | Reduction on LS6 CPU nodes; plots locally (`plot_lead_sensitivity.jl`, Pillar maps) |
| `grid_scale_share.jl`, `grid_scale_directions.jl` | Grid-scale share of the forcing sensitivities by band, convecting/stable split for stress and heat flux; alternating and smooth parts of a gradient as scan directions |
| `fd_scan` and `step_dot` stages of `test_omip_adjoint.jl` | Central differences along any forcing direction (`fd_controls`, `fd_directions`, per-label `fd_h0`); one compiled step's central difference against its reverse pass at saved states (`gc_every`) |
| `calibrate.jl`, `calibrate_mpi.slurm`, `sharding_probe.jl`, `adjoint/ifrt/` | Per-rank evaluation, GC and reduction times; MPICH progress thread; `PROJECT=`/`SCRIPT=`; multi-process sharding probe (`distributed` key) under the IFRT environment; `scan` stage (J on κ grids, adjoint at chosen points) |
| `src/flux_control_smoothing.jl`, `flux_smoothing_passes` key | Flux controls through the smoothing S: gradient stage saves Sᵀ g, `fd_scan` moves along S d |
| `test_omip_adjoint.jl` keys since 11-07 | `checkpoints`, `average_days`, `frozen_mixing`, `eddy_diffusivity`, `advection`, `Nx`/`Ny`, `horizontal_diffusivity`, `biharmonic_tracer_timescale`, `convective_adjustment`, `laplacian_viscosity`, `adjoint_viscosity`/`adjoint_diffusivity`, `start_date`, `repeat_evaluation` |

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

### Jobs since 11-11 (2°, ECCO4 forcing, frozen mixing, AMOC averaged over the last 30 days unless noted)

| Job | What | Outcome |
|---|---|---|
| 3574422 / 3574421 | 90 days from 1993-01, frozen / full CATKE (0.113.5) | frozen bounded (max 3.9, J 16.7094 Sv); full NaN |
| 3576176, 3576763 | 1 year, 180 days (0.113.5) | NaN (J 20.13, 15.70 Sv) |
| 3576463, 3576772, 3576780 | 1 year without GM/Redi, with upwind3 advection, at Δt = 1 h (0.113.5) | NaN, 1.5e161, NaN (then out of memory in the repeat) |
| 3577970 | 8-step checkpointed gradient on 0.117.2 | verified; the reference for the damping checks |
| Vista 1144920 | 2-step gradient on Vista (aarch64) | verified; first call 52 min |
| Vista 1145882–4 | 1-year forward health (no / Laplacian 1000 / biharmonic λ = 50 tracer diffusion) | forward bounded; adjoint hot spots all in unstable columns |
| Vista 1146807 / 1146808 | convective adjustment 10 m²/s: health / 1-year gradient | worst N² 34× weaker; adjoint 2e20 at the Agulhas |
| Vista 1147583–6 | forward Laplacian viscosity 10⁴ / 5×10⁴: health and 1-year gradients | forward bounded; adjoint 3e13 / 2e17 |
| Vista 1147598 | Reactant 0.2.309, Julia 1.13.3, 2-step test | primal compile stalled > 3 h 49 min (TIMEOUT) |
| LS6 3587706, Vista 1147994 | damping dot-product test | 2.6e-15, 3.9e-14 |
| Vista 1148278 / 1148279 | 8-step gradient, damping 0 / 10⁴ m²/s | 2e-12 / 0.6–2.4 % from 3577970 |
| LS6 3588711 / 3588712 | 90 days from 1993-01, damping 0 / 10⁴ (0.117.2) | 1.2e23 / 7e21 (J 17.9094 Sv) |
| LS6 3588713 / 3588714 | 1 year, damping 10⁴ / 5×10⁴ | NaN |
| Vista 1149976 | 60 days from 1993-11 (`start_date`) | 8.9e18 at 60°W 43°N |
| Vista 1150762 / 1150763 | twin forward runs from 1993-11 / 1993-01 | e-fold 15 / 12 days (finding 7) |
| Vista 1150765 | Reactant 0.2.309, Julia 1.12.7, 2-step test | verified; gradient first call 19.1 s |
| Vista 1153869 | Reactant 0.2.309 on `oc117`, 2-step ECCO4 test | primal 8.1e-13 from the CPU twin; Taylor verified; first gradient call 19.5 s |
| Vista 1153870 | Reactant 0.2.309 on `oc117`, 8-step gradient with `adjoint_trace_every = 4` | J 4.6e-14, stress gradients 1.7e-12, flux gradients 5e-10 from 3577970; same Taylor rows; first call 27.1 s |
| Vista 1153999 / 1154000 | 90 days from 1993-01, frozen, traced daily: WENO / upwind3 (0.117.2, 0.2.309) | 1.3e22 (J 17.9094 Sv) / 2.3e101 (J 15.0931 Sv); finding 10 |
| Vista 1155069 | 90 days from 1993-01, frozen, traced daily, GM/Redi off (WENO) | bounded: adjoint ≤ 1.25e5, max\|∂J/∂τx\| 2.7, J 18.2888 Sv; finding 11 |
| Vista 1155076 | `step_jacobian` at x₉₇₈ and x₁₀₄₄ of the 90-day window, 60 iterations | 16.2 and 3.99 per step, single-cell modes (Barents Sea, Grand Banks, level 18); finding 11 |
| Vista 1156053 | `step_jacobian` as 1155076, GM/Redi off | 1.21 and 1.30 per step; finding 12 |
| Vista 1156065 | `step_jacobian` as 1155076, `frozen_buoyancy_gradients` | 1.00 and 0.99 per step; finding 12 |
| Vista 1156066 | 90 days traced as 1153999, `frozen_buoyancy_gradients` | bounded: prognostic adjoint ≤ 4.9e3, max\|g\| 3.9, J 17.9075 Sv (1.1e-4 from 1153999; lagged N² in CATKE's TKE step); finding 12 |
| Vista 1157589 | 1156066 rerun with `6df20f28` | prognostic adjoint ≤ 2.8e3, max\|g\| 3.9, J 17.90934 Sv (2.1e-6 from 1153999); finding 12 |
| Vista 1157526 | 1 year from 1993-01, traced daily, frozen mixing and gradients (`6df20f28`) | finite; S burst to 2.7e11 at 297°E 61°S on day 204, max\|g\| 7.6e5, J 21.2056 Sv; finding 13 |
| LS6 3578000, 3580072 | 1° 1 year and 90 days (H100) | node c318-002 could not create a CUDA context |
| LS6 3591885 | 1° 90 days (H100, c318-001) | forward NaN at Δt = 2 h, 50 substeps |
| Vista 1158106 | 1° forward health, 90 days, Δt = 2 h, 50 substeps | NaN by day 1; finding 14 |
| Vista 1158107 | as 1158106, 100 substeps | finite 90 days, J 15.353 Sv on day 90; finding 14 |
| Vista 1158108 | as 1158106, Δt = 1 h, 50 substeps | finite 90 days, J 15.396 Sv on day 90; finding 14 |
| Vista 1158947 | 1° 90 days traced, 100 substeps, frozen mixing and gradients | adjoint NaN, growth from day 85 in the SW Indian Ocean; J 16.3630 Sv; finding 15 |
| Vista 1159299 | 90 days, frozen mixing and gradients, secant check (h = 0.1, 0.05, 0.025) | gᵀd 18.823 vs secants 18.93, 18.94, 18.87 (0.2–0.6 %); `@test taylor.verified` fails on the small-h criterion; finding 16 |
| Vista 1159828, 1159829 | 1° `step_jacobian`, frozen gradients / GM/Redi off | failed at once: `jacobian_steps` not ascending (config error) |
| Vista 1160555 | 1° `step_jacobian` at 1002, 1026, GM/Redi off | 10.28 and 6.462 per step, momentum-led; finding 17 |
| Vista 1160554, 1160721, 1160722 | 1° `step_jacobian`, frozen gradients (both states; one state each) | GPU out of memory in the forward steps; fixed by `GC.gc()` every 20 steps (`be2f14a4`) |
| Vista 1160770, 1160771 | 1° `step_jacobian` at 1002 / 1026, frozen mixing and gradients | 23.7 and 19.2 per step, momentum-led; finding 17 |
| Vista 1161247 | 1° `step_jacobian` at 1002, 1026, frozen mixing and gradients, upwind3 | 0.99 per step at both, diffuse mode; finding 18 |
| Vista 1161248 | 1° `step_jacobian` at 2004, 2052, frozen mixing and gradients, Δt = 1 h | 2.91 and 4.66 per step; finding 18 |
| Vista 1161433 | 1° 90 days traced, frozen mixing and gradients, upwind3 | bounded: adjoint ≤ 7.2e2, max\|g\| 1.198, J 14.5108 Sv; finding 19 |
| Vista 1161897 | 2° 8 steps, κ as fields, Taylor check in κ | J and forcing gradients equal 1153870's; ∂J/∂κ NaN (sum over the parent array includes the fold's NaN shadow); rerun with the κ census as 1162370 |
| Vista 1161899–1161901 | 2° 90 days traced, frozen mixing and gradients, from 1993-04, 07, 10 | momentum bursts: 3.3e8 and 1.5e9 (recovering), July 8.5e14 (max\|g\| 2.8e7); finding 20 |
| Vista 1162370 | Kap8 rerun with the κ census | κ shadow NaN at i = 1, 90, 91, 180, j = 60, k = 0, 1, 20, 21: interior; finding 21; rerun with the fold fix as 1164017 |
| Vista 1162458–1162461 | 2° 90 days traced, frozen mixing and gradients, upwind3, from 1993-01, 04, 07, 10 | all bounded, ≤ 2.5e4, max\|g\| 3.9–4.6; compile 12.2–12.6 min, gradient 159–165 s; finding 20 |
| Vista 1161912 | MPI probe, 2 nodes, MPICH_jll with `srun --mpi=pmi2` | cross-node Allreduce and bcast correct; PMIx unsupported |
| Vista 1164017 | Kap8 with the fold fix | census 0 of 13 NaN; ∂J/∂κ_skew −3.24e-7, ∂J/∂κ_symmetric −7.03e-7 Sv per m² s⁻¹; stopped at the known 8-step forcing Taylor mismatch (4.255e-3, as 1153870) |
| Vista 1164313 | Kap8, κ Taylor checks only | κ shadows finite and equal to ∂J/∂κ; both checks MISMATCH: 8-step AMOC differences are noise in κ (finding 22) |
| Vista 1164260–1164263 | 8-step gradient with both patterns excluded, `while_deadresult` re-enabled, `cse_reduce_transposed` re-enabled, all patterns | compile 21.3, 21.8, 21.4, 21.4 min; identical J, gradient and Taylor table: both exclusions dropped |
| Vista 1165326–1165329 | Calibration gradient checks: 90 days CATKE differentiated; 14 days frozen, slopes differentiated, nothing frozen | NaN; 0.38/−0.39; 0.45/−0.12; 1.22/1.60 (finding 22) |
| Vista 1166153 | 14-day check, nothing frozen, h = 0.8 to 7.8e-4 | central differences do not converge: rough misfit (finding 22) |
| Vista 1165330 | Calibration phase 2: 4 windows (1993-01, 04, 07, 10), 4 nodes, MPI, frozen | compile 12.5 min; κ 600, 600 → 1000.0, 1000.0 at iteration 8 (J 3.2e5 → 6.1 at iteration 10); then out of GPU memory after 11 gradients despite `GC.gc()` in `evaluate!`, other ranks hung to the 3 h limit; memory traced in 1166395 |
| Vista 1166395 | Phase 1 rerun logging device memory per BFGS iteration | 1.0–1.1 GiB in use after every iteration, 0.15 GiB after GC: no held buffers; still out of memory in the forward window of iteration 10 (XLA's pool is ~72 GB). Garbage of the forward window's steps outran the host-triggered GC; stepping loops now collect when the pool is half full |
| Vista 1166514 | 1166395 with the half-pool collection (`2aa9e25b`) | peak 35.9 of 71.3 GiB, iterations unchanged (~2.2 min); κ 1000.5, 999.7 after 9 iterations; iteration 10's line search found no decrease (J 2.1, the frozen gradient at the misfit's roughness) and BFGS stopped |
| Vista 1166675 | Calibration phase 2 rerun with the half-pool collection: 4 windows, 4 nodes | κ 600, 600 → 1000.0, 1000.0 at iteration 8, J 3.2e5 → 6.1 at iteration 10; device memory peak 35.9 of 71.3 GiB, 0.15 GiB after GC every iteration; 1 h 39 min. ~7 min per iteration against ~2.2 min with one window (1166514), and rank 0's GPU at 0 % in 17 of 20 five-minute samples: the ranks wait (`adjoint/handoffs/parallelization.md`) |
| Vista 1167067–1167069 | 2° 14 days traced from 1993-01, upwind3: nothing frozen, CATKE frozen, CATKE and slopes frozen | compile 17.2, 16.9, 12.4 min; gradient 48–52 s; J 15.5628 Sv; max\|g\| 5.74, 4.56, 4.56; heat and freshwater flux sensitivities grid-scale in all three (finding 23) |
| Vista 1167070 | forward_health, 2° 14 days from 1993-01, upwind3 | 1947–2389 unstable interfaces; surface T grid-scale residual 0.26 → 0.35 K rms; daily maps for finding 23 |
| Vista 1168067–1171121 (`Chk…`) | Checkerboard agent: biharmonic, flux finite differences, step bisection, `step_dot`, `frozen_tke`, 1° scans; table in `handoffs/checkerboard.md` | findings 24, 25 |
| Vista 1168099–1170933, LS6 3612502 (`Par…`) | Parallel agent: per-rank timing, MPICH progress thread, kernel cache, sharding probe, preallocated slots; `handoffs/parallelization.md` | findings 26, 27 |
| Vista 1171559, LS6 3615047 | `ParShardV` (2 nodes, IFRT, multi-process), `ParShardI` (two A100s, IFRT) | both fail in the sharded build (finding 27) |
| Vista 1171615 | `ChkSmooth`: central differences along smoothed flux directions, 1 pass | finding 28 |
| Vista 1172435, 1172436 | `KapScan14`, `KapScan90`: J on κ grids (17 × 17 from 250 to 4000 m² s⁻¹, cuts through 600, 600 in log steps of 0.0125) and the frozen adjoint at five points, 14- and 90-day twins (`calibration_twin_ecco4_nz20_{14,90}day_scan.json`, `50fd79d6`) | finding 29 |
| Vista 1164314 | Calibration phase 1: twin, 2° 90 days from 1993-01, upwind3, frozen | compile 12.8 min, window 16 s, gradient ~130 s; check ratios 0.31, 0.76 (finding 22); BFGS 600, 600 → 1000.5, 999.7 in 9 iterations, then out of GPU memory (discarded shadow models; `GC.gc()` before each evaluation) |

## Open

- 1-year window with frozen buoyancy gradients (finding 13): find what grows S 1e6× in two days at 297°E
  61°S on day 204 (e.g. `step_jacobian` at states 2448–2472). Also diff the closure between 0.113.5 and 0.117.2.
- Oceananigans: `step_closure_prognostics!` should refresh materialized buoyancy gradients itself (TODO in
  `src/frozen_mixing.jl`); with `materialize_gradients = true` CATKE otherwise reads lagged N².
- 1°: the 90-day adjoint explodes with frozen mixing and gradients (finding 15); the one-step Jacobian is
  unstable in momentum with or without GM/Redi (finding 17) and neutral with upwind3 advection (finding 18).
  The 1° 90-day adjoint with upwind3 and frozen gradients is bounded (finding 19). Open: why WENO is benign at
  2°; holding WENO's weights constant in derivatives.
- JAMES manuscript draft `adjoint/papers/differentiable_ocean_manuscript.tex`: placeholders in red.
- Julia 1.13 compile stall with 0.2.309: rerun with the SIGUSR1 profile for an upstream report.
- Upstream reports still open: Enzyme-JAX #3965 (batch fission); Oceananigans #6529, #6530, PR #6531 (posted
  from ReactOcean).
- Calibration: the frozen adjoint's κ gradient is biased over 90 days, and the 14-day misfit is rough in κ
  (finding 22; its unfrozen reference is itself wrong past ~120 steps, finding 24). Twin recovery works with 1
  and 4 windows (1166514, 1166675). The J(κ) scans (finding 29): the frozen adjoint is 0.25–0.9 of the true
  slope, and central differences in log κ (h ≈ 0.1–0.2) are exact and cheaper for the two κ.
- Flux controls: smoothing merged (finding 28); `calibrate.jl` does not use flux controls yet.
- Sharding: paused 11-26 (finding 27). Resume when a window no longer fits on one GPU, after the two Reactant
  gaps are fixed and IFRT's per-call cost is understood; then `Partition(3, 1)`, 1°, and windows × shards.
- Unfrozen CATKE adjoint: would need smoothed switches in the forward model (finding 24), a physics change;
  not pursued (bias accepted 11-26).
- Upstream: Reactant.jl #3844 (sharded PJRT `copyto!`), #3845 (kernel-cache preference), filed 11-26.
  Reactant.jl #3847 (multi-process IFRT `copyto!`) and an MWE on #2234 (aliased reshard), posted 11-26.
- Resolved since 11-06: finding 4 (`4cc69c3e`); `cse_reduce_transposed` excluded (`fb295782`) and fixed upstream
  (Enzyme-JAX #3970); zero shadow carries (#3971) and `uitofp` (#3969) fixed upstream, in Reactant 0.2.309;
  the Vista first-call failure (gone with per-step thunks); `atlantic_ocean_basin` avoided by the staircase
  section; `oc117` on Reactant 0.2.309 (11-19); `while_deadresult` and `cse_reduce_transposed` run again (11-23).

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

### 2026-11-07 to 11-10 (moved from STATUS.md on 11-18)

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

### 2026-11-11 to 11-13

- Time-averaged objective and flux gradients (`b413a808`), Pillar-style lead maps (`1b807e70`). 90 days frozen
  bounded (3574422), full CATKE NaN (3574421); 1 year and 180 days frozen NaN (3576176, 3576763). Variants
  without GM, with upwind3, at Δt = 1 h all blow up (3576463, 3576772, 3576780).
- 11-12: merged upstream `ss/omip-prototype` (`f9d52dcb`), Oceananigans 0.117.2; advection helpers called with
  the new signature (`e00134e3`). Re-verified: JRA55 2-step Taylor 1.6e-4, ECCO4 8-step (3577970).
- Writeup `papers/omip_amoc_adjoint.tex`, three-page and seven-page reports for the advisor.

### 2026-11-14 to 11-16

- Forward health checks (`forward_health.jl`): CFL small (internal-wave 0.44 in the smallest wet cell), SST
  drifts 3 °C over a year, grid-scale noise grows at fronts and convective regions. Tracer diffusion
  (Laplacian, biharmonic) and the departures from Simone's `omip_simulation(:halfdegree)` recorded in the
  advisor report (`2359d646`).
- Vista gradients work (1144920). `REACTANT_ENZYME_ISSUES.md` lists eight issues; 11-16 Enzyme-JAX fixed #3969
  (`uitofp`), #3970 (`cse_reduce_transposed`) and #3971 (zero shadow carries), reported from ReactOcean.
- Convection check: 1-year adjoint hot spots all in unstable columns; convective adjustment and forward
  viscosity did not remove the growth (dead ends).

### 2026-11-16 to 11-18

- Adjoint-only damping implemented (`939d9986`): the forward damping step on scratch fields between halo fills,
  transposed by Enzyme so the adjoint's pending halo contributions and the fold signs come out right. Dot
  product 2.6e-15; 8-step zero damping 2e-12 from the undamped gradient; 10⁴ m²/s changes it 1–2 %. It does not
  bound the 90-day or 1-year adjoints.
- `start_date` (`7e70e544`): 60 days from 1993-11 explode, so the forward drift is not the cause. Twin forward
  runs (`c5b51c48`): forward perturbations grow at 12–15 days per e-fold in both seasons (finding 7).
- The 90-day January window explodes on 0.117.2 (finding 8); the earlier healthy result was 0.113.5's.
- Reactant 0.2.309 under Julia 1.12 removes the first-call cost (finding 9); under Julia 1.13 the first compile
  stalls.
- 1°: the first H100 attempts hit a bad node (c318-002); on c318-001 the forward model goes NaN at Δt = 2 h.

### 2026-11-19

- `oc117` on Reactant 0.2.309 (`8cfba671`): 2-step test verified (1153869); the 8-step gradient matches 3577970 to
  round-off with a 27 s first call (1153870). The `cse_reduce_transposed` exclusion stays.
- `AdjointTrace` (`4d86b63a`). The 90-day January window traced with WENO and with upwind3 (1153999, 1154000):
  both explode in bursts at the Kuroshio separation, then the Grand Banks (finding 10), so the near-boundary
  WENO change is not the cause.
- Job 3577970's own Taylor verdict is MISMATCH (relative error 4.3e-3): R1 is non-monotonic above h = 0.0125
  and flat below it, the jumps in J; the smallest-h secant matches gᵀd to 1e-5.
- GM/Redi off, same traced window (1155069): bounded (adjoint ≤ 1.25e5, max|∂J/∂τx| 2.7). `step_jacobian`
  stage (`7fbf624e`; 1155076): one step's adjoint grows 16.2 per step at x₉₇₈ and 3.99 at x₁₀₄₄, in single
  cells near the surface (Barents Sea, Grand Banks) where Redi's `ϵκR₃₃` shadow also peaks (finding 11).

### 2026-11-20

- GM/Redi off, `step_jacobian` (1156053): 1.21 and 1.30 per step, so GM/Redi carried the 16.2.
- `frozen_buoyancy_gradients` (`7f69e3bd`): one-step growth 1.00 and 0.99 (1156065); the traced 90-day window is
  bounded, prognostic adjoint ≤ 4.9e3, max|g| 3.9 (1156066). J moved 1.1e-4 from 1153999 (finding 12).
- The J shift: stored gradients made CATKE's TKE step read the stage's starting N² (CPU,
  `buoyancy_gradient_check.jl`). Fixed in `6df20f28`; rerun 1157589 gives J 2.1e-6 from 1153999, still bounded.
- 1-year window with frozen gradients (1157526): finite for the first time, but S bursts to 2.7e11 at 297°E
  61°S on day 204 and max|g| is 7.6e5 (finding 13).

### 2026-11-21

- 1° forward health (1158106–8): Δt = 2 h with 50 substeps is NaN by day 1; 100 substeps and Δt = 1 h both
  finite for 90 days and agree to 0.04 Sv in J (finding 14). Next: the 1° gradient at Δt = 2 h, 100 substeps.
- 1° gradient (1158947): forward fine (J 16.36 Sv), adjoint NaN with frozen mixing and gradients; bursts from
  day 85 in the southwest Indian Ocean (finding 15). Next: 1° `step_jacobian`, with and without GM/Redi.
- First draft of the JAMES manuscript (`8678e808`). Forget et al. (2015, Sect. 4.2): the MITgcm adjoint already
  omits the dependence of the isopycnal slopes on density, so the paper claims the diagnosis, not the remedy.

### 2026-11-22

- Secant check of the 2° 90-day frozen-gradient adjoint: 0.2–0.6 % along its own direction (1159299, finding 16).
- 1° `step_jacobian`: 19–24 per step with frozen gradients, 6.5–10 with GM/Redi off, all in momentum
  (finding 17). The stage ran out of GPU memory at 1° until the stepping loop collected garbage (`be2f14a4`).
- 1° `step_jacobian` with upwind3: 0.99 per step; at Δt = 1 h with WENO: 2.9 and 4.7 per step (finding 18).
  WENO's weights carry the 1° one-step growth.
- 1° 90-day traced gradient with upwind3 (1161433): bounded, ≤ 7.2e2, max|g| 1.2, J 14.51 Sv (finding 19).
- 1° lead-time sensitivity maps from 1161433 (`configs/plot_lead_ecco4_onedegree_nz20_90day_frozen_gradients_upwind3.json`)
  and `uniform_response.jl`: the response to a uniform unit flux over the Atlantic agrees with 2° (WENO) to
  1–7 % at a one-month lead for all four fluxes; zonal stress 20–55 % at 59 and 90 days; meridional stress, a
  small residual, does not agree beyond one month. 1° per-area maxima are 1.7–5.5× the 2° ones.
- Pillar et al. (2016), read from the CC BY 4.0 copy on Oxford ORA: MITgcm 1°, 33 levels, 1300-year
  equilibrium under climatological NCEP forcing; objective the monthly-mean *maximum* overturning at 25°N;
  their adjoint neglects KPP ("standard practice", citing Hoteit et al. 2005) and the GM eddy advection ("for
  technical reasons"). So the advisor report's "as Pillar et al. freeze KPP" stands. Their only number for
  Fig. 1: +0.01 N m⁻² eastward stress over 3° × basin width at 25°N in the averaging month → −0.84 Sv
  (per-area ≈ −0.4e-10 Sv N⁻¹). Ours (`uniform_response.jl`, last two slices together, 25–28°N, 80.5–13.5°W):
  −0.53 Sv at 2° (WENO), −0.78 Sv at 1° (upwind3); earlier slices +0.02 to +0.07 Sv, as their positive 3-month
  band.
- Calibration prototype (GM/Redi κ, twin experiment, one MPI rank per window; `calibration.jl`, `calibrate.jl`,
  `calibrate_mpi.slurm`, `sharding_probe.jl`). Phase 0: Kap8 (1161897) gave a NaN ∂J/∂κ; windows from
  April, July and October burst with WENO (finding 20), so the windows are repeated with upwind3. Sharding
  probe on LS6 (3605904, two A100s, partition 2 × 1).

### 2026-11-23

- Fold metrics regularized (finding 21): Kap8 census clean, ∂J/∂κ finite (1164017, 1164313).
- `while_deadresult` and `cse_reduce_transposed` are no longer excluded: same compile time and results on the
  pin (1164260–1164263, `REACTANT_ENZYME_ISSUES.md`).
- Calibration phase 1 (1164314): BFGS recovers κ to 0.05 % in 9 iterations, but the frozen adjoint is 0.31 and
  0.76 of the true ∂J/∂log κ (finding 22). It then ran out of GPU memory. Not held buffers (1166395): the
  forward window's 1080 steps each leave device garbage that only a host-triggered GC frees; stepping loops now
  collect when XLA's pool is half full. Gradient checks to attribute the
  shortfall: 90 days with CATKE differentiated, and 14 days frozen, with slopes differentiated, and with
  nothing frozen.

### 2026-11-24

- Calibration phase 2 (1166675): 4 windows over MPI recover κ = 1000.0, 1000.0 by iteration 8; device memory
  peaks at 35.9 of 71.3 GiB with the half-pool collection.
- Grid-scale noise attribution (1167067–1167070, finding 23): the 14-day exact adjoint's flux sensitivities are
  as noisy as the frozen ones; freezing CATKE only moves the noise into the convecting columns.

### 2026-11-25 to 11-26

- Both handoff agents finished; their branches are merged into `oc117` (`6a775aa6`, `b38c2d84`) without the
  `frozen_tke` and `frozen_mixing_above` flags (`e64484e6`). Findings 24–27.
- Calibration keeps the frozen-CATKE adjoint. Reactant.jl #3844 and #3845 filed.
- Flux smoothing merged (`1bc70878`, finding 28). Convecting-column bias accepted; sharding paused after both
  IFRT probes failed in the build (finding 27). κ scan stage (`50fd79d6`); KapScan14/90 submitted.

### 2026-11-27

- KapScan14/90 (finding 29): J(κ) is smooth at h ≥ 0.0125; the frozen adjoint is 0.25–0.9 of its slope.
