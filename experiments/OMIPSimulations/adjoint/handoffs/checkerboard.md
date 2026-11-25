# Handoff: grid-scale noise in the buoyancy-flux sensitivities

Written 2026-11-24 on `ajs/omip-adjoint-oc117`. Read first: repo-root `STATUS.md`, `LOGBOOK.md`
findings 12, 16, 19, 22 and **23**, and `adjoint/handoffs/coordination.md` (shared rules with the parallelization
agent).

## Goal

The adjoint sensitivities of the AMOC objective to the surface heat and freshwater fluxes are grid-scale noise (a
2×2 checkerboard and stripes) at 1° and 2°; the wind-stress sensitivities are much smoother. The aim is calibration
at < 1°, where flux controls with grid-scale gradients are useless. Deliverables, in order:

1. Decide whether the noise is **real sensitivity of the forward model** (the discrete model really responds to a
   checkerboard flux that way) or an **artifact of the linearization** (the adjoint is wrong along grid-scale
   directions). Everything else depends on this.
2. If an artifact: find the term and fix or regularize it in the adjoint. If real: find which forward term carries
   the grid-scale response (CATKE convection, upwind3, the C-grid) and propose the remedy (scale-selective
   dissipation in the forward model, or a smoothness prior / coarse control basis in the calibration).
3. A smooth (grid-scale share ≲ 0.3) flux sensitivity on the 2° 14-day window, then at 1° for 90 days.

## The measure

`adjoint/grid_scale_share.jl [health.jld2] results.jld2 ...`: the energy share of the alternating components
(checker, stripe in i, stripe in j) of each 2×2 block, Atlantic 260–380°E, by latitude band and forcing slice.
White noise gives 0.75, a smooth field near 0. With a `forward_health.jl` file of the same grid first, it also
splits the heat-flux share by convecting and stable columns. Run locally (`julia +1.12 --project=adjoint`) on
files copied from Vista. `adjoint/plot_convection_vs_adjoint.jl` maps |∂J/∂Q| against unstable columns (its
title still says "1-year window"; fix the title if you use it).

## What is known (finding 23)

| Run | Heat-flux share by band | Convecting / stable columns |
|---|---|---|
| 1° 90 days, frozen CATKE and slopes, upwind3 (Vista 1161433) | 0.19–0.92, rising toward the objective time; stress 0.06–0.45 | 57 % of the last slice's checker energy at 282°E 24°N (Florida Straits) |
| 2° 90 days, frozen (several), GM/Redi on or off | 0.4–0.9 | |
| 2° 14 days from 1993-01, upwind3, **nothing frozen** (1167067) | 0.63–0.84 | 0.77 / 0.95 |
| same, CATKE frozen (1167068) | 0.58–0.96 | 0.88 / 0.75 |
| same, CATKE and slopes frozen (1167069) | identical to CATKE frozen to two digits | |

- The exact adjoint is already noisy: freezing is not the cause, it only moves the noise into convecting columns.
- 79–83 % of the 100 largest heat-flux sensitivities lie in convecting columns (16 % of wet columns), in all three.
- The forward run (1167070, `forward_health.jl`) has a surface-T grid-scale residual of 0.26 K rms on day 1,
  0.35 K on day 14, max 13 K at 119°E 8°N; 1947–2389 statically unstable interfaces; CATKE κ at its 3 m² s⁻¹ cap.
- Freshwater-flux shares equal the heat-flux ones: both enter as surface tracer fluxes, so the noise is in the
  surface adjoint T and S.
- 2° 90-day frozen adjoint matches secants along its own (full) direction to 0.2–0.6 % (finding 16), but that
  direction is dominated by stress and by the smooth part; the grid-scale part was never checked separately.

## Suggested experiments (ranked)

1. **Directional check along the grid-scale part.** Split the 14-day unfrozen flux gradient into its alternating
   and smooth parts; run central differences of the primal window along each (flux perturbations only) and
   compare with the adjoint's inner product. `test_omip_adjoint.jl`'s `fd_scan` stage (`finite_difference_scan`)
   perturbs only stress (`gradient_u`, `gradient_v`); extending it to the flux fields is a small change. Agreement
   on the alternating part means the noise is real; disagreement localizes an adjoint error.
2. **Biharmonic tracer diffusion** (config key `biharmonic_tracer_timescale`, days; the 1-year tests used 50):
   a 14-day unfrozen gradient plus a `forward_health.jl` run with the same key. It damps two-cell modes in both
   directions (self-adjoint operator), so it tests whether an undamped tracer grid mode carries the noise.
3. **Another vertical closure.** `omip_closure` takes `:catke` or `:simple` (convective adjustment, κ = 1
   m² s⁻¹), but neither `test_omip_adjoint.jl` nor `forward_health.jl` passes `vertical_closure` to
   `ecco_adjoint_setup` yet (one keyword each).
4. **The forward state's own grid-scale structure.** Is the 0.26 K residual already in the ECCO4 initial state
   (record day 0 of the health file) or made by the first steps? The upwind3 adjoint's stencil and the frozen
   CATKE κ follow the forward state, so grid-scale forward structure becomes grid-scale adjoint coefficients.
5. At 1°, the Florida Straits box (282°E 24°N) holds most of the checker energy: look at the boundary
   (staircase section of the objective, coastline masking) there.

## How to run

- Configs to copy: `adjoint/configs/ecco4_nz20_14day_upwind3_{unfrozen,frozen_mixing,frozen_gradients}.json`
  (gradient, ~20 min each: compile 12–17 min, gradient ~50 s) and `health_ecco4_nz20_14day_upwind3.json`
  (`forward_health.jl`, 10 min; start date hard-coded to 1993-01-01).
- Vista, from `experiments/OMIPSimulations` in your worktree (see `coordination.md`):
  `CONFIG=adjoint/configs/<c>.json sbatch -J <name> -p gh -t 01:30:00 -A ADCIRC adjoint/omip_adjoint.slurm`;
  for the health run add `SCRIPT=adjoint/forward_health.jl`. Logs:
  `adjoint/logs/<name>.o<jobid>`, results in `adjoint/results/`.
- Results of the runs above are in the shared worktree's `adjoint/results/` on Vista
  (`/work/09603/ajseth/vista/.julia/dev/NumericalEarth-oc117/experiments/OMIPSimulations/adjoint/results/`):
  `ecco4_adjoint_nz20_14day_upwind3_*.jld2`, `forward_health_ecco4_nz20_14day_upwind3.jld2`,
  `ecco4_adjoint_onedegree_nz20_90day_frozen_gradients_upwind3_trace.jld2`.
- Gradient files hold `gradient_u`, `gradient_v`, `gradient_heat_flux`, `gradient_freshwater_flux` as
  (Nx, Ny, 1, slices); the last slice is outside the window and zero. `geometry` has λ (may be Nx + 1 wide), φ,
  area, land.

## Out of scope

The long-window adjoint growth (STATUS "The long-window adjoint growth"), the day-204 Jacobian and the
0.113.5/0.117.2 closure diff (need the user's approval), calibration of κ, parallelization.

## Log

### 2026-11-24

- Branch `ajs/omip-adjoint-checkerboard` from `caf55b76`; worktrees `NumericalEarth-checkerboard` (Mac, Vista).
- Experiment 2 configs: `ecco4_nz20_14day_upwind3_unfrozen_bh50.json` (gradient) and
  `health_ecco4_nz20_14day_upwind3_bh50.json` (`forward_health.jl`), the 1167067/1167070 setups with
  `biharmonic_tracer_timescale` 50 days (2Δ checker e-folds in ~0.13 day).
- Plan for experiment 1 (code):
  - `finite_difference_scan(model, objective!, g; direction = g, ...)`: scans along `direction / |direction|` and
    compares with `gᵀd`; `g` and `direction` are named tuples over any of the forcing controls (`u`, `v`,
    `heat_flux`, `freshwater_flux`), set through one `forcing_fields(model)` (which also gives
    `forcing_gradient`).
  - The 14-day gradients average the AMOC over the last 7 days, so the scan's objective is that average:
    `window_average_objective!` runs the checkpointed thunks forward (refresh, steps, objective), the forward half
    of `checkpointed_gradient`.
  - Directions built on the Mac from 1167067's `gradient_heat_flux` (and freshwater): the alternating part of
    non-overlapping 2×2 blocks of wet cells (an orthogonal projection P, so gᵀd = |Pg|) and the rest (I − P)g,
    saved as results-like files (`gradient_heat_flux`, `J`) that the `fd_scan` stage reads as `fd_direction`.
- Experiment 1 code: `finite_difference_scan(...; direction)`, `forcing_fields`, `control_vector`/`set_control!`
  (were `wind_control`/`set_wind_control!`), `window_average_objective!`; `fd_scan` stage takes `fd_controls`,
  `fd_directions` ({label: file}) and per-label `fd_h0`, and with `average_days` scans the 7-day-mean AMOC through
  the checkpointed thunks. Directions: `adjoint/grid_scale_directions.jl` +
  `configs/grid_scale_directions_14day_upwind3_unfrozen.json` → `results/directions_14day_upwind3_unfrozen_*.jld2`
  (copied to the Vista worktree with 1167067's results). Parts of 1167067's gradient, |part|² / |g|²: heat flux
  alternating 0.806, smooth 0.194, alternating in never-convecting blocks (1167070) 0.200; freshwater 0.764,
  0.236, 0.140. The unit directions are concentrated: the heat-flux alternating one has max entry 0.57 and 25
  cells above a tenth of it, led by 320°E 24.6°N (i, j = 127, 38, the row just south of the 26.5°N section),
  340°E 19.4°N and 329°E 25°N; the never-convecting one is led by 340°E 19.4°N (0.78). Scan config
  `ecco4_nz20_14day_upwind3_unfrozen_fd.json`: h0 100 (heat, W m⁻² per unit d; max perturbation 57 W m⁻²) and
  2e-4 (freshwater, kg m⁻² s⁻¹), 13 halvings.
- Commit `2439af00`. Submitted: ChkBh50 1168067 (gh, experiment 2 gradient), ChkFD 1168096 (gh-dev, experiment 1
  scans). CPU check of the new control helpers (2-step CPU window, random heat-flux direction): set/restore exact,
  gᵀd as expected.
- 19:06: ChkFD 1168096 (c642-022) and ChkBh50 1168067 (c637-062) running. Not yet submitted: the experiment-2
  health run (`SCRIPT=adjoint/forward_health.jl CONFIG=adjoint/configs/health_ecco4_nz20_14day_upwind3_bh50.json
  sbatch -J ChkBh50H -p gh -t 00:45:00 -A ADCIRC adjoint/omip_adjoint.slurm`), held back by the 2-job limit.
- Side observation: 53 % of the heat-flux alternating energy lies within 4° of 26°N (the Atlantic section rows),
  where the local share is 0.85; elsewhere it is 0.76. The AMOC objective is a one-row staircase, so part of the
  grid-scale structure is the objective's own footprint.

### 2026-11-24, results of experiments 1 and 2

- **ChkFD 1168096** (29.5 min; compile of the checkpointed thunks 17 min, each 13-step scan 41–53 s): f0 =
  15.56282979659 Sv, J of 1167067 to 2.3e-16. Central slopes (Sv per unit h) against 1167067's gᵀd:

  | Direction | gᵀd (unfrozen) | central slope, h0 … h0/128 (spread) | FD / adjoint | slope at h ≤ h0/256 | frozen-CATKE gᵀd (1167068) |
  |---|---|---|---|---|---|
  | heat, alternating | 1.04e-3 | 2.35e-4 (±18 %) | 0.23 | 7.5e-4, 1.3e-3, 5.8e-4, 1.1e-3 | 4.7e-5 |
  | heat, smooth | 5.09e-4 | 1.67e-4 (±8 %) | 0.33 | 4.5e-4, 7.4e-4, 4.0e-4, 4.9e-4 | 1.5e-5 |
  | heat, alternating, never convecting | 5.16e-4 | 6.31e-5 (±3 %) | 0.12 | 7e-5, 1.1e-4, −9e-5, 6e-5, 4.7e-5 | 5.58e-5 (FD / frozen 1.13) |
  | freshwater, alternating | 506 | 247 (±35 %) | 0.49 | 381, 531, 368, 411, 612 | 21 |
  | freshwater, smooth | 281 | 190 (±27 %) | 0.68 | 325, 299, 289, 335, 470 | 3.2 |

  h0 = 100 (heat) and 2e-4 (freshwater) per unit direction; the largest entry of each unit direction is 0.36–0.78,
  so h0/128 is a perturbation of at most 0.3–0.6 W m⁻². Reading: below h ≈ h0/256 (≲ 0.2 W m⁻² at the largest
  cell) the central slopes approach the unfrozen adjoint in four of five directions (alternating and smooth
  alike), so the adjoint is the exact local derivative. Above it, J responds linearly with a slope 2–8× smaller,
  stable over 7–8 octaves. The one-sided differences show why: along the heat-flux alternating direction
  f(x₀ + h d) − f0 stays near 3e-5 Sv from h = 0.024 to 0.78, i.e. a step of ~3e-5 Sv (2e-6 of J) within the first
  0.02 W m⁻². The local derivative is carried by tiny steep switches of the forward map (finding 9), the same in
  the smooth and alternating parts; it is not an adjoint error and not specific to the grid scale. In the
  never-convecting direction there is no small-h regime at all, and the frozen-CATKE adjoint matches the
  finite-amplitude slope (1.13); the unfrozen one is 8× too large.
- **ChkBh50 1168067** (26.8 min; J 11.0834 Sv, max|g| 7.28): biharmonic tracer diffusion (λ = 50 days) barely
  lowers the share. Heat flux slice 1 by band (−35–0, 0–22, 22–32, 32–45, 45–80°N): 0.60 0.68 0.67 0.68 0.59
  (1167067: 0.63 0.79 0.76 0.76 0.75); slice 2 0.53 0.67 0.70 0.66 0.73; Atlantic 0–80°N convecting 0.68, stable
  0.66 (0.77, 0.95). Stress-x stays 0.40–0.85 (1167067: 0.28–0.85). Note the 14-day unfrozen stress
  sensitivities are as grid-scale as the fluxes (0.55–0.85 outside 22–32°N). So an undamped tracer grid mode
  is not what carries the noise.
- Submitted ChkBh50H 1168467 (health run of experiment 2) and ChkFD2 1168469 (gh-dev): scans along 1167067's
  stress gradient, single heat-flux cells at 340°E 19.4°N and 320°E 24.6°N (slice 1; directions written by a
  scratch script: unit entry at (i, j) = (136, 36) and (127, 38)), and the alternating, smooth and
  never-convecting alternating parts of the frozen-CATKE gradient (1167068). These test whether the
  frozen adjoint is the finite-amplitude sensitivity along its own directions.

### 2026-11-24, controls (ChkFD2 1168469, 26.6 min) and the verdict

FD slopes are the median central slope over h0 … h0/128 (8 octaves), same forward model and f0 (2.3e-16 from
1167067's J) as ChkFD; gᵀd of each gradient recomputed offline along each direction:

| Direction | FD slope (spread) | FD / unfrozen (1167067) | FD / frozen CATKE (1167068) | FD / frozen CATKE + slopes (1167069) |
|---|---|---|---|---|
| 1167067's stress gradient | 13.908 (±0 %, linear over 12 octaves) | 0.889 | **1.000** | **1.000** |
| heat flux, one cell 340°E 19.4°N, slice 1 | 7.20e-5 (±3 %) | 0.161 | **0.999** | **0.997** |
| frozen gradient's heat-flux alternating part, never-convecting blocks | 7.79e-5 (±9 %) | 0.212 | **0.997** | 0.995 |
| frozen gradient's heat-flux alternating part | 1.69e-4 (±18 %) | 0.729 | 0.806 | 0.805 |
| frozen gradient's heat-flux smooth part | 9.54e-5 (±23 %) | 1.441 | 0.810 | 0.811 |
| heat flux, one cell 320°E 24.6°N (south of the section) | −7.0e-5 (±73 %, switching) | 0.089 | −3.6 | −3.4 |

From ChkFD, along the unfrozen gradient's own parts: FD / unfrozen 0.23 (alternating), 0.33 (smooth), 0.12
(alternating, never convecting), 0.49 and 0.68 (freshwater).

**Verdict.**
1. Differentiating CATKE is the artifact. Where the forward map is smooth (the stress direction, a single
   stable cell), J is linear in the control over 9–12 octaves of h, with no small-h regime. That slope equals
   the frozen-CATKE adjoint to 0.1–0.3 %. The exact (unfrozen) adjoint is 1.12× too large along stress and 6–8×
   along heat flux. So the reverse pass through CATKE adds a sensitivity that the discrete forward model does not
   have: it is not the derivative of J, even locally. My earlier reading (local derivative carried by tiny steep
   switches, after ChkFD) is wrong. The small-h drift in ChkFD's directions is roughness from the convecting
   cells those directions concentrate on.
2. The grid-scale structure of the frozen-CATKE gradient is real. Its alternating part (all blocks and
   never-convecting blocks) is confirmed by the forward model at 0.81 ± 0.18 and 0.997 ± 0.09, about the same as
   its smooth part (0.81 ± 0.23). The frozen gradient's heat-flux share is 0.58–0.96 (finding 23), so the
   discrete model really responds to checkerboard fluxes, mostly in and near convecting columns.
   Near the 26.5°N section the response switches (single-cell scan at 320°E 24.6°N: Δ± of the same sign, no
   linear regime).
3. Experiment 2 (biharmonic tracer diffusion, λ = 50 days) lowers the unfrozen share only to 0.53–0.73, and
   the convecting / stable split to 0.68 / 0.80. So an undamped tracer grid mode is not the carrier. Its
   forward run (ChkBh50H 1168467, 11 min) cuts the surface-T grid-scale residual from 0.35 to 0.18 K by day 14
   (S 0.082 → 0.050), with 2318 unstable interfaces (2380 without). The 0.26 K residual is already in the
   ECCO4 initial state (day 0 of both health files): that is experiment 4's answer.
4. For the calibration: use the frozen-CATKE adjoint (it is the better gradient, here exact to 0.1 % where the
   map is smooth). Treat the flux controls' grid-scale part as real but useless: a smoothness prior or a coarse
   control basis, not an adjoint fix.

Open: why the CATKE reverse pass is wrong. The 8-step test ChkFD8 1168601 (`ecco4_nz20_8step_upwind3_fd.json`:
primal, then FD along single heat-flux cells (131, 38), (132, 38), (136, 36) and the stress direction, then the
checkpointed unfrozen gradient) was queued at 21:10 on gh-dev. If its unfrozen flux gradient already disagrees
with the FD after 8 steps, the CATKE derivative error is debuggable on a short window. Compare each scan's
central slope with this run's own `gradient_*` offline: the scan's printed gᵀd uses 1167067's file only for its
shapes.

### 2026-11-24, the 8-step check (ChkFD8 1168601, 27.4 min)

Config `ecco4_nz20_8step_upwind3_fd.json`: 8 steps from 1993-01, upwind3, nothing frozen, AMOC at the window's end.
J 24.1118 Sv (primal and checkpointed gradient agree to 2.2e-14). FD slope (mean of the 4 smallest h) against
this run's own gradient:

| Direction | FD slope | adjoint gᵀd | FD / adjoint |
|---|---|---|---|
| heat flux (131, 38), 329°E 25°N | +4.7350e-5 | +4.7351e-5 | 0.99998 |
| heat flux (132, 38), 331°E 25°N | −4.770e-5 | −4.751e-5 | 1.004 |
| heat flux (136, 36), 340°E 19.4°N | +7.1134e-7 | +7.1125e-7 | 1.0001 |
| 1167067's stress direction | 16.5736 | 16.5735 | 1.00001 |

All four are linear over 12 octaves of h. **At 8 steps the unfrozen adjoint is exact, heat flux included.** So the
CATKE reverse-pass error arises between 8 and 168 steps. The opposite-signed neighbors (131, 38) and (132, 38)
are a real two-cell sensitivity after 16 h.

Reconciling with the earlier records: the 2-step (1153869) and 8-step (3577970, 1153870) Taylor checks had CATKE
differentiated (no `frozen_mixing` key; default false) and WENO advection, and they perturbed the wind only
(`taylor_remainder_check` scans the wind gradient). They agree with this run. Their "known 8-step forcing Taylor
mismatch 4.255e-3" is the three-point slope criterion; the central differences here match to 1e-5.

Proposed bisection (not run):
1. Step count: end-of-window AMOC at 42 and 84 steps (3.5 and 7 days), with the same four directions plus
   1167067's never-convecting alternating heat-flux part. That is two jobs of ~30 min on gh-dev
   (`stages` "primal,fd_scan,gradient", as `ecco4_nz20_8step_upwind3_fd.json`). Then halve the failing interval
   once or twice more (~0.5 GPU-h per job).
2. The term, once the step is known: a one-step dot-product check of the compiled step at a state just past the
   onset. Compare ⟨λ, (x(s + εv) − x(s − εv)) / 2ε⟩ with ⟨reverse_step(λ), v⟩ for λ the stored adjoint and v a
   random perturbation of T, S, e. This is a small extension of the `step_jacobian` stage. A wrong derivative of
   one step must show here, since a chain of exact step derivatives is exact.
3. Split the freeze: a flag that stop-gradients only e after the TKE step (so κ still varies with N² and shear),
   against one that stop-gradients only the N² and shear inputs of the mixing length. That says whether the
   error is in the TKE equation or in the diffusivity formula. Needs code in `frozen_mixing.jl`.

### 2026-11-24, bisection (steps 1 and 2 of the plan; step 3 held for the user)

- Submitted ChkB42 1168771 and ChkB84 1168772 (`ecco4_nz20_{42,84}step_upwind3_fd.json`, `220be3c8`): end-of-window
  AMOC, FD along heat-flux cells (131, 38) and (136, 36), 1167067's never-convecting alternating heat-flux part and its
  stress direction, then the unfrozen checkpointed gradient.
- New `step_dot` stage (`step_dot_products`, `07c6aa8c`): a one-step dot-product test of the compiled reverse pass
  at fixed states. λ is noise in u, v, T, S, e of the step's output; v perturbs one input field at a time.
  Config `ecco4_nz20_14day_upwind3_unfrozen_step_dot.json` covers states 1, 8, 24, 48, 84, 120, 156 and 167 of
  1167067's window. CPU check of its host helpers: add/read of the interior exact.
- CATKE state over the window (1167070, daily): 44 unstable interfaces and κc_max 0 on day 0, then 1947
  interfaces in 940 convecting columns and κc_max at the 3 m² s⁻¹ cap by day 1 (step 12). It then stays nearly
  steady (2380 interfaces, 1009 columns, on day 14). Surface κc at (127, 38) is at the cap from day 1 (2–3
  unstable interfaces), and (131, 38) convects similarly; (136, 36) never convects (surface κc 0.003–0.006).
  So the 8-step window (16 h) ends before the convective regime is established. An onset between 8 and
  ~12–24 steps would point to the CATKE state rather than the step count.
- **ChkB42 1168771** (27 min): 42 steps (3.5 days, past the onset of convection on day 1), J 53.6036 Sv
  (an adjustment transient; primal and gradient agree to 4.4e-13). FD / own unfrozen adjoint, median over 8
  octaves: heat-flux cell (136, 36) 0.992 (FD 4.89–5.15e-5 against 5.148e-5), the never-convecting alternating
  heat-flux part 1.013, stress 0.998. The convecting cell (131, 38) switches (FD from +3e-6 to −2.9e-4 across h;
  median 1.09). **At 42 steps the unfrozen adjoint is still exact where J is smooth**, so the onset is after
  the convective regime is established: the CATKE state alone is not enough. ChkDot 1168959 (step_dot) queued.
