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
- **ChkB84 1168772** (27 min): 84 steps (7 days), J 16.6787 Sv. FD / own unfrozen adjoint: stress 1.0007,
  heat-flux cell (136, 36) 1.013, never-convecting alternating part 0.990; (131, 38) switches. **Exact at 84
  steps too.** The failing 14-day case differs in three ways: 168 steps, the 7-day mean objective (states
  85–168), and 14 Revolve checkpoints (3 in the bisection runs). The frozen adjoint with that same averaging
  and those checkpoints matched (stress 1.000), so the averaging is not the cause by itself.
- **ChkDot 1168959** (22 min; `ecco4_nz20_14day_upwind3_unfrozen_step_dot.json`, global noise λ in u, v, T, S,
  e and one input field at a time, ε = 1, 1/4, 1/16 of amplitudes u, v 1e-3 m s⁻¹, T 1e-2 K, S 1e-3 g kg⁻¹,
  e 1e-7 m² s⁻²). The T, S and e inputs do not converge in ε at any state, step 1 included: e grows like 1/ε,
  the signature of discontinuities. A global perturbation crosses some of the ~2000 near-neutral unstable
  interfaces, where CATKE switches. So this test cannot judge the derivative in T, S, e. The velocity inputs
  are smooth enough to read:
  - u agrees to 1–9 % at steps 1, 24, 84, 120, 156 and 167, but not at 8 (−0.31 to −0.95) or 48 (+0.29 to +0.54).
  - v agrees to 1–6 % at steps 24, 48, 84, 156 and 167.
  - At step 120, v's FD is 4.8e-3, 4.8e-3, 3.2e-3 against an adjoint of 7.1e-5 (67× apart, steady in ε). Step 8
    disagrees in sign (FD 3.3e-4, adjoint −1.3e-3).
  Step 120 is inside the failing 168-step window and outside the exact 84-step ones, so it is a candidate onset.
  Submitted ChkDot120 1169075 (`..._step_dot_around120.json`: steps 4–10 and 112–128) and ChkB168 1169004
  (168 steps, end-of-window AMOC, 14 checkpoints).

### 2026-11-25, onset between 84 and 168 steps; the garbage collector is the suspect

- **ChkB168 1169004** (27.8 min; 168 steps, AMOC at the window's end, 14 checkpoints; J 18.7899 Sv): FD / own
  unfrozen adjoint is 0.82 for stress, 0.069 for heat-flux cell (136, 36) (FD 3.7–6.8e-5 against 6.6e-4) and 0.067
  for the never-convecting alternating part. So the window length breaks it, not the 7-day averaging: **onset
  between 84 (exact) and 168 steps.** The CATKE state is steady from day 1 (step 12) to day 14 (1167070:
  1947–2389 unstable interfaces, κc at its cap), so the onset is not tied to a change of CATKE regime.
- **ChkDot120 1169075** (22 min; steps 4–10 and 112–128): the v-input adjoint at step 120 is 7.1e-5 against an
  FD of 4.8e-3 (×67). It is the only outlier among 112–128, where the adjoint is 4.1–6.8e-3 and agrees with FD
  to 2–30 %. Step 10's FD does not converge in ε (2.6e-3, 1.7e-3, −3.9e-4), so it is not read.
- Step 120 is the one state in both dot tests that followed a host `GC.gc()`: the step loop collects every 20th
  step, right before the snapshot. In the gradients, `collect_device_garbage` (`2aa9e25b`, 11-24) collects
  whenever XLA's pool is half full. That plausibly first happens between 84 and 168 steps of a Revolve sweep, and
  later or never with the smaller frozen-CATKE reverse thunk. Hypothesis: a collection frees (finalizes) a
  device buffer the reverse pass still uses. Every 14-day run on 11-24 (1167067–69) postdates `2aa9e25b`.
- Test: ChkDotGC7 (`..._step_dot_gc7.json`, `284a2e89`) collects every 7th step and probes 112, 118, 119, 120,
  121, 126 and 127. Under the hypothesis, 112, 119 and 126 go wrong and 120 is fine.
- **ChkDotGC7 1169296** (22.5 min): **the GC hypothesis is wrong.** Collecting every 7th step leaves every probed
  value bit-identical to 1169075: 112, 119 and 126 agree as before (v-input FD / adjoint − 1 within −0.29…+0.17,
  u within −0.15…+0.18), and step 120 again gives 7.1143e-5 against FD 4.83e-3, 4.78e-3, 3.20e-3. So the
  step-120 reverse pass is deterministically wrong in its velocity input, and that is a property of the state
  x₁₂₀. It lies inside the failing 84–168 interval. Since the frozen-CATKE adjoint is right over the same window,
  the likely carrier is a CATKE term whose derivative is singular at some cell of that state (e.g. √e or
  e^{3/2}/ℓ near e's floor, or a stability function near N² = 0). There AD's exact local derivative is
  enormous on a set too small for any finite perturbation to see.
- Submitted ChkDotF120 (1169321, `..._step_dot_fields120.json`, `25b17312`): the same test at 119, 120 and 121, saving the state
  and λ̄ fields to locate the cells where λ̄ at 120 departs from its neighbors.
- **ChkDotF120 1169321** (22 min): **the step-120 anomaly is one column at the base of the mixed layer.** λ̄_v at
  (i, j, k) = (126, 46, 16), 314.7°E 46.7°N (off the Grand Banks, the long-window burst site in STATUS), is −2.2
  at step 119, **+13.4 at step 120** and −0.12 at step 121 (max |λ̄_v| elsewhere ≈ 1–2). The rest of that column
  also flips sign at 120: k = 17…20 go from +0.15…+0.50 to −1.7…−5.2. That one cell contributes −3.3e-3 to the
  inner product, which cancels the global 6.8e-3 → 7.1e-5. Its state: e = 6.1e-8 m² s⁻², between 3.5e-8 below
  (k = 15) and 1.2e-4 above (k = 17), T 6.8 °C. So it sits at the mixed-layer base, where e drops by four
  decades to near its floor. The next-largest departures share that structure, e.g. (46, 44, 16–17) at 160°E
  42.5°N (Kuroshio extension), e 2.9e-9 under 5.5e-5. The u-input spike at (126, 45, 16) has e 1.8e-4 over
  4.3e-7. The step's forward map is smooth for finite perturbations there (FD 4.8e-3 at three ε), so the
  reverse pass has an enormous local derivative at a cell where e is at the bottom of the turbulent layer.
  That is a CATKE term whose derivative blows up as e → its floor (√e in the mixing length or κ = ℓ√e,
  e^{3/2}/ℓ dissipation, or the floor's clamp itself).

**Summary of the bisection.** The unfrozen adjoint is exact (FD / adjoint 0.99–1.01 where J is smooth) at 8, 42
and 84 steps. It is wrong at 168 steps with or without the 7-day average (0.07–0.82). Garbage collection is not
the cause (identical bits at any collection cadence). The first wrong state found is step 120. It is a property
of the state, not of the step count: one mixed-layer-base cell with e near its floor gets a one-step λ̄ 6–60× its
neighbors'. CATKE's regime is steady from day 1, so it is the arrival of particular cells at e's floor under an
active layer that matters, not convection onset. GPU use on this thread: ≈ 4.9 GPU-h (12 jobs).

Recommended (for the user to choose): (a) calibrate with the frozen-CATKE adjoint, which matches the finite
differences of the full model to 0.1–0.3 % on the same window; or (b) step 3 of the plan, restricted to what this
points at. Stop the derivative through e (not κ's N² and shear dependence) where e is within a few decades of
its floor, or regularize the singular factor (√e → √(e + e₀) in the derivative only), then repeat ChkFD2's stress
and single-cell scans. To name the exact CATKE line first, run the one-column version of the step-120 test:
perturb v only in column (126, 46) and λ only at (126, 46, 16) on the saved x₁₂₀. That needs code that snapshots
x₁₂₀ to disk, not yet written.

### 2026-11-25, fix (approved step 3)

- Reading CATKE (Oceananigans 0.117.2, `TKEBasedVerticalDiffusivities`), the terms singular as e → 0 are:
  `wb⁻_e = wb⁻ / e * (e > eᵐⁱⁿ)` in the implicit TKE operator (`fast_tke_tendency`, eᵐⁱⁿ = 1e-9);
  `ω = √|e| / ℓᴰ` (`dissipation_rate`, with `ifelse(e < 0, …)`); and `√max(eᵐⁱⁿ, e)` in the diffusivities
  (`turbulent_velocityᶜᶜᶜ`). All of them depend on velocity through ℓ, wb and shear, so the v-input spike at a
  cell with e = 6e-8 fits any of them. I skipped the one-column test (scope 1): it would only localize again to
  the column already found, and naming the line needs instrumenting the TKE kernels.
- Remedy (i), `17ae1305`: `freeze_tke!` / config `frozen_tke`. It is a method of
  `step_closure_prognostics!(closure_fields, ::FlavorOfCATKE, ::HydrostaticFreeSurfaceModel, Δt)` that calls
  Oceananigans' and then stop-gradients e (`ignore_derivatives`, the pattern of `freeze_mixing!`). So no
  derivative enters the TKE equation; κ still varies with N² and shear. `frozen_mixing` is unchanged, and
  `checkpointed_gradient.jl` is untouched.
  - A CPU check of forward identity is not possible with this harness: two CPU runs from `restore_state!`
    already differ without the flag. The GPU runs check it through J.
- Submitted ChkTke168 1170952 (`ecco4_nz20_168step_upwind3_frozen_tke.json`: 168-step end-of-window gradient plus
  the step 119–121 dot test) and ChkTke14 1170953 (`ecco4_nz20_14day_upwind3_frozen_tke.json`: 1167067's
  configuration). The FD rows of ChkB168, ChkFD and ChkFD2 still apply, since the forward model is unchanged.

### 2026-11-25, why stress is smooth and flux is grid-scale at 1° (added scope)

- **Component split** (offline; the overlapping 2×2 share of `grid_scale_share.jl` split into its checker,
  stripe-alternating-in-i and stripe-alternating-in-j parts; Atlantic 0–80°N; scratch script):

  | Gradient | checker | stripe-i | stripe-j | total |
  |---|---|---|---|---|
  | 1° 90 d frozen upwind3 (1161433), ∂J/∂τx, slices 1–4 | 0.00–0.01 | 0.01–0.03 | 0.11–0.14 | 0.14–0.17 |
  | same, ∂J/∂τy | 0.01–0.02 | 0.17–0.25 | 0.02–0.04 | 0.22–0.29 |
  | same, ∂J/∂Q | 0.03–0.16 | 0.09–0.33 | 0.14–0.22 | 0.26–0.71 |
  | 2° 90 d frozen WENO (1157589), ∂J/∂τx | 0.01–0.02 | 0.01–0.02 | 0.27–0.45 | 0.31–0.49 |
  | same, ∂J/∂τy | 0.03–0.04 | 0.22–0.36 | 0.04–0.10 | 0.33–0.47 |
  | same, ∂J/∂Q | 0.16–0.23 | 0.15–0.26 | 0.32–0.41 | 0.70–0.80 |
  | 2° 14 d frozen CATKE (1167068), ∂J/∂τx / τy / Q | 0.01 / 0.03–0.05 / 0.12–0.13 | 0.01 / 0.37 / 0.12 | 0.34–0.41 / 0.06–0.09 / 0.52–0.60 | 0.36–0.42 / 0.46–0.51 / 0.77–0.83 |

  **The stress gradient is filtered at the grid scale by construction.** `_prescribe_surface_fluxes!` puts the
  stress at cell centers and interpolates it to the faces (`ℑxᶠᵃᵃ` for τx, `ℑyᵃᶠᵃ` for τy). So ∂J/∂τx at a center
  is the mean of the two neighboring u-face adjoints, which annihilates any component alternating in i (the
  checker and the i-stripe), and ∂J/∂τy likewise loses everything alternating in j. Exactly those components are
  at 0.00–0.03 in every run, while the component the interpolation cannot see (τx's j-stripe, τy's i-stripe) is
  as large as the flux's (2°: 0.27–0.45 against 0.32–0.41). The heat and freshwater fluxes enter at the centers
  where they are applied, so their gradient keeps all three components. Part of the contrast is therefore
  measurement: the stress control space carries a two-point average that the flux control space does not.
  At 1° the stress's surviving stripe (0.11–0.25) is also smaller than the flux's (0.14–0.33), so resolution
  adds a real difference on top.
- Preliminary convecting / stable split at 1° (`grid_scale_share.jl` with the WENO health file 1158107; the
  upwind3 one is queued). Heat flux, slices 1–4: convecting 0.39, 0.58, 0.69, 0.85; stable 0.25, 0.30, 0.55, 0.91.
  Stress-x: convecting 0.13–0.28, stable 0.12–0.17. In the slice nearest the objective, the stable columns are
  as grid-scale as the convecting ones, so convection is not the carrier at 1°. `grid_scale_share.jl` now prints
  the split for stress-x and heat flux in every slice. `grid_scale_directions.jl` adds the alternating part in
  convecting blocks and writes zero stress fields.

### 2026-11-25, verification of fix (i) frozen_tke

- **ChkTke168 1170952** (62 min) and **ChkTke14 1170953** (34 min). Both compile in 27 min instead of 15–17.
- **The forward model is not bit-identical; it differs at rounding.** J differs from the flag-free runs by
  −2.2e-10 Sv (168 steps: 18.789870576046 against …576269) and +1.6e-11 Sv (14 days), and the step-dot FDs differ
  in the 9th digit. The stop-gradient assignment changes XLA's fusion. This is far below every FD signal
  (≥ 1e-6 Sv) but is not the "bit-identical" asked for.
- Step-120 dot test: **the anomaly is gone.** The v input gives adjoint 5.06e-3 against FD 4.83e-3 / 4.78e-3 (ε 1 and
  1/4), and 119 and 121 agree to 1–10 % as before. So the step-120 spike comes through e.
- FD / adjoint with frozen_tke (FD rows of ChkB168, ChkFD, ChkFD2):

  | Window, direction | unfrozen | frozen_tke | frozen CATKE |
  |---|---|---|---|
  | 168 steps, stress | 0.82 | 0.928 | — |
  | 168 steps, heat cell (136, 36) | 0.069 | 0.974 (FD ± 24 %) | — |
  | 168 steps, alternating never-convecting | 0.067 | 1.114 (FD ± 18 %) | — |
  | 14 d, stress | 0.889 | **1.015** | 1.000 |
  | 14 d, heat cell (136, 36) | 0.161 | **1.022** | 0.999 |
  | 14 d, frozen gradient's alternating part, never convecting | 0.212 | **1.017** | 0.997 |
  | 14 d, unfrozen gradient's alternating part, never convecting | 0.12 | 1.093 | — |
  | 14 d, unfrozen gradient's heat-flux alternating / smooth | 0.23 / 0.33 | **0.026 / 0.029** | — |
  | 14 d, unfrozen gradient's freshwater alternating / smooth | 0.49 / 0.68 | 0.081 / 0.108 | — |
  | 14 d, frozen gradient's alternating / smooth | 0.73 / 1.44 | **−0.19 / −0.16** | 0.81 / 0.81 |
  | 14 d, heat cell south of the section (convecting) | 0.089 | 0.005 | −3.6 |

  **Fix (i) is right in stable columns and along stress, and badly wrong along convecting directions.** There its
  gᵀd is 9e-3 against FD 2.4e-4, and max|g| is 11.2 against 5.6. With e frozen, what remains is κ's derivative
  in N² and shear. In convecting cells that runs through CATKE's convective and near-neutral switches
  (`step(x, c, w)` ramps, `w★ / √N²⁺` as N² → 0⁺, the 3 m² s⁻¹ caps). These are steep, so their local derivative
  has nothing to do with the response to any practical perturbation. Remedy (ii) (√e regularized in the derivative)
  would not touch them, so I did not build it.
- Fix (iii), `180dc3be`: `freeze_mixing_above!` / config `frozen_mixing_above` (m² s⁻¹). It holds κu, κc and κe
  constant in derivatives in the cells where κc ≥ κ★, keeps them differentiated elsewhere, and is combined with
  frozen_tke. Submitted ChkTkeA01 1171105 (14 days, κ★ = 0.1 m² s⁻¹; convecting cells sit at the 3 m² s⁻¹ cap,
  stable surface cells at 0.003–0.006 in 1167070). Also queued: ChkH1deg 1171100 (1° upwind3 health run).
- Grid-scale share of the frozen_tke 14-day gradient (2°): heat flux 0.48–0.83 by band (unfrozen 0.63–0.84); Atlantic
  convecting / stable 0.77 / 0.76 and 0.76 / 0.83 in slices 1 and 2. Stress-x 0.30–0.84, convecting 0.43,
  stable 0.26–0.28.
- **ChkH1deg 1171100** (14 min; `health_ecco4_onedegree_nz20_90day_upwind3.json`, the forward state of 1161433):
  convecting / stable split with `grid_scale_share.jl`, Atlantic 0–80°N, slices 1–4. Heat flux: convecting 0.35,
  0.47, 0.57, 0.66; stable 0.23, 0.28, 0.55, **0.91**. Stress-x: convecting 0.21, 0.28, 0.12, 0.16; stable 0.17,
  0.12, 0.14, 0.13. These agree with the WENO-state preliminary. In the slice nearest the objective the
  never-convecting blocks are the most grid-scale, so at 1° convection does not carry the flux checkerboard.
- 1° directions (`grid_scale_directions_onedegree_90day_upwind3.json`, 1161433's gradient and 1171100's state; all
  slices; global non-overlapping blocks), heat flux, |part|² / |g|²: alternating 0.379, smooth 0.621, alternating in
  never-convecting blocks 0.097, in convecting blocks 0.110. The unit directions are far less concentrated than
  at 2°: the largest entries are 0.24 (never convecting) and 0.41 (convecting), with 259 and 39 cells above a
  tenth of the largest. The stress direction is 1161433's (τx, τy) gradient (max entry 0.11).
- Submitted ChkFD1deg 1171121 (gh, 2.5 h limit; `ecco4_onedegree_nz20_90day_upwind3_fd.json`): central differences of
  the 30-day-mean AMOC over the full 90-day window, h0 = 100 (heat), 0.4 (stress), 5 halvings, along the three
  directions. The forward window is about 2–3 min at 1°, so ~1.5–2 GPU-h; the 90-day window is kept so the
  directions are 1161433's own, and no shorter-window share check is needed.
- 2026-11-25 ~20:05: my Mac-side job watchers were stopped by Claude Code because the Mac was low on memory. They
  were not restarted. In flight: ChkTkeA01 1171105 (running, ~24 min in; output
  `results/ecco4_adjoint_nz20_14day_upwind3_frozen_tke_above0.1.jld2`) and ChkFD1deg 1171121 (pending on gh;
  output `results/ecco4_fd_onedegree_nz20_90day_upwind3.jld2`). To evaluate them, copy the files to the Mac and
  run the FD-vs-gradient comparison against ChkFD/ChkFD2 (fix iii), or against 1161433's gradient (1°).

### 2026-11-26, fix (iii) result; 1° scans; resolution versus advection

- **ChkTkeA01 1171105** (33 min; frozen_tke plus κ held constant where κc ≥ 0.1 m² s⁻¹). J equals frozen_tke's
  to all digits. FD / adjoint: stress 1.004, heat cell (136, 36) 1.021, never-convecting alternating parts
  1.018 and 1.094. But along convecting directions it is still wrong: the unfrozen gradient's heat alternating /
  smooth give 0.099 / 0.112, the freshwater ones 0.33 / 0.46, the frozen gradient's alternating / smooth −1.7 /
  −1.3, and the section cell 0.018. **Fix (iii) fails.** The spurious term also lives in cells with κc < 0.1
  near convection, or in other N²-dependent CATKE paths (surface buoyancy flux, convective penetration), so a
  κ threshold does not isolate it. As recommended and agreed, I am stopping the CATKE-derivative work here. The
  usable adjoint is frozen CATKE, which matches the full model's FD to 0.1–0.3 % at 2° along smooth directions.
  `frozen_tke` and `frozen_mixing_above` stay on the branch as off-by-default diagnostics; `frozen_tke` shifts J
  at rounding (2e-10 Sv over 168 steps).
- **ChkFD1deg 1171121** (35.7 min; all three 5-step scans ran, 5.5 min each, ~33 s per 1° forward window; f0 is
  1161433's J to 1.8e-15). Central FD / 1161433's frozen adjoint, h = h0 … h0/16:
  - heat-flux alternating part, never-convecting blocks: 1.019, 1.039, 1.006, 0.984, 0.966, i.e. **1.00 ± 0.03**;
  - heat-flux alternating part, convecting blocks: 1.79, 2.03, 1.84, 3.23, 2.08 (rough; FD ≈ 2× the frozen adjoint);
  - stress: 0.888, 0.948, 0.986, 0.9987, 1.0034, converging to 1 as h falls (nonlinear only at the largest h).
  **At 1° the grid-scale heat-flux sensitivity in stable columns is real**, the stress sensitivity is exact,
  and in convecting columns the full model responds about twice as strongly as the frozen-CATKE adjoint
  predicts, with roughness.
- **Task 3 offline** (no GPU): 1162458's 2° 90-day frozen-gradients upwind3 gradient (finding 20) against the
  2° WENO one. Heat flux 0.34–0.91 by band (WENO 0.52–0.92); components τx j-stripe 0.27–0.43 (WENO 0.27–0.45),
  τy i-stripe 0.21–0.36, Q 0.57–0.71 (WENO 0.70–0.80). So advection does not explain the 2°/1° difference;
  resolution does (1°: τx stripe 0.11–0.14, Q 0.26–0.71).
- GPU since the fix was approved: fix scope ≈ 2.1 GPU-h (ChkTke168 1.03, ChkTke14 0.56, ChkTkeA01 0.55); 1° scope
  ≈ 0.8 GPU-h (ChkH1deg 0.23, ChkFD1deg 0.60). Thread total ≈ 7.8 GPU-h.

## Log (flux-control smoothing)

Branch `ajs/omip-adjoint-flux-smoothing` from `bf0c4078` (approved 11-26).

### Plan

- **Control:** Q = Q_ECCO + S c, and the same for the freshwater flux. So ∂J/∂c = Sᵀ ∂J/∂Q, and at c = 0 the forward
  run is unchanged. The stress control stays as it is. Everything is linear post-processing on the host, so the
  compiled program does not change and nothing is recompiled.
- **The operator.** One pass is S₁ = Bⱼ Bᵢ, and p passes are S = (Bⱼ Bᵢ)ᵖ, so Sᵀ = (Bᵢ Bⱼ)ᵖ. Here
  B_d = P (I − L_d / 4) P:
  - P zeroes land;
  - L_d is the graph Laplacian of the wet–wet edges in direction d, symmetric by construction, so each B_d is
    symmetric and Sᵀ is just the reverse order;
  - between wet cells, B is the 1-2-1 average, which removes the checker and both stripes exactly;
  - at a land neighbor the edge is missing, so no weight leaks to land and the cell keeps it (Neumann);
  - i is periodic. The southern edge and the tripolar fold (row Ny) have no edges: the U-pivot fold repeats row Ny,
    so connecting across it would couple duplicated controls, and the Arctic fold row is far from the objective.
- **Code (OMIPAdjoint `src/`).**
  - `smooth_flux_controls(controls, wet; passes, transpose = false)` acts on a named tuple of `(Nx, Ny, 1, Nt)`
    arrays, applying S, or Sᵀ, to `heat_flux` and `freshwater_flux` and leaving `u` and `v` alone.
  - `finite_difference_scan` gains `map_direction` (default `identity`): the controls move along
    `map_direction(d̂)` while gᵀd is taken in the scanned space.
- **`test_omip_adjoint.jl`.** Config key `flux_smoothing_passes` (default 0, today's behavior):
  - the gradient stage stores Sᵀ g as `gradient_heat_flux` and `gradient_freshwater_flux`;
  - `fd_scan` perturbs along S d.
- **Verification.**
  1. Local CPU dot test ⟨S x, y⟩ = ⟨x, Sᵀ y⟩ on the 2° and 1° land masks, for 1 and 2 passes. Also check that S
     removes the checker and the stripes in an all-wet region and leaves land at zero.
  2. One GPU job: FD scan on the 2° 14-day frozen-CATKE window (1167068's configuration, 7-day mean),
     `flux_smoothing_passes` 1, against Sᵀ g computed on the host from 1167068's saved gradient. Directions:
     random white c (heat flux, both slices; then S d is smooth), a single c cell at (136, 36), and random white c
     in the freshwater flux. Target: FD / adjoint within 0.1–0.3 % where ChkFD2 achieved it.
  3. `grid_scale_share.jl` on Sᵀ g, for 1 and 2 passes, on 1167068 (2° 14 days) and 1161433 (1° 90 days),
     compared with the stress shares. Done offline.
