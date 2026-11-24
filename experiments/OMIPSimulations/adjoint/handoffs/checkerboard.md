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
