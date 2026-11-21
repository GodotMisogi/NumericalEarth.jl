# OMIP adjoint: status as of 2026-11-21

Branch `ajs/omip-adjoint-oc117` (fork GodotMisogi, Oceananigans 0.117.2, Reactant 0.2.309), shared by Vista and LS6
through `/work/09603/ajseth/vista/.julia/dev/NumericalEarth-oc117`. Reactant 0.2.309 test branch
`ajs/omip-adjoint-r309` (worktree `NumericalEarth-r309`). The pre-0.117.2 code (Oceananigans 0.113.5,
`ss/for-omip`) is `ajs/omip-adjoint`, worktree `NumericalEarth-adjoint` on TACC. Records: `LOGBOOK.md`
(decisions, dead ends, jobs), `experiments/OMIPSimulations/adjoint/PROGRESS.md` (detail to 11-06),
`adjoint/REACTANT_ENZYME_ISSUES.md`, reports in `adjoint/papers/`.

## Where we are

| Item | State |
|---|---|
| Short-window gradient (ECCO4 prescribed fluxes, 2° × 20 levels, Δt = 2 h) | **exact**: 2-step Taylor verified on LS6 (3577970) and Vista (1144920, 1150765); 8-step checkpointed gradient on 0.117.2 is the reference (3577970) |
| Checkpointed (Revolve) gradient | compile flat in window length (~23–30 min); averaged objective; forcing gradients wrt monthly stress, heat and freshwater flux |
| Vista (aarch64) gradients | work since the per-step thunks (1144920); the `fixup` failure is gone |
| Long windows on 0.117.2 | **adjoint explodes** (see below), forward model healthy; bounded with GM/Redi off (1155069) and with frozen buoyancy gradients (1157589); 1 year with frozen gradients finite but bursts to 2.7e11 (1157526) |
| Adjoint-only damping (`AdjointDamping`, `939d9986`) | implemented and verified (dot product 2.6e-15 LS6, 3.9e-14 Vista; zero damping reproduces the 8-step gradient to 2e-12); does **not** contain the growth |
| Reactant 0.2.309 (git `2bb32e28`, Reactant_jll 0.0.445) | **on `oc117` since 11-19** (`8cfba671`): 2-step Taylor verified (1153869); 8-step gradient matches 3577970 to 1.7e-12 (stress) and 5e-10 (fluxes), compile 21.6 min, **first call 27 s instead of 14 min** (1153870); under Julia 1.13 the first compile stalls > 3 h 49 min (1147598) |
| 1° (360 × 180) | forward model NaN by day 1 at Δt = 2 h, 50 substeps (3591885, 1158106); finite for 90 days with 100 substeps (1158107) or Δt = 1 h (1158108) |

## The long-window adjoint growth

The adjoint grows by about one e-fold per 17 h and reaches 10¹³–10²³ (or NaN) within one to three months
before the end of the window, near coasts and fronts. What is known:

| Run (2°, frozen mixing, AMOC averaged over the last 30 days) | Code | Result |
|---|---|---|
| 90 days from 1993-01 (3574422) | 0.113.5 | bounded, max\|∂J/∂τx\| 3.9; J 16.71 Sv |
| 90 days from 1993-01 (3588711) | 0.117.2 | **4.9e22–1.2e23** at 50°W 42–48°N (Grand Banks); J 17.91 Sv |
| 60 days from 1993-11 (1149976) | 0.117.2 | 8.9e18 at 60°W 43°N |
| 1 year and 180 days, all variants | both | NaN or ≥ 1e13 in the last two months |
| 1 year from 1993-01, frozen buoyancy gradients (1157526) | 0.117.2 | finite; S 2.7e11 at 297°E 61°S on day 204, max\|g\| 7.6e5; J 21.21 Sv |

- Ruled out for the 1-year blow-up (each run still NaN): Δt 1 h (3576780), no GM/Redi (3576463), linear
  third-order upwind advection (3576772, 1.5e161), Laplacian (1000 m²/s) and biharmonic (λ = 50 d) tracer
  diffusion, convective adjustment 10 m²/s (1146808), forward Laplacian viscosity 10⁴ and 5×10⁴ m²/s
  (1147584, 1147586), adjoint-only damping 10⁴ and 5×10⁴ m²/s (3588713/4), and forward drift (1149976 starts
  from ECCO's own November state).
- **The forward model does not have this growth.** Twin forward runs of the compiled step with 10⁻⁶ K noise
  (1150762 from 1993-11, 1150763 from 1993-01): rms ΔT grows ×60–110 in 60 days (e-fold 12–15 days, not 17 h);
  10⁻³ K saturates (×4–9). The ε = 0 twins are bitwise identical. Single cells jump 100× within 3 days, i.e.
  the forward map has steep switches. So the explosion is an artifact of the linearization: a steep but
  continuous term whose derivative, chained over many steps, explodes.
- **The 0.117.2 migration made it worse**: same window, J moved 1.2 Sv and the January adjoint went from 3.9 to
  1e23. **Not because of the near-boundary WENO change**: with linear third-order upwind (no WENO) the same
  window still explodes, to 2.3e101 (1154000).
- **Where and how it grows (11-19, `AdjointTrace`, LOGBOOK finding 10).** In both runs the adjoint stays near
  2e4, then jumps several decades per day in short bursts: first at the Kuroshio separation (145–160°E,
  39–43°N), then at the Grand Banks (310°E, 43°N), both in the top levels, led by S and the momentum
  tendencies. WENO bursts at days 83–79 and 56–51 of the window, upwind3 at days 69–66 and 64–60.
- **GM/Redi drives the 90-day explosion (11-19, LOGBOOK finding 11).** With GM/Redi off the same traced
  window is bounded: adjoint ≤ 1.25e5, max\|∂J/∂τx\| 2.7, J 18.29 Sv (1155069). Power iteration of one step's
  reverse pass at a fixed state (`step_jacobian`, 1155076) grows 16.2 per step at day 81.5 and 3.99 at day 87,
  each in one cell at level 18 (Barents Sea; Grand Banks) where Redi's `ϵκR₃₃` shadow peaks. The tangent-linear
  step has the same eigenvalue. Frozen mixing holds CATKE constant but still differentiates the isopycnal
  slopes, whose derivative scales with 1/N². The 1-year blow-up on 0.113.5 is NaN without GM/Redi too
  (3576463), so it has a second cause.
- **Frozen buoyancy gradients bound the 90-day adjoint (11-20, LOGBOOK finding 12).** Config key
  `frozen_buoyancy_gradients` holds ∂xᵣb, ∂yᵣb, ∂zb (so the slopes and N²) constant in derivatives. One-step
  growth 16.2 → 1.00 and 3.99 → 0.99 per step (1156065; GM/Redi off: 1.21, 1.30, 1156053). The traced 90-day
  window stays bounded: prognostic adjoint ≤ 2.8e3, max|g| 3.9, J 2.1e-6 from the inline-gradient run
  (1157589). The first try (1156066) moved J 1.1e-4 because CATKE's TKE step read lagged stored N²; fixed
  in `6df20f28` (CPU agreement to rounding, `buoyancy_gradient_check.jl`).
- **The 1-year window with frozen gradients is finite but not bounded (11-20, LOGBOOK finding 13).** 1157526:
  ≤ 7e3 back to day 242 (one u spike of 2.5e6 on day 240), then S grows 1.8e5 → 2.7e11 within two days (day 204, 297°E 61°S, levels 6–8)
  and stays near 1e9 in that column to day 0; max|∂J/∂τx| 7.6e5 (129°E 1°S).
- Adjoint damping at 10⁴ m²/s lowers the 90-day explosion only ~17×, and costs 1–2 % of the 8-step gradient.

## Waiting on the user

1. Done 11-19: `ajs/omip-adjoint-oc117` is on Reactant 0.2.309 (git pin until registered).
2. Done 11-19: the upwind3 90-day window also explodes (1154000), so WENO's boundary change is not the cause.
3. Done 11-19: reverse sweep traced (`AdjointTrace`; 1153999, 1154000); GM/Redi off is bounded (1155069);
   one-step Jacobian 16.2 per step at the burst (1155076). Done 11-20: GM/Redi off gives 1.2–1.3 (1156053);
   frozen buoyancy gradients give 1.00 and a bounded 90-day adjoint (1156065, 1157589); the J shift was
   lagged N² in CATKE's TKE step (`6df20f28`); the 1-year window is finite but bursts (1157526). Next
   (proposed): `step_jacobian` at the day-204 burst (states 2448–2472) to find the growing term.
4. Done 11-21: 1° forward health; Δt = 2 h needs 100 barotropic substeps (LOGBOOK finding 14). Next
   (proposed): the 1° 90-day gradient at Δt = 2 h, 100 substeps, frozen mixing and buoyancy gradients.
5. Optional: Julia 1.13 + 0.2.309 rerun with the SIGUSR1 profile, for an upstream report.

## Constraints to remember

- LS6 jobs charge `ADCIRC`; Vista jobs too (`-A ADCIRC`). Every job mails ajseth@utexas.edu. `gh-dev` allows 3
  queued jobs per user, ≤ 2 h. LS6 H100 node c318-002 fails to create a CUDA context (exclude it).
- Push via the ssh remote; `git pull --ff-only fork <branch>` on TACC. Pulling into the shared oc117 worktree
  changes the code of every queued job there.
- ECCO's JPL server has been unreachable at times (11-15, 11-17); Vista's depot cache has 1993-01 to 1994-03.
- No environment variables for the Julia scripts; everything comes from the JSON configs.
- Local plots with `julia +1.12 --project=adjoint/plots`; Reactant cannot lower kernels on macOS.
- Manifests and `omip_adjoint_handoff/` stay uncommitted.
