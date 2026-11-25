# Handoff: parallel calibration (windows over MPI, domain sharding)

Written 2026-11-24 on `ajs/omip-adjoint-oc117`. Read first: repo-root `STATUS.md`, `LOGBOOK.md` finding 22 and the
Vista job rows 1161912, 1164314, 1165330, 1166395, 1166514, 1166675, and `adjoint/handoffs/coordination.md`
(shared rules with the checkerboard agent).

## Goal

Calibration of the ECCO4 prescribed-flux ocean at < 1° resolution. Two levels of parallelism:

- **Across windows (phase 2, works):** one MPI rank and GPU per time window; ranks sum misfits and gradients, all
  run the same BFGS. Deliverable: near-perfect weak scaling (an iteration with N windows costs about one window's
  time) at 4 and then 12 windows.
- **Within a window (phase 3, untested):** the grid sharded across GPUs with
  `Distributed(ReactantState(); partition)`, so a 1/2° or 1/4° window fits and runs fast enough. Deliverables:
  sharded J and gradients equal to single-device ones to rounding at 2°; then a 1° sharded 90-day gradient with
  time and memory per GPU; then the largest resolution that fits, and how the gradient time scales with GPUs.

## Code

- `adjoint/calibrate.jl` (twin calibration; header lists the config keys), `adjoint/calibrate_mpi.slurm` (Vista
  `gh`, one rank per node, `srun --mpi=pmi2`; MPICH_jll inside the container; PMIx does not work, job 1161912).
- `adjoint/src/calibration.jl` (`build_eddy_calibration`, `window_tracer_means!`, `bfgs_minimize`),
  `adjoint/src/checkpointed_gradient.jl` (Revolve over per-step compiled thunks; `collect_device_garbage`, which
  runs a full GC whenever XLA's pool is half full, because each compiled step's replaced buffers are freed only by
  finalizers).
- `adjoint/sharding_probe.jl` + `adjoint/configs/sharding_probe_ecco4_nz20_8step.json`: builds the 8-step 2°
  model on one device and on `Distributed(ReactantState(); partition = Partition(2, 1))`, attempts each stage
  (build, snapshot, compile primal, primal, compile gradient, gradient) separately, reports failures with stack
  traces, and compares J and every forcing gradient.
- Configs: `calibration_twin_ecco4_nz20_90day.json` (1 window), `calibration_twin_ecco4_nz20_90day_4windows.json`
  (1993-01, 04, 07, 10), `calibration_twin_ecco4_nz20_90day_memory.json` (14 iterations).

## What is known

- 1 window (Vista 1166514): build 206 s, compile 12.8 min, truth window 16 s, **~2.2 min per BFGS iteration**
  (one ~130 s gradient), GPU 51–88 % busy; device memory peak 35.9 of 71.3 GiB.
- 4 windows, 4 nodes (1166675): recovers κ = 1000.0, 1000.0 from 600, 600 by iteration 8, memory as above, but
  **~7 min per iteration**, and rank 0's GPU reads 0 % in 17 of 20 five-minute `nvidia-smi` samples
  (`adjoint/logs/gpu.1166675.csv`; only rank 0's GPU is sampled). Every rank compiles the same program
  separately (12.5–12.7 min each). The earlier 90-day gradients from 1993-04, 07, 10 took 159–165 s each
  (1162458–1162461), so the windows themselves are not 3× slower. First suspects: one slow rank or node, Slurm
  giving each task one CPU (`--ntasks-per-node 1` without `--cpus-per-task`) so the host side of Julia, XLA and
  the MPI progress engine share a core, or blocking in `MPI.Allreduce`/`bcast` (`total` in `calibrate.jl`).
  Time each evaluation per rank before the reduction to tell these apart.
- Sharding probe: never ran. LS6 3605904 (`Shard2`, `gpu-a100`, 3 h) has been pending on Priority since 11-22
  behind hundreds of jobs. `gpu-a100-dev` (≤ 2 h) is the faster route; Vista `gh` nodes have one GH200 each, so
  sharding there needs several processes (Reactant's multi-process distributed setup), which nothing here has
  tried.
- 1° single GPU (Vista GH200, 96 GB): the 90-day upwind3 frozen gradient runs (1161433: compile 12.8 min,
  gradient 437 s); 1° needs Δt ≤ 2 h with 100 barotropic substeps or Δt = 1 h (finding 14).

## Suggested steps

1. Instrument `calibrate.jl` with per-rank evaluation times (and the time inside the reductions); rerun 4 windows
   with `--cpus-per-task` set (e.g. 72 on `gh`) and compare. Fix until an iteration costs ~2.2 min.
2. Share the compile: all ranks trace and compile the same program. Try XLA's persistent compilation cache or
   compile on one rank first; measure.
3. 12 monthly windows of 1993 (check ECCO4 coverage first: Vista's depot cache has 1993-01 to 1994-03).
4. Run the sharding probe (LS6 `gpu-a100-dev`, `CUDA_VISIBLE_DEVICES=0,1`,
   `SCRIPT=adjoint/sharding_probe.jl CONFIG=adjoint/configs/sharding_probe_ecco4_nz20_8step.json sbatch -p
   gpu-a100-dev -t 02:00:00 -A ADCIRC adjoint/omip_adjoint.slurm`); cancel 3605904 if you resubmit. Then
   `Partition(3, 1)`, then 1°.
5. Combine: windows over ranks, each window sharded over its GPUs.

## Running

- Vista multi-window: `CONFIG=adjoint/configs/<c>.json sbatch -J <name> -N <windows> -A ADCIRC
  --mail-user=<the user's address, given at submit time, never committed> adjoint/calibrate_mpi.slurm` from
  `experiments/OMIPSimulations` in your worktree. Logs `adjoint/logs/<name>.o<jobid>`, GPU samples
  `adjoint/logs/gpu.<jobid>.csv`.
- `adjoint/omip_adjoint.slurm` runs any script with `SCRIPT=` and `CONFIG=` on one node (Vista `gh` or LS6).

## Out of scope

The grid-scale noise in the flux sensitivities (the other agent), the accuracy of the frozen κ gradient
(finding 22; open question to the user: a J(κ) scan on the 14-day window or tapering the slope derivative),
long-window adjoint growth.

## Log

### 2026-11-24

- Branch `ajs/omip-adjoint-parallel` from `caf55b76`; worktrees `NumericalEarth-parallel` on the Mac and on
  TACC (manifest and preferences copied from `NumericalEarth-oc117`).
- Reading 1166675: the truth (forward-only) windows take 15.5–16.1 s on every rank, as with one window, so the
  3× is in the gradient (checkpoint copies to the host, host allocation, GC), not the stepping. After each
  reduction rank 0 prints its post-GC memory line ~2 s later, ranks 1–3 11–23 s later (rank 3 always last),
  every iteration: the callback's `GC.gc()` looks 5–10× slower on ranks 1–3.
- Vista Slurm 23.11, `TaskPlugin = (null)`, `select/linear`: no CPU binding, so `--cpus-per-task` should not
  change a task's affinity. Checked by printing `Cpus_allowed_list` per rank instead of an A/B job.
- Reactant 0.2.309: the XLA autotune cache is on by default (depot scratchspace
  `xla_persistent_cache_0_0_445`, shared by all jobs); the kernel cache (`persistent_kernel_cache_enabled`) is
  off; `@compile serializable = true` exports StableHLO only, no reload of a compiled executable.
- Plan (deliverable 1): per-rank timing in `calibrate.jl` (evaluation wall time, its GC time and host
  allocation, the wait in the reduction, the callback's GC), plus host/threads/affinity per rank; 4 windows, 3
  iterations (`calibration_twin_ecco4_nz20_90day_4windows_timing.json`). Probe: sharded case first, so a 2 h
  dev job shows how far sharding gets even if the single-device half runs out of time.
- `94351713`: per-rank timing; probe sharded-first. Submitted Vista 1168099 `ParTime4` (4 nodes, `gh-dev`, 3
  iterations) and LS6 3612502 `ParShard2` (`gpu-a100-dev`, 2 h, `CUDA_VISIBLE_DEVICES=0,1`); cancelled 3605904.
- `92723ace`: config key `xla_kernel_cache` (XLA's `xla_gpu_kernel_cache_file`, with
  `xla_gpu_enable_llvm_module_compilation_parallelism` as Reactant's own preference sets it), passed through
  `adjoint_compile_options(; xla_debug_options)`. Vista 1168110 `ParKc1`: 1 window, 1 iteration, cold cache;
  a warm rerun follows. Both must reproduce 1166514's iteration 0 (J 7.138303e+04, |∂J/∂log κ| 7.147e+04).
- **1168099 `ParTime4` found it** (cancelled after iteration 0; nodes c642-[022,071,081-082]): every rank has 72
  CPUs (`Cpus_allowed_list 0-71`, so `--cpus-per-task` is moot), and every rank's evaluation takes one window's
  time: gradient 146.7, 149.3, 151.3, 154.1 s (GC 41–48 s, 110 GiB host allocation each), misfit 9–10 s,
  compile 12.6–12.7 min. The reductions are staggered: after the truth misfit rank 0 leaves `total` in 0.2 s,
  ranks 1 and 2 after 151 s, rank 3 after 304 s; ranks 1–2 return exactly when rank 0 next enters MPI (end of
  its gradient), rank 3 when ranks 1–2 do. Iteration 0's gradient reductions: 307, 169, 167, 23 s. So MPICH
  (`MPICH_jll` 5.0.3, ch4:ofi) delivers a collective's last messages only when the sender calls MPI again, and
  the ranks run their windows two to three deep in series: ~7 min per iteration.
- Fix: MPICH's progress thread (`MPIR_CVAR_ASYNC_PROGRESS=1` in `calibrate_mpi.slurm`,
  `MPI.Init(threadlevel = :multiple)` in `calibrate.jl`).
- 1168110 `ParKc1` failed at the first compile: this XLA's `DebugOptions` has no
  `xla_gpu_enable_llvm_module_compilation_parallelism` (Reactant 0.2.309's own `persistent_kernel_cache_enabled`
  preference sets it, so that preference fails the same way). The key now sets `xla_gpu_kernel_cache_file` only.
- Vista `gh-dev` runs one job per user at a time (QOS `qdevelopment`: MaxJobsPU 1, MaxSubmitPU 3), shared with
  the checkerboard agent; `gh` allows 20. Resubmitted on `gh`: 1168552 `ParTime4b` (4 windows with the progress
  thread, 3 iterations) and 1168553 `ParKc1b` (cold kernel cache), both at `69493d5f`.
- **1168552 `ParTime4b` (progress thread): 4-window iterations 2.2, 2.3, 2.2 min** (iteration 2 has two
  line-search misfits), against ~2.2 min for one window (1166514) and ~7 min before (1166675). Reduction waits
  0–8 s (the spread of the ranks' evaluation times); gradients 115–122 s per rank after the first (148–156 s), GC
  37–39 s of them; misfits 9.3–10 s. J and κ per iteration equal 1166675's to every printed digit (3.156515e+05,
  1.081492e+05, 4.572485e+04, 2.253984e+04). Build 201–203 s, compile 12.5–12.7 min on each rank.
- 1168553 `ParKc1b` (one window, cold kernel cache): compile 12.6 min (no change), cache file 76 MB;
  iterations 0 and 1 equal 1166514's (J 7.138303e+04, 2.433435e+04). Warm rerun: 1168843 `ParKc2`.
- Remaining cost in an iteration: the gradient's host side, ~38 s of GC and 110 GiB of host allocation per
  gradient (the checkpoint copies `map(Array, arrays)` are allocated anew at each store).
- 1168843 `ParKc2` (warm kernel cache, 76 MB): compile **12.5 min, no saving**, and the first compiled time step
  then fails with `CUDA_ERROR_ILLEGAL_INSTRUCTION` in `reactant_time_step_`: XLA reused a cached kernel that does
  not belong to this program. The kernel cache is unsafe here and buys nothing; the `xla_kernel_cache` key and its
  config are removed again, and the cache file deleted. The autotune cache was already on and warm.
- Deliverable 2, conclusion: no cheap way to share the compile. Reactant 0.2.309 has no executable
  (de)serialization in Julia (only the raw `PJRT_Executable_Serialize`/`DeserializeAndLoad` C bindings), and
  `serializable = true` exports StableHLO only. Within one job the ranks compile in parallel, so sharing would
  save node-hours (4 × 12.5 min of idle GPUs per 4-window job), not wall time.
- Self-review of the branch diff: cut the one-off affinity print and the callback's GC timing (answered); kept the per-rank evaluation and reduction times.

### 2026-11-25 01:10, status

- Deliverable 1 done (`a4b0aaf8`): 4-window iteration 2.2 min = one window's. Deliverable 2: no cheap way
  (above). Deliverable 3 blocked on the queue: LS6 3612502 `ParShard2` (`gpu-a100-dev`, at `94351713`'s probe,
  sharded first) still pending after 5 h, priority 2378 → 2465, 16 higher-priority jobs (27 requested node-hours)
  ahead on the partition's 3 working nodes. Left queued; its log will be
  `NumericalEarth-parallel/experiments/OMIPSimulations/adjoint/logs/ParShard2.o3612502`.
- Alternatives: LS6 `gpu-h100` (4 nodes, 1 drained, c318-002 known bad); or Vista multi-process sharding, which
  needs Reactant's IFRT runtime (`xla_runtime = "IFRT"` preference; Reactant warns that `Distributed.initialize`
  under the default PJRT "will never be properly supported"), i.e. a separate environment, since the preference
  changes the array types of every job in this one.
- Compute used: ~5 GPU-node-hours (1168099 2.2, 1168552 1.9, 1168110/1168553/1168843 0.8).

### 2026-11-25, sharded build and checkpoint slots

- LS6 3612502 `ParShard2` ran (52 min): the sharded build failed in `ImmersedBoundaryGrid` (`set!` of the bottom
  height → `interior(u) .= interior(v)` → `TypeError: expected ConcretePJRTArray{Float64, 3, 1}, got
  ConcretePJRTArray{Float64, 3, 2}`); single device: build 185 s, compile primal 499 s, gradient compile 1590 s,
  gradient 18.9 s, J 24.1308164238 Sv (= Vista 1153870), max|∂J/∂τx| 4.494038. Saved file has J only (no
  gradients); kept as `results/reference_sharding_probe_3612502.jld2`.
- Reproduced on the Mac (Reactant 0.2.309, CPU, two virtual devices, `scratchpad` scripts): with PJRT arrays,
  (a) `view(a, ...) .= view(b, ...)` and `copyto!` between views of sharded arrays raise the same `TypeError`
  and leave `a` donated; (b) `copyto!(a, host)` into a whole sharded array writes all elements into the first
  shard's buffer (`copyto!(dest::ConcretePJRTArray, doffs, src::Array, ...)` uses `dest.data[1]`, no
  single-shard check), i.e. a buffer overrun: the readback crashes Julia (illegal instruction); (c) `a .= host`
  hangs; (d) `copyto!(view(a, ...), host)` is correct. (b) is what `regularize_fold_metrics!` and
  Oceananigans' `restore_prognostic_state!(::Field, from)` (`copyto!(parent(restored), from.data)`) do, so a
  sharded PJRT build and every checkpoint restore would corrupt memory even past the reported failure.
  `ConcreteIFRTArray` has no such fast path (host data become a sharded IFRT array, then a compiled copy).
- Checkpoint slots (user-approved): `slots` kept across evaluations, a store copies into the slot's host arrays
  (`memory_snapshot!`) instead of allocating new ones. References on current code copied aside on Vista:
  `results/reference_1window_1168553.jld2` (1 window, iterations 0–1), `reference_4windows_1168552.jld2`.
