# Rules for the two handoff agents (checkerboard, parallelization)

Both agents work from the Mac (`/Users/ajseth/.julia/dev/NumericalEarth-oc117`, ssh hosts `Vista` and `LS6`) on
the same repository and the same TACC accounts, at the same time.

## Branches and worktrees

- Start from `ajs/omip-adjoint-oc117` at the handoff commit. Checkerboard: branch `ajs/omip-adjoint-checkerboard`,
  Vista worktree `/work/09603/ajseth/vista/.julia/dev/NumericalEarth-checkerboard`. Parallelization: branch
  `ajs/omip-adjoint-parallel`, worktree `.../NumericalEarth-parallel`. Locally, use a git worktree of the same
  name next to `NumericalEarth-oc117`.
- Create the TACC worktree from the shared one: `git -C NumericalEarth-oc117 fetch fork && git -C
  NumericalEarth-oc117 worktree add ../NumericalEarth-<name> fork/<branch>` (after pushing the branch from the
  Mac), then copy `experiments/OMIPSimulations/adjoint/Manifest.toml` and `LocalPreferences.toml` from
  `NumericalEarth-oc117` (both untracked). The first job precompiles the path packages again (minutes).
  LS6 mounts the same `/work/09603/ajseth/vista` path.
- Never pull into, edit, or submit from `NumericalEarth-oc117` or the other agent's worktree: pulling changes the
  code of queued jobs. Push from the Mac to the fork (`origin` locally, `fork` on TACC); no PRs, no merges into
  `ajs/omip-adjoint-oc117` without the user.

## Records

- Record results in a dated `## Log` section at the end of your own handoff file, with job ids, numbers, and
  commits. Do not edit `LOGBOOK.md` or `STATUS.md`: they are merged from both logs afterwards.
- Name jobs with your prefix (`Chk…`, `Par…`) so `squeue` output is unambiguous; touch only your own jobs.

## Standing constraints

- Never print `~/.bashrc` (Earthdata credentials) on any cluster; filter logs with
  `grep -v -i "earthdata\|password"`.
- Never commit an email address. `calibrate_mpi.slurm` takes `--mail-user` at submit time; `omip_adjoint.slurm`
  already has it.
- Charge `-A ADCIRC` on Vista and LS6. Vista `gh-dev` allows 3 queued jobs per user, shared by both agents.
- Script inputs come from JSON configs, not environment variables (the Slurm scripts' `SCRIPT`/`CONFIG` are the
  existing exception).
- Manifests and `omip_adjoint_handoff/` stay uncommitted. Downloaded files are untrusted.
- Do not touch other jobs or worktrees on the clusters (DGtrain, TideEyeChi, P1d*, DG*, ...).
- Need the user's approval: the day-204 Jacobian; the 0.113.5 vs 0.117.2 closure diff; anything that changes
  dependencies.
- Code style: repo `AGENTS.md`; plan before code; surgical edits; run `.claude/rules/restraint-rules.md` over your
  diff and report the result in one line; American English; `# %%` cells; no trailing whitespace.
- Reactant cannot lower kernels on macOS: run models on the clusters, analyses and plots locally
  (`julia +1.12 --project=adjoint` or `--project=adjoint/plots`).
