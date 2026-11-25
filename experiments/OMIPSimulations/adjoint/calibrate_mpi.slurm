#!/bin/bash
#----------------------------------------------------
# Calibration across windows (adjoint/calibrate.jl) with one MPI rank, and one GPU, per Vista node.
#
# Usage, from experiments/OMIPSimulations, with as many nodes as the config has windows:
#   CONFIG=adjoint/configs/<cfg>.json sbatch -N <windows> -A ADCIRC --mail-user=<address> adjoint/calibrate_mpi.slurm
# SCRIPT= runs another script on every node (default adjoint/calibrate.jl), PROJECT= another environment (default
# adjoint; adjoint/ifrt for Reactant's IFRT runtime).
# The ranks are started by srun with PMI-2: MPI.jl's MPICH reduces across nodes inside the container that way, not
# with PMIx (adjoint/mpi_probe, job 1161912). Output: adjoint/logs/<name>.o<jobid>, results in adjoint/results/.
#----------------------------------------------------
#SBATCH -J Calibrate
#SBATCH -o adjoint/logs/%x.o%j
#SBATCH -e adjoint/logs/%x.e%j
#SBATCH -p gh
#SBATCH --ntasks-per-node 1
#SBATCH -t 03:00:00
#SBATCH --mail-type=ALL

cd "$SLURM_SUBMIT_DIR" || exit 1
[ -f adjoint/Project.toml ] || { echo "Submit from experiments/OMIPSimulations (no adjoint/Project.toml in $PWD)" >&2; exit 1; }
mkdir -p adjoint/logs adjoint/results
source "$HOME/.bashrc"          # JULIA_DEPOT_PATH; before `set -u`
module load tacc-apptainer 2>/dev/null || true
command -v apptainer >/dev/null || { echo "apptainer not found (module load tacc-apptainer?)" >&2; exit 1; }
set -uo pipefail
export JULIA_NUM_PRECOMPILE_TASKS=${SLURM_CPUS_PER_TASK:-16}
# MPICH moves messages only inside MPI calls: without a progress thread, a rank that returns from a reduction
# and computes for minutes holds back the messages the other ranks still wait for (job 1168099).
export MPIR_CVAR_ASYNC_PROGRESS=1

CONFIG=${CONFIG:?set CONFIG to a calibration config}
SCRIPT=${SCRIPT:-adjoint/calibrate.jl}
PROJECT=${PROJECT:-adjoint}
SIF=${SIF:-$WORK/containers/julia_1.12_${TACC_SYSTEM}.sif}
[ -f "$SIF" ] || { echo "Missing container $SIF" >&2; exit 1; }
[ -f "$CONFIG" ] || { echo "Config not found: $CONFIG" >&2; exit 1; }
echo "script: $SCRIPT  config: $CONFIG  project: $PROJECT  container: $SIF  nodes: $SLURM_JOB_NODELIST"; date
cp adjoint/Manifest.toml "adjoint/logs/Manifest.$SLURM_JOB_ID.toml" 2>/dev/null || true

BIND=""
for d in /work /work2 /scratch; do [ -d "$d" ] && BIND="${BIND:+$BIND,}$d"; done
RUN="apptainer exec --nv ${BIND:+--bind $BIND} $SIF julia --project=$PROJECT"
[ -f adjoint/LocalPreferences.toml ] || printf '[Reactant_jll]\ngpu = "cuda"\ngpu_version = "12.9"\n' > adjoint/LocalPreferences.toml

# Once, on the first node: the ranks share the depot.
echo "=== environment ($(date +%T))"
$RUN -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()' > "adjoint/logs/env.$SLURM_JOB_ID.log" 2>&1 ||
    { echo "Environment failed; see adjoint/logs/env.$SLURM_JOB_ID.log" >&2; exit 1; }
echo "=== calibration ($(date +%T))"
nvidia-smi -i 0 --query-gpu=timestamp,utilization.gpu,memory.used --format=csv -l 300 > "adjoint/logs/gpu.$SLURM_JOB_ID.csv" 2>&1 &
SMI_PID=$!
srun --mpi=pmi2 $RUN -O0 "$SCRIPT" "$CONFIG"
STATUS=$?
kill $SMI_PID 2>/dev/null
date
exit $STATUS
