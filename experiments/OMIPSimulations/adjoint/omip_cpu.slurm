#!/bin/bash
#----------------------------------------------------
# Post-processing on one TACC CPU node (Lonestar6 `development`/`normal`): runs SCRIPT with CONFIG in the
# adjoint environment, e.g. reduce_gradients.jl next to the results files.
#
# Usage, from experiments/OMIPSimulations:
#   SCRIPT=adjoint/reduce_gradients.jl CONFIG=adjoint/configs/reduce_twodegree_nz20.json sbatch adjoint/omip_cpu.slurm
# Unlike omip_adjoint.slurm this does not precompile the whole environment: Reactant's CUDA extension fails to
# initialize without a GPU, and the post-processing scripts load only packages the GPU jobs already precompiled.
#----------------------------------------------------
#SBATCH -J OMIPPost
#SBATCH -o adjoint/logs/%x.o%j
#SBATCH -e adjoint/logs/%x.e%j
#SBATCH -p development
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 00:30:00
#SBATCH -A ADCIRC
#SBATCH --mail-user=arjitseth@gmail.com
#SBATCH --mail-type=ALL

cd "$SLURM_SUBMIT_DIR" || exit 1
[ -f adjoint/Project.toml ] || { echo "Submit from experiments/OMIPSimulations (no adjoint/Project.toml in $PWD)" >&2; exit 1; }
mkdir -p adjoint/logs adjoint/results
source "$HOME/.bashrc"          # JULIA_DEPOT_PATH; before `set -u`
module load tacc-apptainer 2>/dev/null || true
command -v apptainer >/dev/null || { echo "apptainer not found (module load tacc-apptainer?)" >&2; exit 1; }
set -uo pipefail

SCRIPT=${SCRIPT:?set SCRIPT=adjoint/<script>.jl}
CONFIG=${CONFIG:?set CONFIG=adjoint/configs/<config>.json}
SIF=${SIF:-$WORK/containers/julia_1.12_${TACC_SYSTEM}.sif}
[ -f "$SIF" ] || { echo "Missing container $SIF" >&2; exit 1; }
[ -f "$CONFIG" ] || { echo "Config not found: $CONFIG" >&2; exit 1; }
echo "script: $SCRIPT  config: $CONFIG  container: $SIF"; date; hostname

BIND=""
for d in /work /work2 /scratch; do [ -d "$d" ] && BIND="${BIND:+$BIND,}$d"; done
apptainer exec ${BIND:+--bind $BIND} "$SIF" julia --project=adjoint "$SCRIPT" "$CONFIG"
STATUS=$?
date
exit $STATUS
