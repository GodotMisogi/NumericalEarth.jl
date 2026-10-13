#!/bin/bash
#----------------------------------------------------
# Differentiable OMIP ocean test on one Vista GH200 (NumericalEarth.jl/experiments/OMIPSimulations).
#
# Usage, from experiments/OMIPSimulations:
#   sbatch adjoint/omip_adjoint_vista.slurm                                   # default config, gh, 8 h
#   CONFIG=adjoint/configs/<cfg>.json sbatch -p gh-dev -t 02:00:00 adjoint/omip_adjoint_vista.slurm
# The job instantiates and precompiles the adjoint environment inside the container first (idempotent),
# then runs the test. Datasets come from NumericalEarth's cache in the depot (`adjoint/prepare_data.jl`
# fills it; compute nodes have internet, login nodes cannot run the container).
# Output: adjoint/logs/OMIPAdjoint.o<jobid>, adjoint/logs/gpu.<jobid>.csv (nvidia-smi every 5 min),
# results in adjoint/results/.
#----------------------------------------------------
#SBATCH -J OMIPAdjoint
#SBATCH -o adjoint/logs/%x.o%j
#SBATCH -e adjoint/logs/%x.e%j
#SBATCH -p gh
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 08:00:00
#SBATCH -A Transfer-of-ECCO-glo

cd "$SLURM_SUBMIT_DIR" || exit 1
[ -f adjoint/Project.toml ] || { echo "Submit from experiments/OMIPSimulations (no adjoint/Project.toml in $PWD)" >&2; exit 1; }
mkdir -p adjoint/logs adjoint/results
source "$HOME/.bashrc"          # JULIA_DEPOT_PATH; before `set -u`
module load tacc-apptainer 2>/dev/null || true
command -v apptainer >/dev/null || { echo "apptainer not found (module load tacc-apptainer?)" >&2; exit 1; }
set -uo pipefail
export JULIA_NUM_PRECOMPILE_TASKS=${SLURM_CPUS_PER_TASK:-16}

CONFIG=${CONFIG:-adjoint/configs/twodegree_nz20_1step.json}
SIF=${SIF:-$WORK/.julia/dev/ReactOcean/vi_julia.sif}
[ -f "$SIF" ] || { echo "Missing container $SIF" >&2; exit 1; }
[ -f "$CONFIG" ] || { echo "Config not found: $CONFIG" >&2; exit 1; }
echo "config: $CONFIG  container: $SIF"; date; hostname
nvidia-smi --query-gpu=name,memory.total --format=csv
cp adjoint/Manifest.toml "adjoint/logs/Manifest.$SLURM_JOB_ID.toml" 2>/dev/null || true

BIND=""
for d in /work /work2 /scratch; do [ -d "$d" ] && BIND="${BIND:+$BIND,}$d"; done
RUN="apptainer exec --nv ${BIND:+--bind $BIND} $SIF julia --project=adjoint"

# The CUDA 13.1 Reactant_jll build fails to load on the GH200 nodes (undefined CUDA symbol); select 12.9.
[ -f adjoint/LocalPreferences.toml ] || printf '[Reactant_jll]\ngpu = "cuda"\ngpu_version = "12.9"\n' > adjoint/LocalPreferences.toml

echo "=== environment ($(date +%T))"
$RUN -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()' > "adjoint/logs/env.$SLURM_JOB_ID.log" 2>&1 ||
    { echo "Environment failed; see adjoint/logs/env.$SLURM_JOB_ID.log" >&2; exit 1; }
echo "=== test ($(date +%T))"

# GPU utilization every 5 min: tells a device-bound evaluation from a host-side hang.
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv -l 300 > "adjoint/logs/gpu.$SLURM_JOB_ID.csv" 2>&1 &
SMI_PID=$!

$RUN -O0 adjoint/test_omip_adjoint.jl "$CONFIG"
STATUS=$?

# Reactant writes the MLIR module of a failed compile to the node's /tmp; keep it with the logs.
if [ $STATUS -ne 0 ]; then
    for d in /tmp/reactant_*; do [ -d "$d" ] && cp -r "$d" "adjoint/logs/reactant_dump.$SLURM_JOB_ID.$(basename "$d")"; done
fi

kill $SMI_PID 2>/dev/null
date
exit $STATUS
