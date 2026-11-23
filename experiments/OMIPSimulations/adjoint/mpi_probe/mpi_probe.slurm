#!/bin/bash
#----------------------------------------------------
# MPI.jl inside the Apptainer container across Vista nodes (calibration phase 2 prerequisite). One task per node.
#   sbatch -N 2 -A ADCIRC --mail-user=<address> adjoint/mpi_probe/mpi_probe.slurm   (from experiments/OMIPSimulations)
#----------------------------------------------------
#SBATCH -J MPIprobe
#SBATCH -o adjoint/logs/%x.o%j
#SBATCH -e adjoint/logs/%x.e%j
#SBATCH -p gh
#SBATCH --ntasks-per-node 1
#SBATCH -t 00:20:00
#SBATCH --mail-type=ALL

cd "$SLURM_SUBMIT_DIR" || exit 1
source "$HOME/.bashrc"
module load tacc-apptainer 2>/dev/null || true
set -uo pipefail
SIF=${SIF:-$WORK/containers/julia_1.12_${TACC_SYSTEM}.sif}
BIND=""
for d in /work /work2 /scratch; do [ -d "$d" ] && BIND="${BIND:+$BIND,}$d"; done
RUN="apptainer exec --nv ${BIND:+--bind $BIND} $SIF julia --project=adjoint/mpi_probe"
echo "nodes: $SLURM_JOB_NODELIST"; date
$RUN -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()' 2>&1 | tail -3
# MPI.jl's default MPICH_jll binary; Slurm starts one container per task and hands it a PMI.
for mpi in pmi2 pmix; do
    echo "=== srun --mpi=$mpi ($(date +%T))"
    srun --mpi=$mpi $RUN adjoint/mpi_probe/mpi_probe.jl 2>&1 | tail -8
done
date
