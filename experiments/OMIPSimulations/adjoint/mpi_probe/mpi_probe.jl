# %%
# ==============================================================================
# Can MPI.jl, inside the Apptainer container, reduce across Slurm tasks on several nodes? Each rank reports its
# host and GPU, then the ranks sum their rank numbers and broadcast a vector from rank 0. The calibration across
# windows needs exactly these two collectives.
#
#   sbatch -N 2 adjoint/mpi_probe/mpi_probe.slurm
# ==============================================================================

using MPI

MPI.Init()
comm = MPI.COMM_WORLD
rank, nranks = MPI.Comm_rank(comm), MPI.Comm_size(comm)
println("rank $rank of $nranks on $(gethostname()), CUDA_VISIBLE_DEVICES = $(get(ENV, "CUDA_VISIBLE_DEVICES", "unset")), ",
        "MPI library: $(MPI.identify_implementation())")
total = MPI.Allreduce(rank, +, comm)
x = rank == 0 ? [1.5, -2.5] : zeros(2)
MPI.Bcast!(x, 0, comm)
ok = total == nranks * (nranks - 1) ÷ 2 && x == [1.5, -2.5]
rank == 0 && println("Allreduce sum $total (expected $(nranks * (nranks - 1) ÷ 2)), Bcast $x: ", ok ? "PASS" : "FAIL")
MPI.Barrier(comm)
MPI.Finalize()
