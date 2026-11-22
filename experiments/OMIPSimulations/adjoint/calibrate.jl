# %%
# ==============================================================================
# Twin calibration of the GM/Redi diffusivities of the ECCO4 prescribed-flux ocean: synthetic observations of the
# window-mean temperature and salinity are made with the `truth` diffusivities, then BFGS recovers them from
# `initial` with the adjoint gradient of the misfit (`build_eddy_calibration`).
#
#   CONFIG=adjoint/configs/<config>.json SCRIPT=adjoint/calibrate.jl sbatch ... adjoint/omip_adjoint.slurm
#
# Config keys: n_months, Nx, Ny (default 180, 60), Nz, steps, dt, substeps, backend, roll_barotropic_substeps,
# checkpoints, average_days, output; optional frozen_mixing, frozen_buoyancy_gradients, advection, start_date.
# truth, initial: [κ_skew, κ_symmetric] (m² s⁻¹). observed_levels (default Nz), observation_stride (default 1),
# sigma_T (K), sigma_S (g kg⁻¹): the observations. stages: "check" (central differences of the misfit in log κ,
# step check_h, against the adjoint) and/or "calibrate" (iterations, initial_step, gtol).
# ==============================================================================

using OMIPAdjoint
using Oceananigans
using Oceananigans.Architectures: ReactantState
using Reactant
using JSON
using JLD2
using Printf
using Dates: now, format, DateTime

ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"   # the WOA download goes through DataDeps, which otherwise prompts

config_path = length(ARGS) >= 1 ? ARGS[1] : error("usage: calibrate.jl <config.json>")
P = JSON.parsefile(config_path)
stages = Symbol.(strip.(split(P["stages"], ",")))
n_steps, Δt = P["steps"], Float64(P["dt"])
average_steps = round(Int, P["average_days"] * 86400 / Δt)
stamp(msg...) = (println("[", format(now(), "HH:MM:SS"), "] ", msg...); flush(stdout))

results = Dict{String, Any}("config" => P)
output_path = joinpath(@__DIR__, P["output"])
function save_results()
    mkpath(dirname(output_path))
    jldopen(output_path, "w") do file
        for (key, value) in results
            file[key] = value
        end
    end
    return nothing
end

# %%
##### Model and compiled misfit

:NVPTX in Reactant.LLVM.backends() ||
    error("Reactant lowers Oceananigans kernels through NVPTX; this Julia's LLVM lacks it (macOS). Run on the GPU cluster.")
Reactant.set_default_backend(P["backend"])
roll_barotropic_substeps!(P["roll_barotropic_substeps"])
freeze_mixing!(get(P, "frozen_mixing", false))
frozen_gradients = freeze_buoyancy_gradients!(get(P, "frozen_buoyancy_gradients", false))

date_kwargs = haskey(P, "start_date") ? (; start_date = DateTime(P["start_date"])) : (;)
t0 = time()
setup = ecco_adjoint_setup(ReactantState(); Nx = get(P, "Nx", 180), Ny = get(P, "Ny", 60), Nz = P["Nz"], Δt,
                           substeps = P["substeps"], n_months = P["n_months"],
                           advection = Symbol(get(P, "advection", "weno")),
                           materialize_buoyancy_gradients = frozen_gradients, eddy_diffusivity_fields = true,
                           date_kwargs...)
model = setup.model
state = snapshot_state(model)
observed = observed_cells(setup.cpu_grid; levels = get(P, "observed_levels", P["Nz"]),
                          stride = get(P, "observation_stride", 1))
stamp(@sprintf("model built in %.1f s; %d observed cells", time() - t0, count(observed)))

# Placeholder observations: replaced by the truth run's means below, before any misfit is evaluated.
Nx, Ny, Nz = size(observed)
observations = TracerMisfit(observed, zeros(Nx, Ny, Nz), zeros(Nx, Ny, Nz), Float64(P["sigma_T"]), Float64(P["sigma_S"]))
t0 = time()
calibration = build_eddy_calibration(model, state, observations; Δt, n_steps, average_steps,
                                     checkpoints = P["checkpoints"])
results["compile_seconds"] = time() - t0
stamp(@sprintf("compiled in %.1f min", results["compile_seconds"] / 60))

# %%
##### Synthetic observations from the truth

x_truth = log.(Float64.(P["truth"]))
t0 = time()
T°, S° = calibration.means!(x_truth)
observations.T .= T°
observations.S .= S°
results["forward_seconds"] = time() - t0
stamp(@sprintf("truth window in %.1f s: observed mean T %.4f, S %.4f", results["forward_seconds"],
               sum(T°[observed]) / count(observed), sum(S°[observed]) / count(observed)))
x0 = log.(Float64.(P["initial"]))

# %%
##### Gradient check: central differences of the misfit in log κ

if :check in stages
    h = P["check_h"]
    J0, g0 = calibration.gradient!(x0)
    fd = map(1:2) do i
        e = [j == i for j in 1:2] .* h
        (calibration.misfit!(x0 .+ e) - calibration.misfit!(x0 .- e)) / 2h
    end
    for (i, name) in enumerate(("κ_skew", "κ_symmetric"))
        stamp(@sprintf("∂J/∂log %s: adjoint %+.6e, central difference %+.6e (h = %.3f), ratio %.4f", name, g0[i], fd[i],
                       h, g0[i] / fd[i]))
    end
    results["check"] = (; x0, J0, adjoint = g0, central_difference = fd, h)
    save_results()
end

# %%
##### Calibration

if :calibrate in stages
    J_truth = calibration.misfit!(x_truth)
    stamp(@sprintf("misfit at the truth %.3e (must be 0)", J_truth))
    records = NamedTuple[]
    callback(r) = begin
        push!(records, r)
        κ = exp.(r.x)
        stamp(@sprintf("iteration %2d: κ_skew %7.1f, κ_symmetric %7.1f m² s⁻¹, J %.6e, |∂J/∂log κ| %.3e, step %.3f, %d evaluations",
                       r.iteration, κ[1], κ[2], r.f, sqrt(sum(abs2, r.g)), r.step, r.evaluations))
        results["records"] = copy(records)
        save_results()
    end
    bfgs_minimize(calibration.gradient!, calibration.misfit!, x0; iterations = P["iterations"],
                  initial_step = P["initial_step"], gtol = get(P, "gtol", 0.0), callback)
    κ = exp.(records[end].x)
    stamp(@sprintf("recovered κ_skew %.1f (truth %.1f), κ_symmetric %.1f (truth %.1f)", κ[1], P["truth"][1], κ[2],
                   P["truth"][2]))
end

stamp("saved ", output_path)
