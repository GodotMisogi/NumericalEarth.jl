# %%
# ==============================================================================
# Forward health check of an OMIP adjoint window: step the compiled model forward (the same Reactant
# `time_step!` the checkpointed gradient runs) and record, every `snapshot_days`, the quantities that can make
# the linearized model unstable while the forward model stays bounded.
#
#   SCRIPT=adjoint/forward_health.jl CONFIG=adjoint/configs/<config>.json sbatch adjoint/omip_adjoint.slurm
#
# Config keys: those of test_omip_adjoint.jl that set up the model (forcing, n_months or n_atmosphere_slices,
# Nx, Ny, Nz, dt, substeps, eddy_diffusivity, advection, backend, roll_barotropic_substeps), plus
#   architecture   optional "reactant" (default) or "cpu" (the plain CPU model, to check the diagnostics locally)
#   steps          number of steps
#   snapshot_days  interval between diagnostics
#   output         the diagnostics file (relative to adjoint/)
#
# Per snapshot: the objective; the largest vertical and horizontal advective CFL numbers (|w| Δt / Δz,
# |u| Δt / Δx, |v| Δt / Δy) and their cells; grid-scale noise of surface T, S, u, v (the rms of the field minus the
# mean of its four neighbors, over cells whose neighbors are all wet) and its worst cell; the count of statically
# unstable interfaces and the most negative N² (TEOS-10 densities of both cells at the interface depth); the
# largest CATKE tracer diffusivity; and, for ECCO4 forcing, the rms SST difference from ECCO4's monthly mean. Each
# snapshot also stores 2-D maps (column maxima of the CFL numbers, surface noise, surface κ) for plotting.
# ==============================================================================

using OMIPAdjoint
using OMIPAdjoint: ocean_model, refresh_diagnostics!, restore_state!, host_number, ecco_to_teos10!
using Oceananigans
using Oceananigans.Architectures: ReactantState
using Oceananigans.Grids: inactive_node, znode, λnode, φnode
using Oceananigans.Operators: Δxᶠᶜᶜ, Δyᶜᶠᶜ, Δzᶜᶜᶠ
using Oceananigans.TimeSteppers: time_step!
using NumericalEarth.DataWrangling: Metadatum
using NumericalEarth.DataWrangling.ECCO: ECCO4Monthly
using SeawaterPolynomials: ρ
using SeawaterPolynomials.TEOS10: TEOS10EquationOfState
using Reactant
using JSON
using JLD2
using Printf
using Dates: DateTime, Second, format, now, year, month

ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

config_path = length(ARGS) >= 1 ? ARGS[1] : error("usage: forward_health.jl <config.json>")
P = JSON.parsefile(config_path)
Nx, Ny, Nz = get(P, "Nx", 180), get(P, "Ny", 60), P["Nz"]
n_steps, Δt, substeps = P["steps"], Float64(P["dt"]), P["substeps"]
κ_eddy = get(P, "eddy_diffusivity", 800)
advection = Symbol(get(P, "advection", "weno"))
forcing = get(P, "forcing", "jra55")
start_date = DateTime(1993, 1, 1)

stamp(msg...) = (println("[", format(now(), "HH:MM:SS"), "] ", msg...); flush(stdout))

# %%
##### Model on ReactantState and the compiled step

# `architecture = "cpu"` steps the plain CPU model instead (for checking the diagnostics locally).
on_reactant = get(P, "architecture", "reactant") == "reactant"
arch = on_reactant ? ReactantState() : CPU()
if on_reactant
    Reactant.set_default_backend(P["backend"])
    roll_barotropic_substeps!(get(P, "roll_barotropic_substeps", true))
end
setup = forcing == "ecco4" ?
    ecco_adjoint_setup(arch; Nx, Ny, Nz, Δt, substeps, n_months = P["n_months"],
                       κ_skew = κ_eddy, κ_symmetric = κ_eddy, advection, start_date) :
    omip_adjoint_setup(arch; Nx, Ny, Nz, Δt, substeps, n_atmosphere_slices = P["n_atmosphere_slices"],
                       κ_skew = κ_eddy, κ_symmetric = κ_eddy, advection)
model, cpu_grid = setup.model, setup.cpu_grid
obj = OverturningStrength(cpu_grid)
if on_reactant
    options = adjoint_compile_options()
    t0 = time()
    refresh = @compile compile_options = options refresh_diagnostics!(model)
    step = @compile compile_options = options time_step!(model, Δt)
    stamp(@sprintf("compiled refresh and step in %.1f min", (time() - t0) / 60))
else
    refresh, step = refresh_diagnostics!, time_step!
end

# %%
##### Host-side geometry

wet = [!inactive_node(i, j, k, cpu_grid, Center(), Center(), Center()) for i in 1:Nx, j in 1:Ny, k in 1:Nz]
Δzᶠ = [Δzᶜᶜᶠ(1, 1, k, cpu_grid) for k in 1:Nz+1]      # spacing around w-points
zᶠ = [znode(1, 1, k, cpu_grid, Center(), Center(), Face()) for k in 1:Nz+1]
Δx = [Δxᶠᶜᶜ(i, j, Nz, cpu_grid) for i in 1:Nx, j in 1:Ny]
Δy = [Δyᶜᶠᶜ(i, j, Nz, cpu_grid) for i in 1:Nx, j in 1:Ny]
λ = [λnode(i, j, Nz, cpu_grid, Center(), Center(), Center()) for i in 1:Nx, j in 1:Ny]
φ = [φnode(i, j, Nz, cpu_grid, Center(), Center(), Center()) for i in 1:Nx, j in 1:Ny]
eos = TEOS10EquationOfState()

# Interior of a (device) field on the host.
function host_interior(f)
    a = Array(parent(f))
    Hx, Hy, Hz = f.grid.Hx, f.grid.Hy, f.grid.Hz
    nx, ny, nz = size(f)
    return a[Hx .+ (1:nx), Hy .+ (1:ny), Hz .+ (1:nz)]
end

where(i, j) = @sprintf("%.0fE %.0fN", mod(λ[i, j] + 180, 360) - 180, φ[i, j])

# Field minus the mean of its four horizontal neighbors (periodic in x), where all five cells are wet.
function grid_scale_residual(f, mask)
    r = fill(NaN, Nx, Ny)
    for j in 2:Ny-1, i in 1:Nx
        ip, im = mod1(i + 1, Nx), mod1(i - 1, Nx)
        (mask[i, j] && mask[ip, j] && mask[im, j] && mask[i, j+1] && mask[i, j-1]) || continue
        r[i, j] = f[i, j] - (f[ip, j] + f[im, j] + f[i, j+1] + f[i, j-1]) / 4
    end
    return r
end

rms(x) = (y = filter(isfinite, x); isempty(y) ? NaN : sqrt(sum(abs2, y) / length(y)))

function argmax_finite(a)
    best, idx = -Inf, CartesianIndex(1, 1)
    for I in CartesianIndices(a)
        v = a[I]
        isfinite(v) && v > best && ((best, idx) = (v, I))
    end
    return best, idx
end

# ECCO4 monthly mean SST (Conservative Temperature), cached per month.
ecco_sst_cache = Dict{DateTime, Matrix{Float64}}()
function ecco_sst(date)
    first_day = DateTime(year(date), month(date), 1)
    get!(ecco_sst_cache, first_day) do
        T = CenterField(cpu_grid); S = CenterField(cpu_grid)
        set!(T, Metadatum(:temperature; dataset = ECCO4Monthly(), date = first_day))
        set!(S, Metadatum(:salinity; dataset = ECCO4Monthly(), date = first_day))
        ecco_to_teos10!(T, S)
        Array(interior(T, :, :, Nz))
    end
end

# CATKE's tracer diffusivity among the model's closure fields (one entry per closure).
catke_κc(om) = only(c.κc for c in om.closure_fields if hasproperty(c, :κc))

# %%
##### One snapshot

function snapshot(n)
    om = ocean_model(model)
    u, v, w = host_interior(om.velocities.u), host_interior(om.velocities.v), host_interior(om.velocities.w)
    T, S = host_interior(om.tracers.T), host_interior(om.tracers.S)
    κ = host_interior(catke_κc(om))
    J = host_number(objective_host(obj, model))

    cfl_w = [maximum(abs(w[i, j, k]) * Δt / Δzᶠ[k] for k in 2:Nz; init = 0.0) for i in 1:Nx, j in 1:Ny]
    cfl_h = [max(abs(u[i, j, Nz]) * Δt / Δx[i, j], abs(v[i, j, Nz]) * Δt / Δy[i, j]) for i in 1:Nx, j in 1:Ny]

    # Interfaces k | k+1 (k+1 above): stable when the lower cell is denser at the interface depth.
    unstable, N²min, N²at = 0, Inf, (1, 1, 1)
    for k in 1:Nz-1, j in 1:Ny, i in 1:Nx
        (wet[i, j, k] && wet[i, j, k+1]) || continue
        z = zᶠ[k+1]
        Δρ = ρ(T[i, j, k], S[i, j, k], z, eos) - ρ(T[i, j, k+1], S[i, j, k+1], z, eos)
        N² = 9.81 * Δρ / 1026 / Δzᶠ[k+1]
        N² < 0 && (unstable += 1)
        N² < N²min && ((N²min, N²at) = (N², (i, j, k)))
    end

    surface = wet[:, :, Nz]
    noise = Dict(name => grid_scale_residual(f[:, :, Nz], surface)
                 for (name, f) in (("T", T), ("S", S), ("u", u), ("v", v)))
    sst_rms = NaN
    if forcing == "ecco4"
        date = start_date + Second(round(Int, n * Δt))
        d = T[:, :, Nz] .- ecco_sst(date)
        sst_rms = rms(ifelse.(surface, d, NaN))
    end

    cw, Iw = argmax_finite(cfl_w); ch, Ih = argmax_finite(cfl_h); κmax, Iκ = argmax_finite(κ[:, :, Nz])
    nT, InT = argmax_finite(abs.(noise["T"]))
    @printf("day %6.1f  J %+8.3f Sv | CFL w %.3f at %s, h %.3f at %s | noise rms T %.2e S %.2e u %.2e v %.2e, max |T| %.2e at %s | unstable %d, min N² %.2e at %s k=%d | κc max %.2e at %s | SST rms %.2f\n",
            n * Δt / 86400, J, cw, where(Iw[1], Iw[2]), ch, where(Ih[1], Ih[2]),
            rms(noise["T"]), rms(noise["S"]), rms(noise["u"]), rms(noise["v"]), nT, where(InT[1], InT[2]),
            unstable, N²min, where(N²at[1], N²at[2]), N²at[3], κmax, where(Iκ[1], Iκ[2]), sst_rms)
    flush(stdout)

    return (; day = n * Δt / 86400, J, cfl_w_max = cw, cfl_h_max = ch,
              noise_rms = Dict(k => rms(v) for (k, v) in noise), unstable, N²min, κc_max = κmax, sst_rms,
              cfl_w = Float32.(cfl_w), cfl_h = Float32.(cfl_h), noise_T = Float32.(noise["T"]),
              noise_u = Float32.(noise["u"]), κc_surface = Float32.(κ[:, :, Nz]))
end

# %%
##### Run

state = snapshot_state(model)
restore_state!(model, state)
refresh(model)
every = round(Int, P["snapshot_days"] * 86400 / Δt)
records = [snapshot(0)]
t0 = time()
for n in 1:n_steps
    step(model, Δt)
    n % every == 0 && push!(records, snapshot(n))
end
stamp(@sprintf("%d steps in %.1f min", n_steps, (time() - t0) / 60))

output_path = joinpath(@__DIR__, P["output"])
mkpath(dirname(output_path))
jldsave(output_path; config = P, records, λ, φ, surface_wet = wet[:, :, Nz])
stamp("saved ", output_path)
