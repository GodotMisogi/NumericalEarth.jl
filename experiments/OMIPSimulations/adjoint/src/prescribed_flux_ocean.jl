#####
##### OMIP ocean driven by prescribed ECCO4 surface fluxes
#####
# The same ocean as the JRA55 setup, but the surface fluxes are ECCO4's monthly wind stress (EXFtaue, EXFtaun),
# net heat flux (oceQnet) and net freshwater flux (oceFWflx), interpolated in time, instead of bulk formulae
# evaluated against a 3-hourly atmosphere. The control is then the wind stress itself on the ocean grid, a
# year is 13 slices instead of 2920, and the initial state is ECCO4's own.
#
# TODO: a prescribed-flux ocean belongs in NumericalEarth.Oceans, with the ECCO variables behind a
# DataWrangling hook.

"""
An ocean simulation whose surface fluxes are read from field time series: `stress` holds the eastward and
northward stress (N m⁻²), `heat_flux` the net heat flux into the ocean (W m⁻²) and `freshwater_flux` the
net freshwater mass flux into the ocean (kg m⁻² s⁻¹), all at cell centers of the ocean grid.
`update_state!` writes them into the ocean's surface flux fields at the current time; `time_step!` steps
the ocean and then updates the fluxes, as the coupled model does.
"""
struct PrescribedFluxOcean{O, S, Q, W, F, FT}
    ocean :: O
    stress :: S
    heat_flux :: Q
    freshwater_flux :: W
    surface_fluxes :: F
    reference_density :: FT
    heat_capacity :: FT
end

@inline function intrinsic_stress(i, j, k, grid, τᵉ, τⁿ, itp)
    @inbounds return intrinsic_vector(i, j, k, grid, τᵉ[i, j, 1, itp], τⁿ[i, j, 1, itp])
end

@inline intrinsic_zonal_stress(i, j, k, grid, τᵉ, τⁿ, itp)      = intrinsic_stress(i, j, k, grid, τᵉ, τⁿ, itp)[1]
@inline intrinsic_meridional_stress(i, j, k, grid, τᵉ, τⁿ, itp) = intrinsic_stress(i, j, k, grid, τᵉ, τⁿ, itp)[2]

# The ocean's surface flux fields are kinematic and positive upward, −τ / ρ₀ and −Q / (ρ₀ cₚ), except the
# freshwater volume flux Jʷ = F / ρ₀, positive into the ocean; the salinity boundary condition turns it into
# the virtual salt flux. The freshwater arrives at the surface temperature (heat content Tᴺ Jʷ), as in the
# coupled model.
@kernel function _prescribe_surface_fluxes!(fluxes, grid, T, τᵉ, τⁿ, Q, F, itp, ρ₀, cₚ)
    i, j = @index(Global, NTuple)
    kᴺ = size(grid, 3)
    inactive = inactive_node(i, j, kᴺ, grid, Center(), Center(), Center())
    @inbounds begin
        Jʷ = F[i, j, 1, itp] / ρ₀
        fluxes.u[i, j, 1] = ifelse(inactive, zero(grid), - ℑxᶠᵃᵃ(i, j, kᴺ, grid, intrinsic_zonal_stress, τᵉ, τⁿ, itp) / ρ₀)
        fluxes.v[i, j, 1] = ifelse(inactive, zero(grid), - ℑyᵃᶠᵃ(i, j, kᴺ, grid, intrinsic_meridional_stress, τᵉ, τⁿ, itp) / ρ₀)
        fluxes.T[i, j, 1] = ifelse(inactive, zero(grid), - Q[i, j, 1, itp] / (ρ₀ * cₚ))
        fluxes.η[i, j, 1] = ifelse(inactive, zero(grid), Jʷ)
        fluxes.freshwater_heat_content[i, j, 1] = ifelse(inactive, zero(grid), T[i, j, kᴺ] * Jʷ)
    end
end

function TimeSteppers.update_state!(model::PrescribedFluxOcean)
    grid = model.ocean.model.grid
    arch = architecture(grid)
    τᵉ, τⁿ = model.stress
    # Calendar months are unevenly spaced, so beyond two slices `times` is not a range and Oceananigans' generic
    # `find_time_index` reads `times[n]` at a traced index; under tracing that read is a dynamic slice.
    # TODO: `cpu_interpolating_time_indices` in OceananigansReactantExt should allow it for traced vectors.
    itp = @allowscalar cpu_interpolating_time_indices(arch, τᵉ.times, τᵉ.time_indexing, model.ocean.model.clock.time)
    launch!(arch, grid, :xy, _prescribe_surface_fluxes!, model.surface_fluxes, grid, model.ocean.model.tracers.T,
            τᵉ, τⁿ, model.heat_flux, model.freshwater_flux, itp, model.reference_density, model.heat_capacity)
    return nothing
end

function TimeSteppers.time_step!(model::PrescribedFluxOcean, Δt)
    time_step!(model.ocean.model, Δt)
    update_state!(model)
    return nothing
end

component_clocks(model::PrescribedFluxOcean) = (ocean_model(model).clock,)
wind_fields(model::PrescribedFluxOcean) = (u = model.stress.east, v = model.stress.north)
forcing_gradient(dmodel::PrescribedFluxOcean) = (; wind_gradient(dmodel)...,
                                                   heat_flux = interior_slices(dmodel.heat_flux),
                                                   freshwater_flux = interior_slices(dmodel.freshwater_flux))
forcing_shadows(dmodel::PrescribedFluxOcean) = (("stress.east", dmodel.stress.east), ("stress.north", dmodel.stress.north),
                                                 ("heat_flux", dmodel.heat_flux), ("freshwater_flux", dmodel.freshwater_flux))

# ECCO's potential temperature and practical salinity → TEOS-10 Conservative Temperature and Absolute Salinity.
function ecco_to_teos10!(T, S)
    grid = T.grid
    Nx, Ny, Nz = size(grid)
    for k in 1:Nz, j in 1:Ny, i in 1:Nx
        θ, Sᴾ = T[i, j, k], S[i, j, k]
        (isnan(θ) || isnan(Sᴾ)) && continue
        λ, φ, z = λnode(i, j, k, grid, Center(), Center(), Center()), φnode(i, j, k, grid, Center(), Center(), Center()),
                  znode(i, j, k, grid, Center(), Center(), Center())
        Sᴬ = Sᴬ_from_Sᴾ(Sᴾ, approx_pressure_dbar(z), λ, φ)
        T[i, j, k] = Θ_from_θᴾ(Sᴬ, θ)
        S[i, j, k] = Sᴬ
    end
    return T, S
end

# Monthly ECCO4 surface fields from `start_date`, regridded onto `cpu_grid` and copied into series on `grid`.
function ecco_surface_series(grid, cpu_grid, name, dates)
    times = [Dates.value(Millisecond(date - first(dates))) / 1000 for date in dates]
    series = FieldTimeSeries{Center, Center, Nothing}(grid, times)
    for (n, date) in enumerate(dates)
        slice = Field{Center, Center, Nothing}(cpu_grid)
        set!(slice, Metadatum(name; dataset = ECCO4Monthly(), date))
        parent(series[n]) .= Array(parent(slice))
    end
    return series
end

"""
$(TYPEDSIGNATURES)

Build the 2° OMIP ocean on `arch` driven by `n_months` of ECCO4 monthly wind stress, net heat flux and net
freshwater flux from `start_date`, initialized from ECCO4's temperature and salinity for that month, and
return `(; model, cpu_grid, Δt, times)` as [`omip_adjoint_setup`](@ref) does, with `model` a
[`PrescribedFluxOcean`](@ref).
"""
function ecco_adjoint_setup(arch; Nz = 20, depth = 5500, Δt = 7200.0, substeps = 50,
                            vertical_closure = :catke, κ_skew = 800, κ_symmetric = 800,
                            biharmonic_timescale = 50days, advection = :weno,
                            start_date = DateTime(1993, 1, 1), n_months = 2)

    n_months >= 2 || throw(ArgumentError("time interpolation needs n_months ≥ 2, got $n_months"))
    grid, cpu_grid = omip_adjoint_grid(arch; Nz, depth)
    ocean = omip_adjoint_ocean(grid, cpu_grid; Δt, substeps, vertical_closure, κ_skew, κ_symmetric,
                               biharmonic_timescale, advection, initial_date = start_date)

    dates = [start_date + Month(n) for n in 0:n_months-1]
    stress = (east  = ecco_surface_series(grid, cpu_grid, :zonal_wind_stress, dates),
              north = ecco_surface_series(grid, cpu_grid, :meridional_wind_stress, dates))
    heat_flux = ecco_surface_series(grid, cpu_grid, :net_heat_flux, dates)
    freshwater_flux = ecco_surface_series(grid, cpu_grid, :net_freshwater_flux, dates)

    FT = eltype(grid)
    model = PrescribedFluxOcean(ocean, stress, heat_flux, freshwater_flux, net_fluxes(ocean),
                                convert(FT, reference_density(ocean)), convert(FT, heat_capacity(ocean)))
    update_state!(model)

    return (; model, cpu_grid, Δt, times = heat_flux.times)
end
