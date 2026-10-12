#####
##### Ocean-only OMIP model for differentiation
#####
# The OMIP ocean physics (tripolar grid, ETOPO bathymetry, WOA initial state, CATKE + GM/Redi +
# biharmonic closures, TEOS-10, split-explicit free surface, SplitRK3) coupled to the prescribed JRA55
# atmosphere and radiation through `OceanOnlyModel`. The same construction runs on `CPU()` and on
# `ReactantState()`: the forcing window is loaded into memory up front and nothing reads from disk
# during a time step.

"""
$(TYPEDSIGNATURES)

The 2° tripolar OMIP grid with ETOPO bathymetry, returned as `(; grid, cpu_grid)`: the grid on `arch`
and its `CPU()` twin. The bathymetry is regridded on the CPU (host work) and copied onto `arch`.
"""
function omip_adjoint_grid(arch; Nx = 180, Ny = 60, Nz = 20, depth = 5500)
    z = ExponentialDiscretization(Nz, -depth, 0; scale = exponential_scale(Nz, depth, nothing))
    size, halo = (Nx, Ny, Nz), (8, 8, 8)

    cpu_underlying = TripolarGrid(CPU(); size, z, halo)
    cpu_bottom = regrid_bathymetry(cpu_underlying; minimum_depth = 20, major_basins = 1, interpolation_passes = 25)
    cpu_grid = ImmersedBoundaryGrid(cpu_underlying, GridFittedBottom(cpu_bottom))
    arch isa CPU && return (; grid = cpu_grid, cpu_grid)

    underlying = TripolarGrid(arch; size, z, halo)
    bottom = Field{Center, Center, Nothing}(underlying)
    parent(bottom) .= Array(parent(cpu_bottom))
    return (; grid = ImmersedBoundaryGrid(underlying, GridFittedBottom(bottom)), cpu_grid)
end

"""
$(TYPEDSIGNATURES)

The OMIP ocean simulation on `grid`: `omip_closure` (CATKE by default, GM/Redi when `κ_skew` and
`κ_symmetric` are given, biharmonic viscosity from `biharmonic_timescale`), split WENO advection (explicit
in the vertical), enstrophy-conserving Coriolis, `substeps` split-explicit
barotropic substeps and SplitRK3 time stepping. The initial state is the WOA annual climatology
converted to TEOS-10 variables on `cpu_grid` and copied onto `grid`.
"""
function omip_adjoint_ocean(grid, cpu_grid; Δt, substeps = 50, vertical_closure = :catke,
                            κ_skew = 800, κ_symmetric = 800, biharmonic_timescale = 50days)

    closure = omip_closure(vertical_closure; grid, κ_skew, κ_symmetric, biharmonic_timescale,
                           background_vertical_diffusivity = :henyey)

    # Vertical advection is explicit: the adaptive implicit discretization keeps host state.
    momentum_advection = split_momentum_advection(:weno, 5, ExplicitTimeDiscretization(), Val(:default))
    tracer_scheme      = split_tracer_advection(7, ExplicitTimeDiscretization(), Val(:default))

    ocean = ocean_simulation(grid;
                             Δt,
                             timestepper = :SplitRungeKutta3,
                             free_surface = SplitExplicitFreeSurface(; substeps),
                             coriolis = HydrostaticSphericalCoriolis(scheme = EnstrophyConserving()),
                             closure,
                             momentum_advection,
                             tracer_advection = (T = tracer_scheme, S = tracer_scheme),
                             radiative_forcing = nothing,
                             implicit_bottom_drag = true,
                             warn = false)

    T_init = CenterField(cpu_grid)
    S_init = CenterField(cpu_grid)
    set!(T_init, Metadatum(:temperature; dataset = WOAAnnual()))
    set!(S_init, Metadatum(:salinity;    dataset = WOAAnnual()))
    woa_to_teos10!(T_init, S_init)
    parent(ocean.model.tracers.T) .= Array(parent(T_init))
    parent(ocean.model.tracers.S) .= Array(parent(S_init))

    return ocean
end

# Apply `op!(dst, src)` to every forcing slice, pairing the destination atmosphere / radiation with the source set.
function map_forcing_slices!(op!, times, dst_atmos, dst_rad, src_atmos, src_rad)
    for n in eachindex(times)
        op!(dst_atmos.velocities.u[n],            src_atmos.velocities.u[n])
        op!(dst_atmos.velocities.v[n],            src_atmos.velocities.v[n])
        op!(dst_atmos.temperature[n],             src_atmos.temperature[n])
        op!(dst_atmos.specific_humidity[n],       src_atmos.specific_humidity[n])
        op!(dst_atmos.pressure[n],                src_atmos.pressure[n])
        op!(dst_atmos.precipitation_flux.rain[n], src_atmos.precipitation_flux.rain[n])
        op!(dst_atmos.precipitation_flux.snow[n], src_atmos.precipitation_flux.snow[n])
        op!(dst_rad.downwelling_shortwave[n],     src_rad.downwelling_shortwave[n])
        op!(dst_rad.downwelling_longwave[n],      src_rad.downwelling_longwave[n])
    end
    return nothing
end

copy_slice!(dst, src) = (parent(dst) .= Array(parent(src)); nothing)

const ocean_surface_radiation = SurfaceRadiationProperties(0.06, 1.0)

"""
$(TYPEDSIGNATURES)

JRA55 repeat-year atmosphere and downwelling radiation for `n_slices` three-hourly slices from
`start_date`, loaded on the CPU and copied slice by slice into in-memory `PrescribedAtmosphere` and
`PrescribedRadiation` objects on `arch`, so that the coupled time step does no file I/O.
"""
function omip_adjoint_forcing(arch; start_date = DateTime(1990, 1, 1), n_slices = 16)
    dataset  = RepeatYearJRA55()
    end_date = start_date + Hour(3 * (n_slices - 1))
    kw = (; dataset, start_date, end_date, time_indices_in_memory = n_slices)

    cpu_atmosphere = JRA55PrescribedAtmosphere(CPU(); kw..., prefetch = false)
    cpu_radiation  = JRA55PrescribedRadiation(CPU(); kw..., ocean_surface = ocean_surface_radiation,
                                              sea_ice_surface = nothing)

    times = cpu_atmosphere.velocities.u.times
    grid  = on_architecture(arch, cpu_atmosphere.velocities.u.grid)

    atmosphere = PrescribedAtmosphere(grid, times; surface_layer_height = cpu_atmosphere.surface_layer_height,
                                      boundary_layer_height = cpu_atmosphere.boundary_layer_height)
    radiation  = PrescribedRadiation(grid, times; ocean_surface = ocean_surface_radiation, sea_ice_surface = nothing)

    map_forcing_slices!(copy_slice!, times, atmosphere, radiation, cpu_atmosphere, cpu_radiation)
    update_state!(atmosphere)
    update_state!(radiation)

    return (; atmosphere, radiation, times)
end

"""
$(TYPEDSIGNATURES)

Build the ocean-only OMIP model on `arch` and return `(; model, cpu_grid, Δt, times)`: the
`OceanOnlyModel`, the `CPU()` twin of its grid (for host-side geometry), the time step and the forcing
times. `arch` is `CPU()` or `ReactantState()`; keyword arguments go to [`omip_adjoint_grid`](@ref),
[`omip_adjoint_ocean`](@ref) and [`omip_adjoint_forcing`](@ref).
"""
function omip_adjoint_setup(arch; Nz = 20, depth = 5500, Δt = 7200.0, substeps = 50,
                            vertical_closure = :catke, κ_skew = 800, κ_symmetric = 800,
                            biharmonic_timescale = 50days,
                            start_date = DateTime(1990, 1, 1), n_atmosphere_slices = 16)

    grid, cpu_grid = omip_adjoint_grid(arch; Nz, depth)
    ocean = omip_adjoint_ocean(grid, cpu_grid; Δt, substeps, vertical_closure, κ_skew, κ_symmetric,
                               biharmonic_timescale)
    forcing  = omip_adjoint_forcing(arch; start_date, n_slices = n_atmosphere_slices)

    # OMIP-2 Large & Yeager coefficients with a fixed iteration count, so the flux solver traces.
    interfaces = ComponentInterfaces(forcing.atmosphere, ocean, default_sea_ice();
                                     radiation = forcing.radiation,
                                     atmosphere_ocean_fluxes = ncar_atmosphere_ocean_fluxes(Float64))

    model = OceanOnlyModel(ocean; atmosphere = forcing.atmosphere, radiation = forcing.radiation, interfaces)

    return (; model, cpu_grid, Δt, times = forcing.times)
end
