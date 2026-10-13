module OMIPAdjoint

export omip_adjoint_setup,
       OverturningStrength, objective_host,
       roll_barotropic_substeps!,
       adjoint_compile_options, snapshot_state, build_reset, build_primal, build_gradient,
       wind_gradient, shadow_census, print_shadow_census, taylor_remainder_check

using Oceananigans
using Oceananigans: prognostic_state, restore_prognostic_state!
using Oceananigans.Units: days
using Oceananigans.Architectures: CPU, ReactantState, architecture, on_architecture, convert_to_device
using Oceananigans.Grids: AbstractGrid, ExponentialDiscretization, LatitudeLongitudeGrid, inactive_node,
                          λnode, φnode, znode, Center, Face, Periodic, Bounded, Flat
using Oceananigans.Operators: Δxᶜᶠᶜ, Δzᶜᶠᶜ
using Oceananigans.Fields: Field, CenterField, interior, interpolate!
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBottom
using Oceananigans.OrthogonalSphericalShellGrids: TripolarGrid
using Oceananigans.Coriolis: HydrostaticSphericalCoriolis, EnstrophyConserving
using Oceananigans.TurbulenceClosures: ExplicitTimeDiscretization
using Oceananigans.TimeSteppers: time_step!, update_state!
using Oceananigans.Utils: configure_kernel
using Oceananigans.Models.HydrostaticFreeSurfaceModels.SplitExplicitFreeSurfaces:
    SplitExplicitFreeSurfaces, SplitExplicitFreeSurface, ExtendedHalos,
    _split_explicit_barotropic_velocity!, _split_explicit_free_surface!

using NumericalEarth: OceanOnlyModel, PrescribedAtmosphere, PrescribedRadiation, SurfaceRadiationProperties,
                      regrid_bathymetry
using NumericalEarth.Oceans: ocean_simulation
using NumericalEarth.DataWrangling: Metadatum
using NumericalEarth.DataWrangling.WOA: WOAAnnual
using NumericalEarth.DataWrangling.JRA55: RepeatYearJRA55, JRA55PrescribedAtmosphere, JRA55PrescribedRadiation
using NumericalEarth.EarthSystemModels: default_sea_ice
using NumericalEarth.EarthSystemModels.InterfaceComputations: ComponentInterfaces

using OMIPSimulations: omip_closure, split_momentum_advection, split_tracer_advection, exponential_scale,
                       ncar_atmosphere_ocean_fluxes, woa_to_teos10!

using Reactant: Reactant, @compile, @trace, @allowscalar
using Enzyme: autodiff, set_strong_zero, ReverseWithPrimal, Active, Duplicated, Const, make_zero
using Dates: DateTime, Hour
using Printf: @printf
using DocStringExtensions: TYPEDSIGNATURES

include("ocean_only_model.jl")
include("overturning.jl")
include("rolled_substeps.jl")
include("gradient.jl")

end # module OMIPAdjoint
