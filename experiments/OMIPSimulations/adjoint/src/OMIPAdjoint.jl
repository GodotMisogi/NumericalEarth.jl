module OMIPAdjoint

export omip_adjoint_setup, ecco_adjoint_setup, PrescribedFluxOcean,
       OverturningStrength, objective_host,
       roll_barotropic_substeps!, freeze_mixing!,
       adjoint_compile_options, snapshot_state, build_primal, build_gradient, build_checkpointed_gradient,
       wind_gradient, shadow_census, print_shadow_census, taylor_remainder_check

using Oceananigans
using Oceananigans: prognostic_state, restore_prognostic_state!
using Oceananigans.Units: days
using Oceananigans.Architectures: AbstractArchitecture, CPU, ReactantState, architecture, on_architecture,
                                  convert_to_device
using Oceananigans.BoundaryConditions: BoundaryConditions, ZBC, NotFluxBC
using Oceananigans.Grids: AbstractGrid, ExponentialDiscretization, LatitudeLongitudeGrid, inactive_node,
                          λnode, φnode, znode, Center, Face, Periodic, Bounded, Flat
using Oceananigans.Operators: Δxᶜᶠᶜ, Δzᶜᶠᶜ, Δyᶠᶜᶜ, Δzᶠᶜᶜ, ℑxᶠᵃᵃ, ℑyᵃᶠᵃ, intrinsic_vector
using Oceananigans.Fields: Field, CenterField, interior, interpolate!
using Oceananigans.OutputReaders: FieldTimeSeries, cpu_interpolating_time_indices
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBottom
using Oceananigans.OrthogonalSphericalShellGrids: TripolarGrid
using Oceananigans.Coriolis: HydrostaticSphericalCoriolis, EnstrophyConserving
using Oceananigans.TurbulenceClosures: TurbulenceClosures, ExplicitTimeDiscretization
using Oceananigans.TurbulenceClosures.TKEBasedVerticalDiffusivities: FlavorOfCATKE
using Oceananigans.Models.HydrostaticFreeSurfaceModels: HydrostaticFreeSurfaceModel
using Oceananigans.TimeSteppers: TimeSteppers, Clock, time_step!, update_state!
using Oceananigans.Utils: configure_kernel, launch!
using Oceananigans.Models.HydrostaticFreeSurfaceModels.SplitExplicitFreeSurfaces:
    SplitExplicitFreeSurfaces, SplitExplicitFreeSurface, ExtendedHalos,
    _split_explicit_barotropic_velocity!, _split_explicit_free_surface!

using NumericalEarth: OceanOnlyModel, PrescribedAtmosphere, PrescribedRadiation, SurfaceRadiationProperties,
                      regrid_bathymetry
using NumericalEarth.Oceans: ocean_simulation
using NumericalEarth.DataWrangling: Metadatum
using NumericalEarth.DataWrangling.WOA: WOAAnnual
using NumericalEarth.DataWrangling.ECCO: ECCO4Monthly
using NumericalEarth.DataWrangling.JRA55: RepeatYearJRA55, JRA55PrescribedAtmosphere, JRA55PrescribedRadiation
using NumericalEarth.EarthSystemModels: default_sea_ice, reference_density, heat_capacity
using NumericalEarth.EarthSystemModels.InterfaceComputations: ComponentInterfaces, net_fluxes

using OMIPSimulations: omip_closure, split_momentum_advection, split_tracer_advection, exponential_scale,
                       ncar_atmosphere_ocean_fluxes, woa_to_teos10!, approx_pressure_dbar

using KernelAbstractions: @kernel, @index
using Reactant: Reactant, @compile, @trace, @allowscalar
using SeawaterPolynomials.TEOS10: Sᴬ_from_Sᴾ, Θ_from_θᴾ
using Enzyme: autodiff, set_strong_zero, Reverse, ReverseWithPrimal, Active, Duplicated, Const, make_zero, ignore_derivatives
using Checkpointing: Checkpointing, Revolve, next_action!
using Dates: Dates, DateTime, Hour, Month, Millisecond
using Printf: @printf
using DocStringExtensions: TYPEDSIGNATURES

include("ocean_only_model.jl")
include("prescribed_flux_ocean.jl")
include("overturning.jl")
include("rolled_substeps.jl")
include("frozen_mixing.jl")
include("zipper_flux_bcs.jl")
include("gradient.jl")
include("checkpointed_gradient.jl")

end # module OMIPAdjoint
