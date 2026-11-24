module OMIPAdjoint

export omip_adjoint_setup, ecco_adjoint_setup, PrescribedFluxOcean,
       OverturningStrength, objective_host,
       roll_barotropic_substeps!, freeze_mixing!, freeze_buoyancy_gradients!,
       adjoint_compile_options, snapshot_state, build_primal, build_gradient, build_checkpointed_gradient,
       AdjointDamping, AdjointTrace, step_adjoint_growth,
       wind_gradient, forcing_gradient, surface_geometry, shadow_census, print_shadow_census,
       taylor_remainder_check, finite_difference_scan,
       eddy_diffusivity_gradient, eddy_diffusivity, set_eddy_diffusivity!, eddy_diffusivity_taylor_check,
       compile_checkpointed_thunks, checkpointed_gradient, window_average_objective!,
       TracerWeights, TracerMisfit, observed_cells, misfit, build_eddy_calibration, bfgs_minimize

using Oceananigans
using Oceananigans: prognostic_state, restore_prognostic_state!
using Oceananigans.Units: days
using Oceananigans.Architectures: CPU, ReactantState, architecture, on_architecture,
                                  convert_to_device, AbstractArchitecture
using Oceananigans.Grids: AbstractGrid, ExponentialDiscretization, LatitudeLongitudeGrid, inactive_node, peripheral_node,
                          λnode, φnode, znode, Center, Face, Periodic, Bounded, Flat, volume_kernel_parameters
using Oceananigans.Operators: Δxᶜᶠᶜ, Δzᶜᶠᶜ, Δyᶠᶜᶜ, Δzᶠᶜᶜ, Azᶜᶜᶜ, ℑxᶠᵃᵃ, ℑyᵃᶠᵃ, intrinsic_vector,
                              Δxᶜᶜᶜ, Δyᶜᶜᶜ, Δxᶠᶜᶜ, Δyᶜᶠᶜ
using Oceananigans.BoundaryConditions: fill_halo_regions!
using Oceananigans.Fields: Field, CenterField, interior, interpolate!, instantiated_location
using Oceananigans.OutputReaders: FieldTimeSeries, cpu_interpolating_time_indices
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBottom
using Oceananigans.OrthogonalSphericalShellGrids: TripolarGrid
using Oceananigans.BuoyancyFormulations: BuoyancyFormulations, BuoyancyForce
using Oceananigans.Coriolis: HydrostaticSphericalCoriolis, EnstrophyConserving
using Oceananigans.TurbulenceClosures: TurbulenceClosures, ExplicitTimeDiscretization, HorizontalScalarDiffusivity,
                                     HorizontalScalarBiharmonicDiffusivity, ConvectiveAdjustmentVerticalDiffusivity,
                                     IsopycnalSkewSymmetricDiffusivity, ∂ⱼ_τ₁ⱼ, ∂ⱼ_τ₂ⱼ, ∇_dot_qᶜ, with_tracers
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

using OMIPSimulations: omip_closure, split_momentum_advection, split_tracer_advection, exponential_scale, νhb,
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
include("gradient.jl")
include("adjoint_damping.jl")
include("adjoint_trace.jl")
include("checkpointed_gradient.jl")
include("step_jacobian.jl")
include("calibration.jl")

end # module OMIPAdjoint
