#####
##### Frozen vertical mixing for the adjoint
#####
# With `freeze_mixing!(true)` the CATKE viscosity and diffusivities are computed as usual in the forward
# pass but enter the derivative as constants: each is passed through `ignore_derivatives` (a stop-gradient
# in the traced graph) right after `compute_closure_fields!` writes it. The tracer and momentum fluxes read
# these stored fields, so no derivative reaches the TKE equation or the mixing-length computation, and
# Enzyme drops their reverse pass. The forward model, hence the objective, is unchanged.
#
# TODO: an Oceananigans option (e.g. a closure keyword) instead of this method override.

const FREEZE_MIXING = Ref(false)

"""
$(TYPEDSIGNATURES)

Choose whether the CATKE viscosity and diffusivities are held constant in derivatives (`true`) or
differentiated (`false`). The choice is read when a model step is traced, so it must be set before
`@compile`. Returns `flag`.
"""
freeze_mixing!(flag::Bool) = (FREEZE_MIXING[] = flag; flag)

function TurbulenceClosures.compute_closure_fields!(closure_fields, closure::FlavorOfCATKE,
                                                    model::HydrostaticFreeSurfaceModel; parameters = :xyz)
    invoke(TurbulenceClosures.compute_closure_fields!, Tuple{Any, FlavorOfCATKE, Any},
           closure_fields, closure, model; parameters)
    if FREEZE_MIXING[]
        for κ in (closure_fields.κu, closure_fields.κc, closure_fields.κe)
            parent(κ) .= ignore_derivatives(parent(κ))
        end
    end
    return nothing
end

const FREEZE_TKE = Ref(false)

"""
$(TYPEDSIGNATURES)

Choose whether the CATKE turbulent kinetic energy is held constant in derivatives after each step of its equation
(`true`): the viscosity and diffusivities still vary with N² and shear in the derivative, but not through e, and no
derivative reaches the TKE equation. The choice is read when a model step is traced, so it must be set before
`@compile`. Returns `flag`.
"""
freeze_tke!(flag::Bool) = (FREEZE_TKE[] = flag; flag)

function TurbulenceClosures.step_closure_prognostics!(closure_fields, closure::FlavorOfCATKE,
                                                      model::HydrostaticFreeSurfaceModel, Δt)
    invoke(TurbulenceClosures.step_closure_prognostics!, Tuple{Any, FlavorOfCATKE, Any, Any},
           closure_fields, closure, model, Δt)
    FREEZE_TKE[] && (parent(model.tracers.e) .= ignore_derivatives(parent(model.tracers.e)))
    return nothing
end

const FREEZE_BUOYANCY_GRADIENTS = Ref(false)

"""
$(TYPEDSIGNATURES)

Choose whether stored buoyancy gradients (∂xᵣb, ∂yᵣb, ∂zb of a `BuoyancyForce` built with
`materialize_gradients = true`) are held constant in derivatives (`true`) or differentiated (`false`). Every
isopycnal slope, slope taper and stratification test then enters the derivative as a constant, while the
fluxes stay linear in the tracer they carry. The choice is read when a model step is traced, so it must be set
before `@compile`. Returns `flag`.
"""
freeze_buoyancy_gradients!(flag::Bool) = (FREEZE_BUOYANCY_GRADIENTS[] = flag; flag)

function BuoyancyFormulations.compute_buoyancy_gradients!(buoyancy::BuoyancyForce{<:Any, <:Any, <:NamedTuple}, grid,
                                                         tracers; parameters = :xyz)
    invoke(BuoyancyFormulations.compute_buoyancy_gradients!, Tuple{Any, Any, Any}, buoyancy, grid, tracers;
           parameters)
    if FREEZE_BUOYANCY_GRADIENTS[]
        for g in buoyancy.gradients
            parent(g) .= ignore_derivatives(parent(g))
        end
    end
    return nothing
end

# CATKE's TKE equation is stepped after each stage's tracer update and before `update_state!` refreshes stored
# gradients, so it would read the stage's starting N² where inline gradients give the updated one. Recomputing them
# first keeps the forward model of stored gradients the same as that of inline ones, to rounding.
# TODO: Oceananigans' `step_closure_prognostics!` for `HydrostaticFreeSurfaceModel` should refresh materialized
# buoyancy gradients itself; `materialize_gradients = true` otherwise changes the CATKE forward model.
const StoredBuoyancyGradientsModel =
    HydrostaticFreeSurfaceModel{<:Any, <:Any, <:AbstractArchitecture, <:Any, <:Any, <:Any, <:Any,
                                <:BuoyancyForce{<:Any, <:Any, <:NamedTuple}}

function TurbulenceClosures.step_closure_prognostics!(model::StoredBuoyancyGradientsModel, Δt::Number)
    grid = model.grid
    BuoyancyFormulations.compute_buoyancy_gradients!(model.buoyancy, grid, model.tracers;
                                                     parameters = volume_kernel_parameters(grid))
    TurbulenceClosures.step_closure_prognostics!(model.closure_fields, model.closure, model, Δt)
    return nothing
end
