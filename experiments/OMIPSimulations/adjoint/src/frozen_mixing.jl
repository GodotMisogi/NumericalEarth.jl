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
