#####
##### Rolled barotropic substepping for Reactant-traced split-explicit free surfaces
#####
# Oceananigans unrolls the barotropic substep loop, so a traced step carries one copy of the two
# barotropic kernels per substep and per Runge–Kutta stage. With `roll_barotropic_substeps!(true)` the
# loop is traced as one `stablehlo.while` whose weights are read from constant arrays by the traced
# substep index. Only the extended-halo strategy (no halo fills between substeps) is covered.
#
# TODO: move into `OceananigansReactantExt`.

const ROLL_BAROTROPIC_SUBSTEPS = Ref(false)

"""
$(TYPEDSIGNATURES)

Choose how split-explicit barotropic substeps are traced on `ReactantState` grids with extended
halos: `true` traces them as a single loop (one copy of the substep kernels), `false` keeps
Oceananigans' unrolled loop (one copy per substep). The choice is read when a model step is
traced, so it must be set before `@compile`. Returns `flag`.
"""
roll_barotropic_substeps!(flag::Bool) = (ROLL_BAROTROPIC_SUBSTEPS[] = flag; flag)

const ReactantGrid = AbstractGrid{<:Any, <:Any, <:Any, <:Any, <:ReactantState}

function SplitExplicitFreeSurfaces.iterate_split_explicit!(free_surface::SplitExplicitFreeSurface{ExtendedHalos},
                                                           grid::ReactantGrid, GUⁿ, GVⁿ, Δτᴮ, F, clock,
                                                           weights, transport_weights, ::Val{N}) where N
    ROLL_BAROTROPIC_SUBSTEPS[] ||
        return SplitExplicitFreeSurfaces.iterate_split_explicit_in_halo!(free_surface, grid, GUⁿ, GVⁿ, Δτᴮ, F, clock,
                                                                         weights, transport_weights, Val(N))
    # Mirrors `iterate_split_explicit_in_halo!` except for the loop.
    η           = free_surface.displacement
    grid        = η.grid
    arch        = architecture(grid)
    state       = free_surface.filtered_state
    timestepper = free_surface.timestepper
    g           = free_surface.gravitational_acceleration
    U, V        = free_surface.barotropic_velocities

    velocity_kernel!, _ = configure_kernel(arch, grid, free_surface.kernel_parameters, _split_explicit_barotropic_velocity!)
    η_kernel!, _        = configure_kernel(arch, grid, free_surface.kernel_parameters, _split_explicit_free_surface!)

    U_args = convert_to_device(arch, (grid, Val(false), Δτᴮ, η, U, V, GUⁿ, GVⁿ, g, state.Ũ, state.Ṽ, timestepper))
    η_args = convert_to_device(arch, (grid, Val(false), Δτᴮ, η, U, V, F, clock, state.η̅, state.U̅, state.V̅, timestepper))

    averaging = Reactant.Ops.constant(collect(weights))
    transport = Reactant.Ops.constant(collect(transport_weights))
    # Indexing a traced array, even by the traced loop index, counts as scalar indexing; under
    # @allowscalar it lowers to a dynamic slice.
    @trace track_numbers = false for substep in 1:N
        velocity_kernel!(@allowscalar(transport[substep]), U_args...)
        η_kernel!(@allowscalar(averaging[substep]), η_args...)
    end
    return nothing
end
