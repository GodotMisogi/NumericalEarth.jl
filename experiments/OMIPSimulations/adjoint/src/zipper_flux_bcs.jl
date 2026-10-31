#####
##### No flux-boundary kernel for a `Zipper` north boundary
#####
# With a no-flux south boundary the y-boundary kernel writes nothing. Raised, the empty launch becomes an
# operand-free `enzymexla.jit_call` that Enzyme cannot differentiate (EnzymeAD/Reactant.jl#3636).
#
# TODO: remove once the pinned Oceananigans includes CliMA/Oceananigans.jl#6396 (`ZBC` in `NotFluxBC`).

BoundaryConditions.compute_y_bcs!(Gc::AbstractArray, ::AbstractGrid, c, ::NotFluxBC, ::ZBC,
                                  ::AbstractArchitecture, args...) = nothing
