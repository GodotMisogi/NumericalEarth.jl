#####
##### Adjoint-only horizontal damping
#####
# Long-window adjoints of the frozen-mixing model grow without bound; the standard remedy is extra
# dissipation in the adjoint alone, with the forward trajectory unchanged (Hoteit et al. 2005). Here the
# extra dissipation is one explicit step of horizontal Laplacian viscosity and diffusion, written as a forward
# operator D on a scratch copy of (u, v, T, S) and applied to the adjoint as Dᵀ, by Enzyme, after each reverse
# step. Differentiating D, rather than hand-writing Dᵀ, gets the halos right: after a reverse step the adjoint's
# halos still hold contributions that the reverse of the previous step's halo fill folds into the interior,
# and Dᵀ folds them first, through the reverse of D's last halo fill, with the tripolar fold's signs.

struct AdjointDamping{C, FT}
    closure :: C
    Δt :: FT
end

"""
$(TYPEDSIGNATURES)

Adjoint-only horizontal Laplacian damping with `viscosity` for the adjoint velocities and `diffusivity` for the
adjoint temperature and salinity (m² s⁻¹), applied once per time step of length `Δt`. Throws an
`ArgumentError` if either coefficient is negative, or if the explicit step is unstable on any wet cell of
`cpu_grid`: max ν Δt (1/Δx² + 1/Δy²) must not exceed 1/2.
"""
function AdjointDamping(cpu_grid; viscosity = 0, diffusivity = 0, Δt)
    viscosity >= 0 || throw(ArgumentError("adjoint viscosity must be non-negative, got $viscosity"))
    diffusivity >= 0 || throw(ArgumentError("adjoint diffusivity must be non-negative, got $diffusivity"))
    number = max(viscosity, diffusivity) * Δt * maximum_inverse_squared_spacing(cpu_grid)
    number <= 1/2 ||
        throw(ArgumentError("explicit adjoint damping is unstable: max ν Δt (1/Δx² + 1/Δy²) = $number > 1/2; " *
                            "lower the coefficients (ν = $viscosity, κ = $diffusivity m² s⁻¹) or Δt = $Δt s"))
    closure = with_tracers((:T, :S), HorizontalScalarDiffusivity(ν = viscosity, κ = diffusivity))
    return AdjointDamping(closure, Δt)
end

# 1/Δx² + 1/Δy² over the surface cells and faces that are wet, at the locations of T, u and v.
function maximum_inverse_squared_spacing(grid)
    Nx, Ny, Nz = size(grid)
    c, f = Center(), Face()
    s = 0.0
    for j in 1:Ny, i in 1:Nx
        inactive_node(i, j, Nz, grid, c, c, c) ||
            (s = max(s, 1 / Δxᶜᶜᶜ(i, j, Nz, grid)^2 + 1 / Δyᶜᶜᶜ(i, j, Nz, grid)^2))
        inactive_node(i, j, Nz, grid, f, c, c) ||
            (s = max(s, 1 / Δxᶠᶜᶜ(i, j, Nz, grid)^2 + 1 / Δyᶠᶜᶜ(i, j, Nz, grid)^2))
        inactive_node(i, j, Nz, grid, c, f, c) ||
            (s = max(s, 1 / Δxᶜᶠᶜ(i, j, Nz, grid)^2 + 1 / Δyᶜᶠᶜ(i, j, Nz, grid)^2))
    end
    return s
end

damped_fields(model) = (u = model.velocities.u, v = model.velocities.v, T = model.tracers.T, S = model.tracers.S)

# Fields with the grid and boundary conditions of the damped ocean fields (the tripolar fold flips the sign of u
# and v), for D's primal. Their values do not matter: D is linear.
function damping_scratch(ocean)
    fields = map(f -> Field(instantiated_location(f), f.grid; boundary_conditions = f.boundary_conditions),
                 damped_fields(ocean))
    tendencies = map(f -> Field(instantiated_location(f), f.grid), fields)
    return (; fields, tendencies)
end

@kernel function _horizontal_damping_tendencies!(G, grid, closure, fields)
    i, j, k = @index(Global, NTuple)
    @inbounds begin
        G.u[i, j, k] = - ∂ⱼ_τ₁ⱼ(i, j, k, grid, closure, nothing, nothing, fields, nothing)
        G.v[i, j, k] = - ∂ⱼ_τ₂ⱼ(i, j, k, grid, closure, nothing, nothing, fields, nothing)
        G.T[i, j, k] = - ∇_dot_qᶜ(i, j, k, grid, closure, nothing, Val(1), fields.T, nothing, fields, nothing)
        G.S[i, j, k] = - ∇_dot_qᶜ(i, j, k, grid, closure, nothing, Val(2), fields.S, nothing, fields, nothing)
    end
end

@kernel function _horizontal_damping_update!(fields, grid, G, Δt)
    i, j, k = @index(Global, NTuple)
    c, f = Center(), Face()
    @inbounds begin
        fields.u[i, j, k] += ifelse(peripheral_node(i, j, k, grid, f, c, c), zero(grid), Δt * G.u[i, j, k])
        fields.v[i, j, k] += ifelse(peripheral_node(i, j, k, grid, c, f, c), zero(grid), Δt * G.v[i, j, k])
        fields.T[i, j, k] += ifelse(inactive_node(i, j, k, grid, c, c, c), zero(grid), Δt * G.T[i, j, k])
        fields.S[i, j, k] += ifelse(inactive_node(i, j, k, grid, c, c, c), zero(grid), Δt * G.S[i, j, k])
    end
end

"""
$(TYPEDSIGNATURES)

The forward damping operator D: one explicit step of `damping`'s horizontal Laplacian viscosity and diffusion
on `scratch.fields` (u, v, T, S), using `scratch.tendencies` as work space, with the halos filled before and
after.
"""
function damping_step!(scratch, damping)
    fields, G = scratch.fields, scratch.tendencies
    grid = fields.u.grid
    foreach(fill_halo_regions!, fields)
    launch!(architecture(grid), grid, :xyz, _horizontal_damping_tendencies!, G, grid, damping.closure, fields)
    launch!(architecture(grid), grid, :xyz, _horizontal_damping_update!, fields, grid, G, damping.Δt)
    foreach(fill_halo_regions!, fields)
    return nothing
end

"""
$(TYPEDSIGNATURES)

Apply Dᵀ, the adjoint of [`damping_step!`](@ref), to the adjoint ocean fields of `dmodel` in place.
`dtendencies` is the zero shadow of the scratch tendencies, and stays zero.
"""
damp_adjoint!(scratch, dmodel, dtendencies, damping) =
    autodiff(set_strong_zero(Reverse), damping_step!, Const,
             Duplicated(scratch, (fields = damped_fields(ocean_model(dmodel)), tendencies = dtendencies)),
             Const(damping))

# Returns the compiled Dᵀ as a function of the shadow model, or a no-op without damping.
compile_adjoint_damping(model, dmodel, ::Nothing, compile_options) = Returns(nothing)

function compile_adjoint_damping(model, dmodel, damping::AdjointDamping, compile_options)
    scratch = damping_scratch(ocean_model(model))
    dtendencies = make_zero(scratch.tendencies)
    thunk = @compile compile_options = compile_options damp_adjoint!(scratch, dmodel, dtendencies, damping)
    return dmodel -> thunk(scratch, dmodel, dtendencies, damping)
end
