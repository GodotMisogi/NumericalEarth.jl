#####
##### Overturning strength at a latitude: the AMOC objective
#####
# ψ(k) = -Σ_{k'≤k} Σ_i AW[i,k'] v(i,k') in m³ s⁻¹ on the model row nearest `latitude`, with the face
# area AW of the wet cells inside the section's longitude span precomputed on a CPU twin of the grid.
# The traced functor is a pure reduction (no element indexing of traced arrays).

"""
    OverturningStrength

Overturning streamfunction objective on the model row nearest `latitude`, over the wet cells whose
longitude lies in `longitudes` (degrees east), in Sverdrups. `mode` is `:fixed_residual` (ψ at
`reference_depth` after removing the section's net barotropic transport, smooth in the control) or
`:fixed_depth` (no residual correction).
"""
struct OverturningStrength{A, L, E}
    row :: Int
    area_weight :: A       # AW[i, k], m²
    cumulative :: L        # lower-triangular ones: (L * x)[k] = Σ_{k' ≤ k} x[k']
    level_selector :: E    # one-hot vector at the reference level
    mode :: Symbol
    latitude :: Float64
    reference_depth :: Float64
end

const overturning_modes = (:fixed_depth, :fixed_residual)

# Model row whose latitude is closest to `latitude`, probed at mid-longitude.
function nearest_row(grid, latitude)
    Nx, Ny, Nz = size(grid)
    distances = [abs(φnode(Nx ÷ 2, j, Nz, grid, Center(), Face(), Center()) - latitude) for j in 1:Ny]
    return argmin(distances)
end

# Face-area weight AW[i, k] = in_section · wet · Δxᶜᶠᶜ · Δzᶜᶠᶜ at the transect (m²), on a CPU grid.
function transect_area_weight(cpu_grid, row, longitudes)
    Nx, Ny, Nz = size(cpu_grid)
    AW = zeros(Float64, Nx, Nz)
    for k in 1:Nz, i in 1:Nx
        λ = mod(λnode(i, row, k, cpu_grid, Center(), Face(), Center()), 360)
        in_section = longitudes[1] <= λ <= longitudes[2]
        wet = !inactive_node(i, row, k, cpu_grid, Center(), Face(), Center())
        AW[i, k] = (in_section && wet) * Δxᶜᶠᶜ(i, row, k, cpu_grid) * Δzᶜᶠᶜ(i, row, k, cpu_grid)
    end
    return AW
end

"""
$(TYPEDSIGNATURES)

Build an [`OverturningStrength`](@ref) objective for the ocean on `cpu_grid` (a `CPU` twin of the
model grid, used for the host-side geometry). The default section is the RAPID line at 26.5°N from
Florida (80°W) to Africa (14°W).
"""
function OverturningStrength(cpu_grid; latitude = 26.5, longitudes = (280, 346), reference_depth = 1000.0,
                             mode = :fixed_residual)
    mode in overturning_modes || throw(ArgumentError("mode must be one of $overturning_modes, got $mode"))
    Nz = size(cpu_grid, 3)
    row = nearest_row(cpu_grid, latitude)
    AW = transect_area_weight(cpu_grid, row, longitudes)
    cumulative = Float64[k2 <= k1 for k1 in 1:Nz, k2 in 1:Nz]
    z_centers = [znode(1, 1, k, cpu_grid, Center(), Center(), Center()) for k in 1:Nz]
    k_ref = argmin(abs.(z_centers .+ reference_depth))
    level_selector = Float64[k == k_ref for k in 1:Nz]
    return OverturningStrength(row, AW, cumulative, level_selector, mode, Float64(latitude), Float64(reference_depth))
end

# Transport per level → ψ (Sv). The residual mode removes the area-weighted section-mean velocity,
# so ψ closes at the surface instead of carrying the open section's net through-flow.
function streamfunction(obj::OverturningStrength, colsum, Acol)
    cs = obj.mode === :fixed_residual ? (colsum .- (sum(colsum) / sum(Acol)) .* Acol) : colsum
    return -(obj.cumulative * cs)
end

# Traced functor on the coupled model: pure reductions on the v-transect.
function (obj::OverturningStrength)(model)
    v = interior(ocean_model(model).velocities.v, :, obj.row, :)     # (Nx, Nz)
    colsum = vec(sum(obj.area_weight .* v; dims = 1))                 # (Nz,) m³ s⁻¹
    Acol   = vec(sum(obj.area_weight; dims = 1))                      # (Nz,) m²
    return sum(obj.level_selector .* streamfunction(obj, colsum, Acol)) / 1e6
end

"""
$(TYPEDSIGNATURES)

Host-side evaluation of `obj` on `model` (pulls the transect to the CPU first).
"""
function objective_host(obj::OverturningStrength, model)
    v = Array(interior(ocean_model(model).velocities.v, :, obj.row, :))
    colsum = vec(sum(obj.area_weight .* v; dims = 1))
    Acol   = vec(sum(obj.area_weight; dims = 1))
    return sum(obj.level_selector .* streamfunction(obj, colsum, Acol)) / 1e6
end

ocean_model(model) = model.ocean.model
