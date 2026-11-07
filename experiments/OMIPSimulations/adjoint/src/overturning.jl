#####
##### Overturning strength across a latitude: the AMOC objective
#####
# The section is the staircase of cell faces closest to `latitude`: in each column the v-face nearest the
# latitude, joined across columns by the u-faces between them. On a curvilinear grid (the tripolar grid's
# rows are not latitude circles) a single row would drift off the latitude. With S the cells south of the
# staircase, the transport per level is the volume flux out of S,
#   T(k) = Σ_v-faces Δx Δz v + Σ_u-faces ± Δy Δz u,
# and ψ(k) = -Σ_{k'≤k} T(k') in m³ s⁻¹. The signed face areas are precomputed on a CPU twin of the grid over
# the few rows the staircase visits, so the traced functor is a pure reduction (no element indexing of
# traced arrays).

"""
    OverturningStrength

Overturning streamfunction objective across `latitude`, over the wet faces whose longitude lies in
`longitudes` (degrees east), in Sverdrups. `mode` is `:fixed_residual` (ψ at `reference_depth` after
removing the section's net barotropic transport, smooth in the control) or `:fixed_depth` (no residual
correction).
"""
struct OverturningStrength{W, A, L, E}
    rows :: UnitRange{Int}  # the rows the section visits
    v_weight :: W           # signed v-face areas, (Nx, length(rows), Nz), m²
    u_weight :: W           # signed u-face areas, (Nx, length(rows), Nz), m²
    section_area :: A       # unsigned face area per level, (Nz,), m²
    cumulative :: L         # lower-triangular ones: (L * x)[k] = Σ_{k' ≤ k} x[k']
    level_selector :: E     # one-hot vector at the reference level
    mode :: Symbol
    latitude :: Float64
    reference_depth :: Float64
end

const overturning_modes = (:fixed_depth, :fixed_residual)

# v-face row of column i closest to `latitude`.
nearest_face_row(grid, i, latitude) =
    argmin([abs(φnode(i, j, 1, grid, Center(), Face(), Center()) - latitude) for j in 1:size(grid, 2)])

# Signed face areas of the staircase section on a CPU grid, over all rows (trimmed by the caller).
function section_face_weights(cpu_grid, latitude, longitudes)
    Nx, Ny, Nz = size(cpu_grid)
    face_row = [nearest_face_row(cpu_grid, i, latitude) for i in 1:Nx]
    in_span = [longitudes[1] <= mod(λnode(i, face_row[i], 1, cpu_grid, Center(), Face(), Center()), 360) <= longitudes[2]
               for i in 1:Nx]
    Wv = zeros(Float64, Nx, Ny, Nz)
    Wu = zeros(Float64, Nx, Ny, Nz)
    for k in 1:Nz, i in 1:Nx
        in_span[i] || continue
        j = face_row[i]
        wet = !inactive_node(i, j, k, cpu_grid, Center(), Face(), Center())
        Wv[i, j, k] = wet * Δxᶜᶠᶜ(i, j, k, cpu_grid) * Δzᶜᶠᶜ(i, j, k, cpu_grid)

        # Step to column i + 1 along the u-faces at index i + 1 (between centers i and i + 1): the cells
        # between the two face rows are in S on one side only, and the flux out of S points away from it.
        i < Nx && in_span[i + 1] || continue
        j₁, j₂ = face_row[i], face_row[i + 1]
        sign = j₁ < j₂ ? -1 : 1
        for j in min(j₁, j₂):max(j₁, j₂) - 1
            wet = !inactive_node(i + 1, j, k, cpu_grid, Face(), Center(), Center())
            Wu[i + 1, j, k] = sign * wet * Δyᶠᶜᶜ(i + 1, j, k, cpu_grid) * Δzᶠᶜᶜ(i + 1, j, k, cpu_grid)
        end
    end
    return Wv, Wu
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
    Wv, Wu = section_face_weights(cpu_grid, latitude, longitudes)
    visited = findall(j -> any(!iszero, view(Wv, :, j, :)) || any(!iszero, view(Wu, :, j, :)), axes(Wv, 2))
    isempty(visited) && throw(ArgumentError("no wet face at latitude $latitude within longitudes $longitudes"))
    rows = first(visited):last(visited)
    section_area = vec(sum(abs, Wv[:, rows, :]; dims = (1, 2)) .+ sum(abs, Wu[:, rows, :]; dims = (1, 2)))
    cumulative = Float64[k2 <= k1 for k1 in 1:Nz, k2 in 1:Nz]
    z_centers = [znode(1, 1, k, cpu_grid, Center(), Center(), Center()) for k in 1:Nz]
    k_ref = argmin(abs.(z_centers .+ reference_depth))
    level_selector = Float64[k == k_ref for k in 1:Nz]
    return OverturningStrength(rows, Wv[:, rows, :], Wu[:, rows, :], section_area, cumulative, level_selector,
                               mode, Float64(latitude), Float64(reference_depth))
end

# Transport per level → ψ (Sv) at the reference level. The residual mode removes a uniform normal
# velocity through the section, so ψ closes at the surface instead of carrying the open section's
# net through-flow.
function overturning(obj::OverturningStrength, u, v)
    transport = vec(sum(obj.v_weight .* v .+ obj.u_weight .* u; dims = (1, 2)))   # (Nz,) m³ s⁻¹
    A = obj.section_area
    T = obj.mode === :fixed_residual ? (transport .- (sum(transport) / sum(A)) .* A) : transport
    return sum(obj.level_selector .* -(obj.cumulative * T)) / 1e6
end

section_velocities(obj, model) = map(f -> interior(f, :, obj.rows, :), ocean_model(model).velocities[(:u, :v)])

# Traced functor on the coupled model.
(obj::OverturningStrength)(model) = overturning(obj, section_velocities(obj, model)...)

"""
$(TYPEDSIGNATURES)

Host-side evaluation of `obj` on `model` (pulls the section to the CPU first).
"""
objective_host(obj::OverturningStrength, model) = overturning(obj, map(Array, section_velocities(obj, model))...)

ocean_model(model) = model.ocean.model
