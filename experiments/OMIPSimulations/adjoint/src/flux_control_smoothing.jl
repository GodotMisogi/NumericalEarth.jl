#####
##### Smoothed heat- and freshwater-flux controls
#####
# The flux controls are c with Q = Q₀ + S c, so ∂J/∂c = Sᵀ ∂J/∂Q. S is the cell-center analogue of the stress's
# center-to-face interpolation: per pass, a 1-2-1 average in i, then in j, over wet cells only.

# P (I − Lᵢ / 4) P for one surface slice: Lᵢ is the graph Laplacian of wet–wet edges in i (periodic), P zeroes land.
function average_in_i(c, wet)
    Nx, Ny = size(c)
    s = zero(c)
    for j in 1:Ny, i in 1:Nx
        wet[i, j] || continue
        e, w = mod1(i + 1, Nx), mod1(i - 1, Nx)
        s[i, j] = c[i, j] + (wet[e, j] * (c[e, j] - c[i, j]) + wet[w, j] * (c[w, j] - c[i, j])) / 4
    end
    return s
end

# As `average_in_i` in j, with no edges across the southern boundary or the tripolar fold (row Ny).
function average_in_j(c, wet)
    Nx, Ny = size(c)
    s = zero(c)
    for j in 1:Ny, i in 1:Nx
        wet[i, j] || continue
        north = j < Ny && wet[i, j + 1] ? c[i, j + 1] - c[i, j] : zero(eltype(c))
        south = j > 1 && wet[i, j - 1] ? c[i, j - 1] - c[i, j] : zero(eltype(c))
        s[i, j] = c[i, j] + (north + south) / 4
    end
    return s
end

function smooth_slices(c, wet, passes, transpose)
    s = copy(c)
    for n in axes(c, 4)
        x = c[:, :, 1, n]
        for _ in 1:passes
            x = transpose ? average_in_i(average_in_j(x, wet), wet) : average_in_j(average_in_i(x, wet), wet)
        end
        s[:, :, 1, n] .= x
    end
    return s
end

"""
$(TYPEDSIGNATURES)

Apply S, or Sᵀ with `transpose = true`, to the `heat_flux` and `freshwater_flux` entries of the named tuple `controls`
(host arrays `(Nx, Ny, 1, Nt)`); other entries pass through. S is `passes` passes of a 1-2-1 average in i (periodic)
and then in j over the cells where `wet` holds. Each average is symmetric: it uses only wet–wet edges, so land stays
zero and gives no weight. It removes the checker and both stripes in all-wet regions, and has no edges across
the southern boundary or the tripolar fold.
"""
function smooth_flux_controls(controls, wet; passes, transpose = false)
    passes == 0 && return controls
    smooth(name, c) = name in (:heat_flux, :freshwater_flux) ? smooth_slices(c, wet, passes, transpose) : c
    return NamedTuple{keys(controls)}(map(smooth, keys(controls), values(controls)))
end
