# %%
# ==============================================================================
# Checks of the adjoint-only damping on the ECCO4 prescribed-flux ocean.
#   cpu stage: the forward damping operator D on random fields is finite, leaves land unchanged, and does not
#              increase the variance of the wet interior (it is a stable diffusion step).
#   reactant stage: dot-product test of the compiled Dᵀ against the compiled D, ⟨D x, y⟩ = ⟨x, Dᵀ y⟩ over the
#              full arrays (halos included), for random x and y.
#
#   julia --project=adjoint adjoint/test_adjoint_damping.jl adjoint/configs/adjoint_damping_dot_test.json
#
# Config keys: n_months, Nz, dt, substeps, adjoint_viscosity and adjoint_diffusivity (m² s⁻¹), stages
# ("cpu,reactant"; the reactant stage needs a Linux Julia), backend (Reactant backend, "gpu" or "cpu"), seed.
# ==============================================================================

using OMIPAdjoint
using OMIPAdjoint: damping_scratch, damping_step!, compile_adjoint_damping, damped_fields, ocean_model
using Oceananigans
using Oceananigans.Architectures: ReactantState
using Oceananigans.Grids: inactive_node
using Reactant
using Enzyme: make_zero
using JSON
using LinearAlgebra: dot
using Printf
using Random
using Test

config_path = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "configs", "adjoint_damping_dot_test.json")
P = JSON.parsefile(config_path)
stages = Symbol.(strip.(split(P["stages"], ",")))
Δt = Float64(P["dt"])
setup_kwargs = (; Nz = P["Nz"], Δt, substeps = P["substeps"], n_months = P["n_months"])
damping_kwargs = (; viscosity = P["adjoint_viscosity"], diffusivity = P["adjoint_diffusivity"], Δt)

randomize!(fields, rng) = foreach(f -> copyto!(parent(f), randn(rng, size(parent(f)))), fields)
host_copies(fields) = map(f -> Array(parent(f)), fields)

# %%
##### Forward operator on CPU

if :cpu in stages
    setup = ecco_adjoint_setup(CPU(); setup_kwargs...)
    damping = AdjointDamping(setup.cpu_grid; damping_kwargs...)
    scratch = damping_scratch(ocean_model(setup.model))
    randomize!(scratch.fields, Xoshiro(P["seed"]))
    x = map(f -> Array(interior(f)), scratch.fields)
    damping_step!(scratch, damping)
    Dx = map(f -> Array(interior(f)), scratch.fields)

    grid = setup.cpu_grid
    Nx, Ny, Nz = size(grid)
    c, f = Center(), Face()
    locations = (u = (f, c, c), v = (c, f, c), T = (c, c, c), S = (c, c, c))
    @testset "forward damping operator" begin
        for name in keys(x)
            wet = [!inactive_node(i, j, k, grid, locations[name]...) for i in 1:Nx, j in 1:Ny, k in 1:Nz]
            ratio = sum(abs2, Dx[name][wet]) / sum(abs2, x[name][wet])
            @printf("%s: wet variance ratio %.4f, land unchanged %s\n", name, ratio, Dx[name][.!wet] == x[name][.!wet])
            @test all(isfinite, Dx[name])
            @test ratio <= 1
        end
    end
end

# %%
##### Dot-product test of the compiled adjoint

if :reactant in stages
    Reactant.set_default_backend(P["backend"])
    setup = ecco_adjoint_setup(ReactantState(); setup_kwargs...)
    model = setup.model
    damping = AdjointDamping(setup.cpu_grid; damping_kwargs...)
    rng = Xoshiro(P["seed"])

    scratch = damping_scratch(ocean_model(model))
    randomize!(scratch.fields, rng)
    x = host_copies(scratch.fields)
    forward = @compile damping_step!(scratch, damping)
    forward(scratch, damping)
    Dx = host_copies(scratch.fields)

    dmodel = make_zero(model)
    adjoint_fields = damped_fields(ocean_model(dmodel))
    randomize!(adjoint_fields, rng)
    y = host_copies(adjoint_fields)
    damp = compile_adjoint_damping(model, dmodel, damping, OMIPAdjoint.adjoint_compile_options())
    damp(dmodel)
    DTy = host_copies(adjoint_fields)

    lhs = sum(dot(Dx[n], y[n]) for n in keys(y))
    rhs = sum(dot(x[n], DTy[n]) for n in keys(y))
    relative = abs(lhs - rhs) / max(abs(lhs), abs(rhs))
    change = sqrt(sum(sum(abs2, DTy[n] .- y[n]) for n in keys(y)) / sum(sum(abs2, y[n]) for n in keys(y)))
    @printf("⟨D x, y⟩ = %.15e\n⟨x, Dᵀ y⟩ = %.15e\nrelative difference %.2e; ‖Dᵀ y - y‖ / ‖y‖ = %.3e\n",
            lhs, rhs, relative, change)
    @testset "adjoint damping dot product" begin
        @test isfinite(lhs) && isfinite(rhs)
        @test relative < 1e-12
    end
end
