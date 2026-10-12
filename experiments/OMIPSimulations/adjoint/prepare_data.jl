# %%
# Download the datasets the differentiable OMIP test needs, from a node with internet access
# (cluster login nodes; compute nodes have none): ETOPO2022 bathymetry, WOA annual temperature and
# salinity, and the JRA55 repeat-year atmosphere and radiation variables. Building the CPU model once
# triggers every download through NumericalEarth's own cache.
#
#   julia --project=adjoint adjoint/prepare_data.jl [config.json]

using OMIPAdjoint
using Oceananigans
using JSON

ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"   # the WOA download goes through DataDeps, which otherwise prompts

config_path = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "configs", "twodegree_nz20_1step.json")
P = JSON.parsefile(config_path)

setup = omip_adjoint_setup(CPU(); Nz = P["Nz"], Δt = Float64(P["dt"]), substeps = P["substeps"],
                           n_atmosphere_slices = P["n_atmosphere_slices"])
println("Data in place; CPU model built on ", summary(setup.model.ocean.model.grid))
