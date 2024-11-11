import numpy as np
import subprocess
import geojson
import rasterio
import xarray
import firedrake
from firedrake import Constant, assemble, exp, ln, sqrt, inner, grad, dx
from petsc4py import PETSc
import icepack
from icepack.constants import (
    ice_density as ρ_I, gravity as g, weertman_sliding_law as m
)

options = PETSc.Options()
outline_filename = options.getString("outline", "kangerlussuaq1.geojson")
output_filename = options.getString("output", "kangerlussuaq-friction.h5")
regularization = options.getReal("regularization", 2.5e3)
refinement = options.getInt("refinement", 1)
degree = options.getInt("degree", 1)

# Fetch the glacier outline, generate mesh, and create function spaces
with open(outline_filename, "r") as outline_file:
    outline = geojson.load(outline_file)

geometry = icepack.meshing.collection_to_geo(outline)
geometry_filename = outline_filename.replace("geojson", "geo")
with open(outline_filename.replace("geojson", "geo"), "w") as geometry_file:
    geometry_file.write(geometry.get_code())

mesh_filename = outline_filename.replace("geojson", "msh")
command = f"gmsh -2 -v 0 -o {mesh_filename} {geometry_filename}"
subprocess.run(command.split())

coarse_mesh = firedrake.Mesh(mesh_filename)
mesh_hierarchy = firedrake.MeshHierarchy(coarse_mesh, refinement)
mesh = mesh_hierarchy[-1]
Q = firedrake.FunctionSpace(mesh, "CG", degree)
V = firedrake.VectorFunctionSpace(mesh, "CG", degree)

# Compute a bounding box for the spatial domain
coords = np.array(list(geojson.utils.coords(outline)))
delta = 2.5e3
extent = {
    "left": coords[:, 0].min() - delta,
    "right": coords[:, 0].max() + delta,
    "bottom": coords[:, 1].min() - delta,
    "top": coords[:, 1].max() + delta,
}

# Read in the elevation and thickness data and interpolate it to the mesh
bedmachine_filename = icepack.datasets.fetch_bedmachine_greenland()
bedmachine = xarray.open_dataset(bedmachine_filename)
h = icepack.interpolate(bedmachine["thickness"], Q)
s = icepack.interpolate(bedmachine["surface"], Q)

# Read in the velocity data and interpolate it to the mesh
measures_filenames = icepack.datasets.fetch_measures_greenland()
velocity_data = {}
for key in ["vx", "vy", "ex", "ey"]:
    filename = [f for f in measures_filenames if key in f][0]
    with rasterio.open(filename, "r") as source:
        window = rasterio.windows.from_bounds(
            **extent, transform=source.transform
        ).round_lengths().round_offsets()
        xmin, ymin, xmax, ymax = source.window_bounds(window)
        transform = source.window_transform(window)
        velocity_data[key] = source.read(indexes=1, window=window)

no_data = -2e9
for key in ["vx", "vy"]:
    val = velocity_data[key]
    val[val == no_data] = 0.0

for key in ["ex", "ey"]:
    val = velocity_data[key]
    val[val == no_data] = 100e3

ny, nx = velocity_data["vx"].shape
xs = np.linspace(xmin, xmax, nx)
ys = np.linspace(ymin, ymax, ny)
kw = {"dims": ("y", "x"), "coords": {"x": xs, "y": ys}}
vx = xarray.DataArray(np.flipud(velocity_data["vx"]), **kw)
vy = xarray.DataArray(np.flipud(velocity_data["vy"]), **kw)
ex = xarray.DataArray(np.flipud(velocity_data["ex"]), **kw)
ey = xarray.DataArray(np.flipud(velocity_data["ey"]), **kw)

u_obs = icepack.interpolate((vx, vy), V)
u_init = u_obs.copy(deepcopy=True)
u = u_obs.copy(deepcopy=True)

σx = icepack.interpolate(ex, Q)
σy = icepack.interpolate(ey, Q)
P = firedrake.Function(Q).interpolate(1.0 / firedrake.sqrt(σx**2 + σy**2))

# Make an initial estimate for the basal friction by assuming it supports some
# fraction of the driving stress
τ = firedrake.project(-ρ_I * g * h * grad(s), V)

area = assemble(Constant(1) * dx(mesh))
u_avg = assemble(sqrt(inner(u, u)) * dx) / area
τ_avg = assemble(sqrt(inner(τ, τ)) * dx) / area

frac = Constant(0.5)
C = frac * sqrt(inner(τ, τ)) / sqrt(inner(u, u)) ** (1 / m)
q = firedrake.Function(Q).interpolate(-ln(u_avg ** (1 / m) * C / τ_avg))

def bed_friction(**kwargs):
    u, q = map(kwargs.get, ("velocity", "log_friction"))
    C = Constant(τ_avg) / Constant(u_avg) ** (1 / m) * exp(-q)
    return icepack.models.friction.bed_friction(velocity=u, friction=C)

# Compute an initial estimate for the ice velocity
T = firedrake.Constant(260.0)
A = icepack.rate_factor(T)

flow_model = icepack.models.IceStream(friction=bed_friction)
opts = {
    "dirichlet_ids": [1, 2, 3, 4],
    "diagnostic_solver_type": "petsc",
    "diagnostic_solver_parameters": {
        "snes_type": "newtontr",
        "snes_max_it": 100,
        "ksp_type": "gmres",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    },
}
flow_solver = icepack.solvers.FlowSolver(flow_model, **opts)
u_init = u.copy(deepcopy=True)
u = flow_solver.diagnostic_solve(
    velocity=u_init,
    thickness=h,
    surface=s,
    fluidity=A,
    log_friction=q,
)

# Estimate the basal friction coefficient
Ω = Constant(area)
α = Constant(regularization)

def simulation(q):
    fields = {"velocity": u_init, "thickness": h, "surface": s, "fluidity": A}
    return flow_solver.diagnostic_solve(**fields, log_friction=q)

def regularization(q):
    return 0.5 * α**2 / Ω * inner(grad(q), grad(q)) * dx

def loss_functional(u):
    return 0.5 * P**2 / Ω * inner(u - u_obs, u - u_obs) * dx

problem = icepack.statistics.StatisticsProblem(
    simulation=simulation,
    loss_functional=loss_functional,
    regularization=regularization,
    controls=q,
)

estimator = icepack.statistics.MaximumProbabilityEstimator(
    problem, gradient_tolerance=5e-5, step_tolerance=1e-8, max_iterations=50
)

q_optimal = estimator.solve()
u = simulation(q_optimal)

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.set_aspect("equal")
colors = firedrake.tripcolor(q_optimal, axes=ax)
fig.colorbar(colors)
plt.show()

# Save the results to disk
with firedrake.CheckpointFile(output_filename, "w") as chk:
    chk.save_function(q_optimal, name="log_friction")
    chk.save_function(u, name="velocity")
    chk.h5pyfile.attrs["mean_stress"] = τ_avg
    chk.h5pyfile.attrs["mean_speed"] = u_avg
