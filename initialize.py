import numpy as np
import subprocess
import geojson
import rasterio
import xarray
import firedrake
from firedrake import Constant, assemble, max_value, exp, inner, grad, dx
import icepack
import icepack2
from icepack2.constants import (
    ice_density as ρ_I,
    water_density as ρ_W,
    weertman_sliding_law as m,
    glen_flow_law as n,
)

# Fetch the glacier outline, generate mesh, and create function spaces
outline_filename = "kangerlussuaq.geojson"
with open(outline_filename, "r") as outline_file:
    outline = geojson.load(outline_file)

geometry = icepack.meshing.collection_to_geo(outline)
with open("kangerlussuaq.geo", "w") as geometry_file:
    geometry_file.write(geometry.get_code())

command = "gmsh -2 -v 0 -o kangerlussuaq.msh kangerlussuaq.geo"
subprocess.run(command.split())

mesh = firedrake.Mesh("kangerlussuaq.msh")
S = firedrake.FunctionSpace(mesh, "CG", 1)
Q = firedrake.FunctionSpace(mesh, "DG", 1)
V = firedrake.VectorFunctionSpace(mesh, "CG", 1)
Σ = firedrake.TensorFunctionSpace(mesh, "DG", 0)
T = firedrake.VectorFunctionSpace(mesh, "DG", 0)
Z = V * Σ * T
z = firedrake.Function(Z)

# Compute a bounding box for the spatial domain
coords = np.array(list(geojson.utils.coords(outline)))
delta = 2.5e3
extent = {
    "left": coords[:, 0].min() - delta,
    "right": coords[:, 0].max() + delta,
    "bottom": coords[:, 1].min() - delta,
    "top": coords[:, 1].max() + delta,
}

# Read in some observational data
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

bedmachine_filename = icepack.datasets.fetch_bedmachine_greenland()
bedmachine = xarray.open_dataset(bedmachine_filename)

# Interpolate the data to the finite element mesh
u_obs = icepack.interpolate((vx, vy), V)
u_in = u_obs.copy(deepcopy=True)
b = icepack.interpolate(bedmachine["bed"], S)
h = firedrake.project(icepack.interpolate(bedmachine["thickness"], S), Q)
s = max_value(b + h, (1 - ρ_I / ρ_W) * h)

σx = icepack.interpolate(ex, Q)
σy = icepack.interpolate(ey, Q)
P = firedrake.Function(Q).interpolate(1.0 / firedrake.sqrt(σx**2 + σy**2))

# Do a continuation method for the initial velocity solve
A = icepack.rate_factor(Constant(260.0))
τ_c = Constant(0.1)
ε_c = Constant(A * τ_c ** n)
print(f"Critical strain rate: {float(ε_c):.3f}")
u_c = Constant(100.0)
q = firedrake.Function(Q)

fns = [
    icepack2.model.viscous_power,
    icepack2.model.friction_power,
    icepack2.model.momentum_balance,
]

u, M, τ = firedrake.split(z)
fields = {
    "velocity": u,
    "membrane_stress": M,
    "basal_stress": τ,
    "thickness": h,
    "surface": s,
}

h_min = Constant(1e-3)
rfields = {
    "velocity": u,
    "membrane_stress": M,
    "basal_stress": τ,
    "thickness": max_value(h_min, h),
    "surface": s,
}

rheology = {
    "flow_law_exponent": n,
    "flow_law_coefficient": ε_c / τ_c ** n,
    "sliding_exponent": m,
    "sliding_coefficient": u_c / τ_c ** m * exp(m * q),
}

linear_rheology = {
    "flow_law_exponent": 1,
    "flow_law_coefficient": ε_c / τ_c,
    "sliding_exponent": 1,
    "sliding_coefficient": u_c / τ_c * exp(q),
}

# Initial solve assuming linear rheology
L_r1 = sum(fn(**rfields, **linear_rheology) for fn in fns)
F_r1 = firedrake.derivative(L_r1, z)
J_r1 = firedrake.derivative(F_r1, z)

L_1 = sum(fn(**fields, **linear_rheology) for fn in fns)
F_1 = firedrake.derivative(L_1, z)
J_1 = firedrake.derivative(F_1, z)

inflow_ids = [1]
bc_in = firedrake.DirichletBC(Z.sub(0), u_in, inflow_ids)
outflow_ids = [2, 3, 4]
bc_out = firedrake.DirichletBC(Z.sub(0), Constant((0.0, 0.0)), outflow_ids)
bcs = [bc_in, bc_out]

problem_params = {}
solver_params = {
    "solver_parameters": {
        "snes_monitor": None,
        "snes_type": "newtonls",
        "snes_divergence_tolerance": 1e300,
        "snes_linesearch_type": "nleqerr",
        "ksp_type": "gmres",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "umfpack",
    },
}
lproblem = firedrake.NonlinearVariationalProblem(F_1, z, J=J_r1, **problem_params)
lsolver = firedrake.NonlinearVariationalSolver(lproblem, **solver_params)
lsolver.solve()

# Nonlinear solve
L_r = sum(fn(**rfields, **rheology) for fn in fns)
F_r = firedrake.derivative(L_r, z)
J_r = firedrake.derivative(F_r, z)

L = sum(fn(**fields, **rheology) for fn in fns)
F = firedrake.derivative(L, z)
J = firedrake.derivative(F, z)

firedrake.adjoint.continue_annotation()

problem = firedrake.NonlinearVariationalProblem(F, z, J=J_r, **problem_params)
solver = firedrake.NonlinearVariationalSolver(problem, **solver_params)
solver.solve()

import pyadjoint
from firedrake.adjoint import ReducedFunctional, Control
u, M, τ = firedrake.split(z)
area = assemble(Constant(1) * dx(mesh))
E = 0.5 * P**2 * inner(u - u_obs, u - u_obs) * dx
print(np.sqrt(assemble(E) / area))
α = Constant(5e3)
R = 0.5 * α**2 * inner(grad(q), grad(q)) * dx
controls = [Control(q)]
K = ReducedFunctional(assemble(E) + assemble(R), controls)
G = K.derivative()
firedrake.adjoint.pause_annotation()

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.set_aspect("equal")
colors = firedrake.tripcolor(z.subfunctions[0], axes=ax)
fig.colorbar(colors)
plt.show()
