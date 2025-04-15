import numpy as np
import geojson
import rasterio
import xarray
import firedrake
from firedrake import assemble, exp, ln, inner, grad, dx, ds, Constant
import firedrake.adjoint
import icepack
import icepack2
from icepack2.model import minimization as model

# Make a mesh
outline_filename = "kangerlussuaq1.geojson"
with open(outline_filename, "r") as outline_file:
    outline = geojson.load(outline_file)

gmsh_mesh = icepack.meshing.collection_to_gmsh(outline)
gmsh_mesh.write("kangerlussuaq1.msh", verbose=False)
mesh = firedrake.Mesh("kangerlussuaq1.msh")

# Create some function spaces
cg1 = firedrake.FiniteElement("CG", "triangle", 1)
dg0 = firedrake.FiniteElement("DG", "triangle", 0)
Q = firedrake.FunctionSpace(mesh, cg1)
V = firedrake.VectorFunctionSpace(mesh, cg1)
Σ = firedrake.TensorFunctionSpace(mesh, dg0, symmetry=True)
T = firedrake.VectorFunctionSpace(mesh, dg0)
Z = V * Σ * T

# Read in the thickness + elevation data
bedmachine_filename = icepack.datasets.fetch_bedmachine_greenland()
bedmachine = xarray.open_dataset(bedmachine_filename)
h = icepack.interpolate(bedmachine["thickness"], Q)
s = icepack.interpolate(bedmachine["surface"], Q)

# Read in the velocity data
measures_filenames = icepack.datasets.fetch_measures_greenland()
vx_filename = [f for f in measures_filenames if "vx" in f][0]
vy_filename = [f for f in measures_filenames if "vy" in f][0]
ex_filename = [f for f in measures_filenames if "ex" in f][0]
ey_filename = [f for f in measures_filenames if "ey" in f][0]

with (
    rasterio.open(vx_filename, "r") as vx_file,
    rasterio.open(vy_filename, "r") as vy_file,
    rasterio.open(ex_filename, "r") as ex_file,
    rasterio.open(ey_filename, "r") as ey_file,
):
    u_obs = icepack.interpolate((vx_file, vy_file), V)
    σx = icepack.interpolate(ex_file, Q)
    σy = icepack.interpolate(ey_file, Q)

# Check and make sure there's no missing data
assert u_obs.dat.data_ro.min() > -10e3

# Compute an initial estimate for the ice velocity
T = Constant(260.0)
A = icepack.rate_factor(T)

ρ_I = Constant(icepack2.constants.ice_density)
g = Constant(icepack2.constants.gravity)
τ = firedrake.project(-ρ_I * g * h * grad(s), V)
area = assemble(Constant(1) * dx(mesh))
u_avg = np.sqrt(assemble(inner(u_obs, u_obs) * dx) / area)
τ_avg = np.sqrt(assemble(inner(τ, τ) * dx) / area)

m = Constant(icepack2.constants.weertman_sliding_law)
n = Constant(icepack2.constants.glen_flow_law)

K = Constant(u_avg / τ_avg ** icepack2.constants.weertman_sliding_law)

τ_c = Constant(0.1)
ε_c = Constant(A * τ_c ** n)
u_c = Constant(K * τ_c ** m)

z = firedrake.Function(Z)
z.sub(0).assign(u_obs)

u, M, τ = firedrake.split(z)
fields = {
    "velocity": u,
    "membrane_stress": M,
    "basal_stress": τ,
    "thickness": h,
    "surface": s,
}


α = Constant(0.01)
linear_rheology = {
    "flow_law_exponent": 1,
    "flow_law_coefficient": α * ε_c / τ_c,
    "sliding_exponent": 1,
    "sliding_coefficient": α * u_c / τ_c,
}

glen_rheology = {
    "flow_law_exponent": n,
    "flow_law_coefficient": ε_c / τ_c**n,
    "sliding_exponent": m,
    "sliding_coefficient": u_c / τ_c**m,
}

L = (
    model.viscous_power(**fields, **linear_rheology) +
    model.viscous_power(**fields, **glen_rheology) +
    model.friction_power(**fields, **linear_rheology) +
    model.friction_power(**fields, **glen_rheology) +
    model.momentum_balance(**fields)
)
F = firedrake.derivative(L, z)

boundary_ids = [1, 2, 3, 4]
bc = firedrake.DirichletBC(Z.sub(0), u_obs, boundary_ids)

qdegree = 6
problem_params = {
    "form_compiler_parameters": {"quadrature_degree": qdegree},
    "bcs": bc,
}

solver_params = {
    "solver_parameters": {
        "snes_monitor": None,
        "snes_stol": 0.0,
        "snes_max_it": 200,
        "snes_divergence_tolerance": 1e20,
        "snes_type": "newtonls",
        "snes_linesearch_type": "nleqerr",
        "ksp_type": "gmres",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
    },
}

problem = firedrake.NonlinearVariationalProblem(F, z, **problem_params)
solver = firedrake.NonlinearVariationalSolver(problem, **solver_params)

num_continuation_steps = 5
λs = np.linspace(0.0, 1.0, num_continuation_steps)
for λ in λs:
    n.assign((1 - λ) + λ * icepack2.constants.glen_flow_law)
    m.assign((1 - λ) + λ * icepack2.constants.weertman_sliding_law)
    solver.solve()
