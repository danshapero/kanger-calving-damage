import numpy as np
import xarray
import firedrake
from firedrake import assemble, Constant, max_value, exp, inner, grad, dx, ds, dS
from petsc4py import PETSc
import icepack
import icepack2
from icepack2.constants import (
    glen_flow_law as n,
    weertman_sliding_law as m,
    ice_density as ρ_I,
    water_density as ρ_W,
)

options = PETSc.Options()
input_filename = options.getString("input", "kangerlussuaq-initial.h5")
output_filename = options.getString("output", "kangerlussuaq-simulation.h5")

with firedrake.CheckpointFile(input_filename, "r") as chk:
    mesh = chk.load_mesh()
    u = chk.load_function(mesh, name="velocity")
    q = chk.load_function(mesh, name="log_friction")
    τ_c = chk.h5pyfile.attrs["mean_stress"]
    u_c = chk.h5pyfile.attrs["mean_speed"]

degree = q.ufl_element().degree()
S = q.function_space()
Q = firedrake.FunctionSpace(mesh, "DG", degree)
V = u.function_space()
Σ = firedrake.TensorFunctionSpace(mesh, "DG", degree - 1, symmetry=True)
T = firedrake.VectorFunctionSpace(mesh, "DG", degree - 1)
Z = V * Σ * T

u_in = u.copy(deepcopy=True)

z = firedrake.Function(Z)
z.sub(0).assign(u_in)

# Read in the thickness and bed data
bedmachine_filename = icepack.datasets.fetch_bedmachine_greenland()
bedmachine = xarray.open_dataset(bedmachine_filename)
b = icepack.interpolate(bedmachine["bed"], S)
h = firedrake.project(icepack.interpolate(bedmachine["thickness"], S), Q)
s = firedrake.project(icepack.interpolate(bedmachine["surface"], S), Q)

# Do a continuation method for the initial velocity solve
A = icepack.rate_factor(Constant(260.0))
ε_c = Constant(A * τ_c ** n)
print(f"Critical strain rate: {float(ε_c):.3f}")

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

problem = firedrake.NonlinearVariationalProblem(F, z, J=J_r, **problem_params)
solver = firedrake.NonlinearVariationalSolver(problem, **solver_params)
solver.solve()

u, M, τ = z.subfunctions
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.set_aspect("equal")
colors = firedrake.tripcolor(u, axes=ax)
fig.colorbar(colors)
plt.show()
