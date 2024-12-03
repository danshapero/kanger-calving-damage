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
    gravity as g,
)

options = PETSc.Options()
input_filename = options.getString("input", "kangerlussuaq-initial.h5")
output_filename = options.getString("output", "kangerlussuaq-simulation.h5")
final_time = options.getReal("final-time", 1.0)
timesteps_per_year = options.getInt("timesteps-per-year", 192)
critical_thickness = options.getReal("crit-thickness", 40.0)
hdegree = options.getInt("degree", 1)
min_thickness = options.getReal("min-thickness", 1e-3)
melt_rate = options.getReal("melt-rate", 500.0)

with firedrake.CheckpointFile(input_filename, "r") as chk:
    mesh = chk.load_mesh()
    u = chk.load_function(mesh, name="velocity")
    q = chk.load_function(mesh, name="log_friction")
    τ_c = chk.h5pyfile.attrs["mean_stress"]
    u_c = chk.h5pyfile.attrs["mean_speed"]

udegree = u.ufl_element().degree()
S = q.function_space()
Q = firedrake.FunctionSpace(mesh, "DG", hdegree)
V = u.function_space()
Σ = firedrake.TensorFunctionSpace(mesh, "DG", udegree - 1, symmetry=True)
T = firedrake.VectorFunctionSpace(mesh, "DG", udegree - 1)
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

h_min = Constant(min_thickness)
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
        "snes_max_it": 300,
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

u_problem = firedrake.NonlinearVariationalProblem(F, z, J=J_r, **problem_params)
u_solver = firedrake.NonlinearVariationalSolver(u_problem, **solver_params)
u_solver.solve()

u, M, τ = z.subfunctions

# Fix the accumulation rate. We used estimates of surface mass balance from the
# regional climate model MAR and remote sensing measurements of surface
# elevation to estimate a linear relationship:
#
#     SMB ~= da_ds * s + a_0
#
# where `da_ds` ~= 2.25 milimeters of water equivalent per year per meter
# elevation gain and `a_0` ~= -3.3 meters of water equivalent per year at sea
# level. We used all the data from 2006-2017 and the fit had `r² = 0.91`.
# See also https://www.climato.uliege.be/cms/c_5652668/fr/climato-greenland.
da_ds = Constant(2.25 * 1e-3)
a_0 = Constant(-3.3)
a = 0.917 * (a_0 + da_ds * s)

# Set up things we need for calving
h_min = firedrake.max_value(0, -ρ_W / ρ_I * b)
δh = Constant(critical_thickness)
m_0 = Constant(melt_rate)
m = m_0 * firedrake.max_value(0, h_min + δh - h) / δh

# Set up the mass balance equation
h_n = h.copy(deepcopy=True)
h0 = h.copy(deepcopy=True)
φ = firedrake.TestFunction(h.function_space())
dt = Constant(1.0 / timesteps_per_year)
flux_cells = ((h - h_n) / dt * φ - inner(h * u, grad(φ)) - (a - m) * φ) * dx
ν = firedrake.FacetNormal(mesh)
f = h * max_value(0, inner(u, ν))
flux_facets = (f("+") - f("-")) * (φ("+") - φ("-")) * dS
flux_in = h0 * firedrake.min_value(0, inner(u, ν)) * φ * ds
flux_out = h * max_value(0, inner(u, ν)) * φ * ds
G = flux_cells + flux_facets + flux_in + flux_out
h_problem = firedrake.NonlinearVariationalProblem(G, h)
h_solver = firedrake.NonlinearVariationalSolver(h_problem)

# Run the simulation
t = Constant(0.0)
h_c = Constant(5.0)
num_steps = int(final_time * timesteps_per_year) + 1

field_names = ["thickness", "velocity", "membrane_stress", "basal_stress"]
with firedrake.CheckpointFile(output_filename, "w") as chk:
    u, M, τ = z.subfunctions
    for field, name in zip([h, u, M, τ], field_names):
        chk.save_function(field, name=name, idx=0)

    timesteps = np.linspace(0.0, final_time, num_steps)
    for step in range(num_steps):
        t.assign(t + dt)

        h_solver.solve()
        h.interpolate(firedrake.conditional(h < h_c, 0, h))
        h_n.assign(h)
        s.interpolate(max_value(b + h, (1 - ρ_I / ρ_W) * h))
        u_solver.solve()

        for field, name in zip([h, u, M, τ], field_names):
            chk.save_function(field, name=name, idx=step + 1)

    chk.h5pyfile.create_dataset("timesteps", data=timesteps)
