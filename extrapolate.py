import subprocess
import geojson
import firedrake
from firedrake import Constant, inner, grad, dx
from petsc4py import PETSc
import icepack

options = PETSc.Options()
input_filename = options.getString("input", "kangerlussuaq-friction.h5")
outline_filename = options.getString("outline", "kangerlussuaq2.geojson")
output_filename = options.getString("output", "kangerlussuaq-initial.h5")

# Read in the estimated friction
with firedrake.CheckpointFile(input_filename, "r") as chk:
    input_mesh = chk.load_mesh()
    u_input = chk.load_function(input_mesh, name="velocity")
    q_input = chk.load_function(input_mesh, name="log_friction")
    τ_avg = chk.h5pyfile.attrs["mean_stress"]
    u_avg = chk.h5pyfile.attrs["mean_speed"]

Δ_input = firedrake.FunctionSpace(input_mesh, "DG", 0)
μ_input = firedrake.Function(Δ_input).interpolate(Constant(1))

# Create the larger mesh and some functio nspaces
with open(outline_filename, "r") as outline_file:
    outline = geojson.load(outline_file)

geometry_filename = outline_filename.replace("geojson", "geo")
geometry = icepack.meshing.collection_to_geo(outline)
with open(geometry_filename, "w") as geometry_file:
    geometry_file.write(geometry.get_code())

mesh_filename = outline_filename.replace("geojson", "msh")
command = f"gmsh -2 -v 0 -o {mesh_filename} {geometry_filename}"
subprocess.run(command.split())

mesh = firedrake.Mesh(mesh_filename)
degree = q_input.ufl_element().degree()
Q = firedrake.FunctionSpace(mesh, "CG", degree)
V = firedrake.VectorFunctionSpace(mesh, "CG", degree)

# Project the mask, log-fluidity, and velocity onto the larger mesh. The
# regions with no data will be extrapolated by zero.
Δ = firedrake.FunctionSpace(mesh, "DG", 0)
μ = firedrake.project(μ_input, Δ)

Eq = firedrake.project(q_input, Q)
Eu = firedrake.project(u_input, V)

q = Eq.copy(deepcopy=True)
u = Eu.copy(deepcopy=True)

# TODO: adjust this
α = Constant(5e2)

bc_ids = [1, 2, 4]
bc = firedrake.DirichletBC(V, Eu, bc_ids)
J = 0.5 * (μ * inner(u - Eu, u - Eu) + α**2 * inner(grad(u), grad(u))) * dx
F = firedrake.derivative(J, u)
firedrake.solve(F == 0, u, bc)

bc = firedrake.DirichletBC(Q, Eq, bc_ids)
J = 0.5 * (μ * inner(q - Eq, q - Eq) + α**2 * inner(grad(q), grad(q))) * dx
F = firedrake.derivative(J, q)
firedrake.solve(F == 0, q, bc)

# Write the results to disk
with firedrake.CheckpointFile(output_filename, "w") as chk:
    chk.save_mesh(mesh)
    chk.save_function(μ, name="mask")
    chk.save_function(q, name="log_friction")
    chk.save_function(u, name="velocity")
    chk.h5pyfile.attrs["mean_stress"] = τ_avg
    chk.h5pyfile.attrs["mean_speed"] = u_avg
