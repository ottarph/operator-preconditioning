import numpy as np
import matplotlib.pyplot as plt


from mpi4py import MPI
from dolfinx import mesh

from dolfinx import fem
from petsc4py.PETSc import ScalarType
import ufl

def run_problem(N, mu_val, SUPG=True):

    domain = mesh.create_unit_interval(MPI.COMM_SELF, N)

    V = fem.FunctionSpace(domain, ("CG", 1))


    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)

    leftfacets = mesh.locate_entities_boundary(domain, dim=0,
                                        marker=lambda x: np.isclose(x[0], 0.0))
    rightfacets = mesh.locate_entities_boundary(domain, dim=0,
                                        marker=lambda x: np.isclose(x[0], 1.0))

    left_bc_dofs = fem.locate_dofs_topological(V=V, entity_dim=0, entities=leftfacets)
    right_bc_dofs = fem.locate_dofs_topological(V=V, entity_dim=0, entities=rightfacets)

    left_bc = fem.dirichletbc(ScalarType(0), left_bc_dofs, V)
    right_bc = fem.dirichletbc(ScalarType(1), right_bc_dofs, V)


    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    f = fem.Constant(domain, ScalarType(0))

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)


    mu = fem.Constant(domain, ScalarType(mu_val))
    w = fem.Constant(domain, ScalarType((-1,)))

    if SUPG:       
        beta = 0.5
        h = 1 / (N-1)
        vv = v + beta*h * ufl.dot(w, ufl.grad(v))
    else:
        vv = v

    a = mu * ufl.dot(ufl.grad(u), ufl.grad(vv)) * ufl.dx + ufl.dot(ufl.grad(u), w) * vv * ufl.dx
    L = f * vv * ufl.dx

    problem = fem.petsc.LinearProblem(a, L, bcs=[left_bc, right_bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    uh = problem.solve()

    return uh

mu = 0.001
N = 21

def u_ex_func(x, mu):
    return ( np.exp(-x / mu) - 1 ) / ( np.exp(-1 / mu) - 1 )

uh_supg = run_problem(N, mu, SUPG=True)
xx_supg = np.copy(uh_supg.function_space.mesh.geometry.x[:,0])
uu_supg = np.copy(uh_supg.vector.array)

uh_cg = run_problem(N, mu, SUPG=False)
xx_cg = np.copy(uh_cg.function_space.mesh.geometry.x[:,0])
uu_cg = np.copy(uh_cg.vector.array)

xx_long = np.linspace(0, 1, 1001)
uu_ex_long = u_ex_func(xx_long, mu)

plt.figure()

plt.plot(xx_supg, uu_supg, 'k-', label=r"$u_\mathrm{supg}$")
plt.plot(xx_cg, uu_cg, 'k--', label=r"$u_\mathrm{cg}$")
plt.plot(xx_long, uu_ex_long, 'k:', label=r"$u_\mathrm{ex}$")

plt.legend()

plt.show()
