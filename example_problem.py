import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


from mpi4py import MPI
from dolfinx import mesh

from dolfinx import fem
from petsc4py.PETSc import ScalarType
import ufl


def create_problem(N):

    domain = mesh.create_unit_square(MPI.COMM_SELF, N, N)

    V = fem.FunctionSpace(domain, ("CG", 1))

    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)

    boundary_facets = mesh.exterior_facet_indices(domain.topology)

    boundary_dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
    bc = fem.dirichletbc(ScalarType(0), boundary_dofs, V)


    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)

    f = fem.Constant(domain, ScalarType(1))

    kappa = 1.0 + 0.5*ufl.sin(x[0]**2 + x[1]**2)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)


    a = kappa * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = f * v * ufl.dx

    problem = fem.petsc.LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
    

    return problem, domain, V


from bindings.petsc_to_scipy import unpack_problem

N = 21
problem, domain, V = create_problem(N)
A_csr, b = unpack_problem(problem)

M = np.zeros((b.shape[0], b.shape[0]+1))
M[:,:-1] = A_csr.todense()
M[:,-1] = b
ims = plt.imshow(M)
plt.colorbar(ims)

plt.figure()
plt.spy(M)


u_np = sp.sparse.linalg.spsolve(A_csr, b)
xx = domain.geometry.x

ax = plt.figure().add_subplot(projection='3d')
ax.plot_trisurf(xx[:,0], xx[:,1], u_np, cmap=plt.cm.viridis)



plt.show()

