import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from mpi4py import MPI
from dolfinx import mesh

from dolfinx import fem
from petsc4py.PETSc import ScalarType
import ufl


def create_mesh(N):

    domain = mesh.create_unit_square(MPI.COMM_SELF, N, N)

    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)

    return domain

def create_riesz_problem(domain, V):
    """ Create the problem f \mapsto u s.t. -\Delta u = f in \Omega, u = 0 on \partial\Omega.
        Only the left-hand-side matrix portion is of interest here, for creating the preconditioner. """
    
    # Create homogeneous Dirichlet boundary conditions.
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    boundary_dofs = fem.locate_dofs_topological(V, domain.topology.dim-1, boundary_facets)
    bc = fem.dirichletbc(ScalarType(0), boundary_dofs, V)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    f = fem.Constant(domain, ScalarType(1))

    a = ufl.dot( ufl.grad(u), ufl.grad(v) ) * ufl.dx
    L = f * v * ufl.dx

    problem = fem.petsc.LinearProblem(a, L, bcs=[bc])

    return problem

def create_inhomogeneous_problem(domain, V, kappa, f):

    # Create homogeneous Dirichlet boundary conditions.
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    boundary_dofs = fem.locate_dofs_topological(V, domain.topology.dim-1, boundary_facets)
    bc = fem.dirichletbc(ScalarType(0), boundary_dofs, V)


    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)


    a = kappa * ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
    L = f * v * ufl.dx

    problem = fem.petsc.LinearProblem(a, L, bcs=[bc])
    return problem

N = 100
domain = create_mesh(N)
V = fem.FunctionSpace(domain, ("CG", 1))

x = ufl.SpatialCoordinate(domain)


f = fem.Constant(domain, ScalarType(1))
kappa = 6.0 + 5.0*ufl.sin(ufl.pi*( (x[0]-0.5)**2 + (x[1]-0.5)**2 ))


from bindings.petsc_to_scipy import unpack_problem


inh_problem = create_inhomogeneous_problem(domain, V, kappa, f)
A_csr, b = unpack_problem(inh_problem)

riesz_problem = create_riesz_problem(domain, V)
B_csr, _ = unpack_problem(riesz_problem)

if N <= 40:

    M = np.zeros((b.shape[0], 2*b.shape[0]+1))
    M[:,:b.shape[0]] = A_csr.todense()
    M[:,b.shape[0]:-1] = B_csr.todense()
    M[:,-1] = b
    ims = plt.imshow(M / np.amax(M))
    plt.colorbar(ims)

    plt.figure()
    plt.spy(M)


import pyamg

ml = pyamg.ruge_stuben_solver(B_csr)
B = ml.aspreconditioner(cycle='V')


class callback:
    def __init__(self, name="Solver"):
        self.k = 0
        self.name = name

    def __call__(self, xk):
        self.k += 1
        print(f"{self.name}: k={self.k}, ||r_k||={np.linalg.norm(A_csr@xk - b):.2e}")
        return

print("With preconditioning: ")
u_np, info = sp.sparse.linalg.cg(A_csr, b, tol=1e-8, maxiter=30, M=B, callback=callback("CG w/ AMG precond."))
print("Error =", np.linalg.norm(A_csr @ u_np - b))
print()

print("Without preconditioning: ")
u_np_wp, info = sp.sparse.linalg.cg(A_csr, b, tol=1e-8, maxiter=30, M=None, callback=callback("CG w/ no precond."))
print("Error =", np.linalg.norm(A_csr @ u_np_wp - b))
print()

xx = domain.geometry.x

step = 1
if N > 40:
    step = N**2 // 1000

ax = plt.figure().add_subplot(projection='3d')
ax.plot_trisurf(xx[::step,0], xx[::step,1], u_np[::step], cmap=plt.cm.viridis)



plt.show()

