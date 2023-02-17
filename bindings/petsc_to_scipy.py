from dolfinx import fem
import scipy as sp

# import petsc4py
# import petsc4py.lib
from petsc4py import PETSc

def initialize_problem(problem: fem.petsc.LinearProblem):
    """ The matrix and vector are not initialized after defining a fem.petsc.LinearProblem, 
        here following the implementation in the source code for fem.petsc.LinearProblem.solve() """
    
    
    # Assemble left-hand-side
    problem._A.zeroEntries()
    fem.petsc._assemble_matrix_mat(problem._A, problem._a, bcs=problem.bcs)
    problem._A.assemble()

    # Assemble right-hand-side
    with problem._b.localForm() as b_loc:
        b_loc.set(0)
    fem.petsc.assemble_vector(problem._b, problem._L)



    # Apply boundary conditions to the rhs
    fem.petsc.apply_lifting(problem._b, [problem._a], bcs=[problem.bcs])
    problem._b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    fem.petsc.set_bc(problem._b, problem.bcs)

    # These are not included in the fem.petsc.LinearProblem.solve() source code, but are included for certainty.
    # problem._b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    # problem._A.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

    # Add a tag to check if initialized
    problem.__my_initialized = True

    return


def problem_to_csr_matrix(problem: fem.petsc.LinearProblem):

    assert problem.__my_initialized

    # https://fenicsproject.discourse.group/t/converting-to-scipy-sparse-matrix-without-eigen-backend/847
    A_csr = sp.sparse.csr_matrix(problem.A.getValuesCSR()[::-1])
    
    return A_csr

def problem_to_np_vector(problem: fem.petsc.LinearProblem):

    assert problem.__my_initialized

    return problem.b.array


def unpack_problem(problem: fem.petsc.LinearProblem):

    initialize_problem(problem)

    A = problem_to_csr_matrix(problem)
    b = problem_to_np_vector(problem)

    return A, b

    
