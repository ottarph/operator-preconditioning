
import pyamg
import numpy as np
A = pyamg.gallery.poisson((500,500), format='csr')              # 2D Poisson problem on 500x500 grid
ml = pyamg.ruge_stuben_solver(A)                                # construct the multigrid hierarchy
print(ml)                                                       # print hierarchy information
b = np.random.rand(A.shape[0])                                  # pick a random right hand side
x = ml.solve(b, tol=1e-10)                                      # solve Ax=b to a tolerance of 1e-10
print("residual: ", np.linalg.norm(b-A*x))                      # compute norm of residual vector

x = ml.solve(b, maxiter=1)
print("residual (one cycle): ", np.linalg.norm(b-A*x))          # compute norm of residual vector after only one cycle

print("residual (zero cycles): ", np.linalg.norm(b))            # compute norm of initial residual vector
