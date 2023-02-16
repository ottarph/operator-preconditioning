# Problem description

Solve a non-homogeneous laplace equation using operator preconditioning.

Solve
$$
    -\nabla \cdot (\kappa \nabla u) = f \, \in \Omega, \\
    u = g \, \in \partial \Omega, 
$$
where $\kappa$ is a smooth space-varying coefficient bounded from below.

This defines a problem 
$$
    \text{Find } u \in H^1_0 : \mathcal{A} u = l
$$
where
$$
    \mathcal{A}: H^1_0 \to H^{-1} = (H^1_0)^*, \, \mathcal{A}u(v) = a(u,v) = \int \kappa \nabla u \cdot \nabla v \, dx \, \forall v \in H^1_0
$$
and 
$$
    l \in H^{-1}, \, l(v) = \int f v \, dx.
$$

We solve 
$$
    \mathcal{A} u = b
$$
with some iterative solver,
e.g.
$$
    u^{k+1} = u^k + \mathcal{B}(\mathcal{A} u^k - b)
$$
with $\mathcal{B} \approx \mathcal{A}^{-1}$ a preconditioner.

We choose for the preconditioner the Riesz operator
$$
    B: H^{-1} \to H^1_0, \, f \mapsto g \text{ s.t. } -\Delta g = f, \, g|_{\partial \Omega} = 0,
$$
spectrally equivalent to $\mathcal{A}$.

We discretize $\mathcal{A}$ as $A$ using `fenicsx` and $\mathcal{B}$ using some multigrid cycle, e.g. from `pyAMG`.

