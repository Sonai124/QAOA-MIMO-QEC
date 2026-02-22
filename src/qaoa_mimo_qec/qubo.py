from __future__ import annotations

import numpy as np
from docplex.mp.model import Model
from qiskit_optimization.translators import from_docplex_mp


def create_mimo_qubo(mu: np.ndarray, sigma: np.ndarray):
    """Create a binary QuadraticProgram from mu and sigma.

    Objective:
        minimize sum_i mu_i x_i + sum_{i,j} sigma_{i,j} x_i x_j

    This matches the structure of your original prototype.
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    n = mu.size

    mdl = Model()
    x = [mdl.binary_var(f"x{i}") for i in range(n)]

    objective = mdl.sum(mu[i] * x[i] for i in range(n))
    objective += mdl.sum(sigma[i, j] * x[i] * x[j] for i in range(n) for j in range(n))

    mdl.minimize(objective)
    return from_docplex_mp(mdl)
