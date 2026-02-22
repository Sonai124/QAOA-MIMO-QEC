from __future__ import annotations

from typing import Optional
import numpy as np

from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import StatevectorSampler
from qiskit_optimization.algorithms import MinimumEigenOptimizer


def make_qaoa_mimo_solver(maxiter: int = 40, reps: int = 1, seed: Optional[int] = 1234) -> MinimumEigenOptimizer:
    sampler = StatevectorSampler(seed=seed)
    qaoa_mes = QAOA(
        sampler=sampler,
        optimizer=COBYLA(maxiter=maxiter),
        reps=reps,
    )
    return MinimumEigenOptimizer(qaoa_mes)


def solve_binary_qp(solver: MinimumEigenOptimizer, qp) -> np.ndarray:
    res = solver.solve(qp)
    return res.samples[0].x.astype(int)
