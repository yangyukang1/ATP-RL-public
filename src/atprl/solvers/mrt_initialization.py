from __future__ import annotations

import numpy as np
from atprl.solvers.optimization_context import OptimizationSnapshot, wrap_phase


def initialize_grouped_mrt(snapshot: OptimizationSnapshot) -> np.ndarray:
    direct, cascade = snapshot.direct_hat, snapshot.cascade_hat
    reference = np.ones(snapshot.k, dtype=complex)
    for k in range(snapshot.k):
        scale = max(abs(direct[k]), np.max(np.abs(cascade[k])))
        if scale == 0:
            continue
        value = direct[k]
        if abs(value) <= scale * 1e-12:
            value = cascade[k].sum()
        if abs(value) <= scale * 1e-12:
            value = cascade[k, np.argmax(np.abs(cascade[k]))]
        reference[k] = value / abs(value)
    coherent = np.mean(reference.conj()[:, None] * cascade, axis=0)
    return wrap_phase(-np.angle(coherent))
