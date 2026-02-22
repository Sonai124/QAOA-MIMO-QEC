from __future__ import annotations

import numpy as np


def repetition_encode(bits: np.ndarray, nrep: int) -> np.ndarray:
    """Repeat each bit nrep times (time-repetition)."""
    b = np.asarray(bits, dtype=int).ravel()
    if nrep < 1:
        raise ValueError("nrep must be >= 1")
    return np.repeat(b, nrep)


def repetition_decode_majority(bits_repeated: np.ndarray, nrep: int) -> np.ndarray:
    """Decode repetition code by majority vote over blocks of size nrep."""
    r = np.asarray(bits_repeated, dtype=int).ravel()
    if r.size % nrep != 0:
        raise ValueError("Length must be divisible by nrep")
    blocks = r.reshape(-1, nrep)
    # majority vote per block
    return (np.sum(blocks, axis=1) > (nrep // 2)).astype(int)
