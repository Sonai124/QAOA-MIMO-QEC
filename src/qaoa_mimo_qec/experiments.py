from __future__ import annotations

import time
from typing import Iterable, Dict, Any, List, Optional

import numpy as np

from .mimo import mimo_channel, snr_db_to_noise_std, bit_error_rate
from .qubo import create_mimo_qubo
from .solvers import make_qaoa_mimo_solver, solve_binary_qp
from .qec import repetition_encode, repetition_decode_majority


def run_mimo_qec_benchmark(
    H: np.ndarray,
    snr_db_values: Iterable[float] = (0, 2, 4, 6, 8, 10),
    experiments: int = 200,
    maxiter: int = 40,
    reps: int = 1,
    nrep: int = 3,
    seed: int = 42,
    use_bpsk_pm1: bool = False,
) -> List[Dict[str, Any]]:
    """Benchmark QAOA MIMO detection with and without repetition coding.

    Parameters
    ----------
    H:
        Channel matrix (Nr x Nt).
    use_bpsk_pm1:
        If True, maps bits {0,1} -> symbols {-1,+1} for transmission and inverse-maps at detection.
        If False, keeps {0,1} symbols to match your original prototype.

    Returns
    -------
    list of dict per SNR value:
        - ber_uncoded: BER directly after per-slot MIMO detection
        - ber_rep_majority: BER after repetition-majority decoding
        - avg_solve_time_s: average QAOA solve time per slot
    """
    H = np.asarray(H, dtype=float)
    nt = H.shape[1]

    rng = np.random.default_rng(seed)
    solver = make_qaoa_mimo_solver(maxiter=maxiter, reps=reps, seed=seed)

    # Precompute sigma = H^T H (constant if H fixed)
    sigma = H.T @ H

    results: List[Dict[str, Any]] = []

    for snr_db in snr_db_values:
        noise_std = snr_db_to_noise_std(float(snr_db))

        bit_errors_uncoded = 0
        total_bits_uncoded = 0

        bit_errors_coded = 0
        total_bits_coded = 0

        total_solve_time = 0.0
        total_slots = 0

        for _ in range(experiments):
            # Generate one Nt-bit vector per "message"
            msg_bits = rng.integers(0, 2, size=(nt,), dtype=int)

            # Encode with repetition across time
            coded_bits = repetition_encode(msg_bits, nrep=nrep)
            # Reshape into time slots (each slot transmits nt bits)
            coded_bits_slots = coded_bits.reshape(nrep, nt)

            detected_slots = []

            for t in range(nrep):
                bits_t = coded_bits_slots[t]

                if use_bpsk_pm1:
                    x_sym = 2 * bits_t.astype(float) - 1.0
                else:
                    x_sym = bits_t.astype(float)

                y = mimo_channel(H, x_sym, noise_std=noise_std, rng=rng)

                # mu = -2 y^T H  (matches your prototype)
                mu = -2.0 * (y.T @ H)

                qp = create_mimo_qubo(mu=mu, sigma=sigma)

                t0 = time.time()
                x_hat_bits = solve_binary_qp(solver, qp)
                total_solve_time += time.time() - t0
                total_slots += 1

                detected_slots.append(x_hat_bits)

                # Uncoded BER counts per-slot against sent bits
                bit_errors_uncoded += int(np.sum(x_hat_bits != bits_t))
                total_bits_uncoded += nt

            detected_slots = np.asarray(detected_slots, dtype=int)  # shape (nrep, nt)
            # Decode repetition per bit position
            decoded_bits = repetition_decode_majority(detected_slots.reshape(-1), nrep=nrep)

            bit_errors_coded += int(np.sum(decoded_bits != msg_bits))
            total_bits_coded += nt

        results.append(
            {
                "snr_db": float(snr_db),
                "experiments": experiments,
                "nt": nt,
                "nrep": nrep,
                "reps": reps,
                "maxiter": maxiter,
                "ber_uncoded": bit_errors_uncoded / total_bits_uncoded,
                "ber_rep_majority": bit_errors_coded / total_bits_coded,
                "avg_solve_time_s": total_solve_time / max(1, total_slots),
            }
        )

    return results
