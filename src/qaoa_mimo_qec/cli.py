from __future__ import annotations

import argparse
import json
import numpy as np

from .experiments import run_mimo_qec_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="QAOA MIMO detection + repetition-code BER benchmark")

    parser.add_argument(
        "--snr-db",
        type=float,
        nargs="+",
        default=[0, 2, 4, 6, 8, 10],
        help="SNR points in dB",
    )
    parser.add_argument("--experiments", type=int, default=200, help="Monte Carlo messages per SNR")
    parser.add_argument("--maxiter", type=int, default=40, help="COBYLA max iterations")
    parser.add_argument("--reps", type=int, default=1, help="QAOA depth")
    parser.add_argument("--nrep", type=int, default=3, help="Repetition factor (odd recommended)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bpsk-pm1", action="store_true", help="Use BPSK symbols {-1,+1} instead of {0,1}")

    args = parser.parse_args()

    # 3x3 MIMO system
    H = np.array(
        [
            [1.24155, -0.174105, 0.332349],
            [-0.080418, -1.51301, 0.321184],
            [-1.7771, 1.55398, 0.23342],
        ],
        dtype=float,
    )

    res = run_mimo_qec_benchmark(
        H=H,
        snr_db_values=args.snr_db,
        experiments=args.experiments,
        maxiter=args.maxiter,
        reps=args.reps,
        nrep=args.nrep,
        seed=args.seed,
        use_bpsk_pm1=args.bpsk_pm1,
    )

    for row in res:
        print(json.dumps(row))


if __name__ == "__main__":
    main()
