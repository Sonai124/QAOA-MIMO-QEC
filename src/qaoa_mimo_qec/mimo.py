from __future__ import annotations

import numpy as np


def mimo_channel(H: np.ndarray, x: np.ndarray, noise_std: float, rng: np.random.Generator) -> np.ndarray:
    """y = H x + n with n ~ N(0, noise_std^2 I)."""
    H = np.asarray(H, dtype=float)
    x = np.asarray(x, dtype=float)
    n = rng.normal(0.0, noise_std, size=(H.shape[0],))
    return H @ x + n


def snr_db_to_noise_std(snr_db: float) -> float:
    """Map SNR(dB) -> noise std.

    This is a simple mapping used for benchmarking; you can adapt it to your comms conventions.
    For compatibility with your prototype that used var = 1/SNR and sampled N(0,var), we use:
        SNR_lin = 10^(snr_db/10)
        std = 1 / sqrt(SNR_lin)
    """
    snr_lin = 10.0 ** (snr_db / 10.0)
    return 1.0 / np.sqrt(snr_lin)


def bit_error_rate(x_hat: np.ndarray, x_true: np.ndarray) -> float:
    x_hat = np.asarray(x_hat, dtype=int).ravel()
    x_true = np.asarray(x_true, dtype=int).ravel()
    if x_hat.size != x_true.size:
        raise ValueError("x_hat and x_true must have the same length")
    return float(np.mean(x_hat != x_true))
