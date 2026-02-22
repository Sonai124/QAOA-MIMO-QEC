import numpy as np
from qaoa_mimo_qec.qec import repetition_encode, repetition_decode_majority


def test_repetition_roundtrip():
    bits = np.array([0, 1, 1, 0])
    enc = repetition_encode(bits, nrep=3)
    assert enc.tolist() == [0,0,0, 1,1,1, 1,1,1, 0,0,0]

    # add one error per block, majority should still recover
    noisy = enc.copy()
    noisy[1] ^= 1
    noisy[4] ^= 1
    noisy[7] ^= 1
    noisy[10] ^= 1

    dec = repetition_decode_majority(noisy, nrep=3)
    assert dec.tolist() == bits.tolist()
