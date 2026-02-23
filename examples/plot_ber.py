import json
import subprocess
import matplotlib.pyplot as plt

cmd = [
    "qaoa-mimo-qec-run",
    "--snr-db", "0", "2", "4", "6", "8", "10",
    "--experiments", "100",
    "--maxiter", "40",
    "--reps", "1",
    "--nrep", "3",
]
proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
rows = [json.loads(line) for line in proc.stdout.splitlines() if line.strip()]

snr = [r["snr_db"] for r in rows]
ber_u = [r["ber_uncoded"] for r in rows]
ber_c = [r["ber_rep_majority"] for r in rows]

plt.plot(snr, ber_u, marker="o", label="Uncoded (QAOA detect)")
plt.plot(snr, ber_c, marker="s", linestyle="--", label="Repetition-coded + majority")
plt.xlabel("SNR (dB)")
plt.ylabel("Bit error rate (BER)")
plt.yscale("log")
plt.grid(True)
plt.legend()
plt.show()
