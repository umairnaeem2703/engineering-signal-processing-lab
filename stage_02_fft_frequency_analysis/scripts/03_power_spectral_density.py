import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import find_peaks

from stage_02_fft_frequency_analysis.src.spectral_analysis import compute_psd_welch


FIGURE_DIR = Path(__file__).resolve().parents[1] / "report" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 1. Signal settings
# --------------------------------------------------
fs = 100
duration = 20
dt = 1 / fs

t = np.arange(0, duration, dt)


# --------------------------------------------------
# 2. Synthetic noisy vibration signal
# --------------------------------------------------
f1 = 5
f2 = 15

A1 = 1.5
A2 = 0.8

np.random.seed(42)
noise = 0.5 * np.random.randn(len(t))

x = (
    A1 * np.sin(2 * np.pi * f1 * t)
    + A2 * np.sin(2 * np.pi * f2 * t)
    + noise
)


# --------------------------------------------------
# 3. Compute PSD using Welch method
# --------------------------------------------------
freqs_psd, psd = compute_psd_welch(
    x,
    fs=fs,
    nperseg=512,
    window="hann",
)


# --------------------------------------------------
# 4. Detect PSD peaks
# --------------------------------------------------
peak_indices, properties = find_peaks(
    psd,
    height=0.01,
)

peak_freqs = freqs_psd[peak_indices]
peak_psd_values = psd[peak_indices]


print("\nDetected PSD peaks:")
for f, p in zip(peak_freqs, peak_psd_values):
    print(f"Frequency = {f:.2f} Hz, PSD = {p:.5f}")


# --------------------------------------------------
# 5. Plot noisy time signal
# --------------------------------------------------
plt.figure(figsize=(10, 4))
plt.plot(t, x)
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.title("Noisy Synthetic Vibration Signal")
plt.grid(True)
plt.tight_layout()
plt.savefig(FIGURE_DIR / "exp03_noisy_time_signal.png", dpi=300)
plt.show()


# --------------------------------------------------
# 6. Plot PSD
# --------------------------------------------------
plt.figure(figsize=(10, 4))
plt.semilogy(freqs_psd, psd)
plt.plot(peak_freqs, peak_psd_values, "ro", label="Detected PSD peaks")
plt.xlabel("Frequency [Hz]")
plt.ylabel("PSD [Amplitude²/Hz]")
plt.title("Power Spectral Density using Welch Method")
plt.grid(True)
plt.xlim(0, 30)
plt.legend()
plt.tight_layout()
plt.savefig(FIGURE_DIR / "exp03_welch_psd.png", dpi=300)
plt.show()
