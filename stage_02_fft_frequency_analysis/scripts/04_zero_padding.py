import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from stage_02_fft_frequency_analysis.src.spectral_analysis import (
    single_sided_fft_amplitude,
)


FIGURE_DIR = Path(__file__).resolve().parents[1] / "report" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 1. Signal settings
# --------------------------------------------------
fs = 100
duration = 2
dt = 1 / fs

t = np.arange(0, duration, dt)
N = len(t)


# --------------------------------------------------
# 2. Signal with non-bin-centered frequency
# --------------------------------------------------
f = 5.3
A = 1.0

x = A * np.sin(2 * np.pi * f * t)


# --------------------------------------------------
# 3. FFT without zero-padding
# --------------------------------------------------
freqs, amplitude = single_sided_fft_amplitude(x, fs)


# --------------------------------------------------
# 4. FFT with zero-padding
# --------------------------------------------------
N_pad = 2048

X_pad = np.fft.fft(x, n=N_pad)
freqs_pad = np.fft.fftfreq(N_pad, d=dt)

positive_mask = freqs_pad >= 0

freqs_pad_pos = freqs_pad[positive_mask]
X_pad_pos = X_pad[positive_mask]

# Normalize by original N, not N_pad, because zero-padding does not add signal energy.
amplitude_pad = np.abs(X_pad_pos) / N
amplitude_pad[1:] *= 2


# --------------------------------------------------
# 5. Plot comparison
# --------------------------------------------------
plt.figure(figsize=(10, 4))
plt.stem(freqs, amplitude, label="No zero-padding")
plt.plot(freqs_pad_pos, amplitude_pad, label="Zero-padded FFT")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Amplitude")
plt.title("Effect of Zero-Padding on FFT Spectrum")
plt.grid(True)
plt.xlim(0, 15)
plt.legend()
plt.tight_layout()
plt.savefig(FIGURE_DIR / "exp04_zero_padding.png", dpi=300)
plt.show()


# --------------------------------------------------
# 6. Print frequency spacing
# --------------------------------------------------
df_original = fs / N
df_padded = fs / N_pad

print(f"Original number of samples N = {N}")
print(f"Original FFT spacing df = {df_original:.3f} Hz")

print(f"\nZero-padded FFT length N_pad = {N_pad}")
print(f"Zero-padded frequency spacing df_pad = {df_padded:.3f} Hz")

print("\nImportant:")
print("Zero-padding makes the plotted frequency grid denser.")
print("It does not improve the true physical frequency resolution.")
print("True frequency resolution depends mainly on record duration.")
