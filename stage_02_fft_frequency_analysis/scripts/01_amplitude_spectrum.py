import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from stage_02_fft_frequency_analysis.src.spectral_analysis import (
    single_sided_fft_amplitude,
    detect_frequency_peaks,
    frequency_resolution,
)


FIGURE_DIR = Path(__file__).resolve().parents[1] / "report" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 1. Signal settings
# --------------------------------------------------
fs = 100          # sampling frequency in Hz
duration = 5      # signal duration in seconds
dt = 1 / fs       # time step

t = np.arange(0, duration, dt)
N = len(t)


# --------------------------------------------------
# 2. Create synthetic vibration signal
# --------------------------------------------------
f1 = 5            # Hz
f2 = 15           # Hz

A1 = 1.5
A2 = 0.8

x = A1 * np.sin(2 * np.pi * f1 * t) + A2 * np.sin(2 * np.pi * f2 * t)


# --------------------------------------------------
# 3. Compute single-sided FFT amplitude spectrum
# --------------------------------------------------
freqs, amplitude = single_sided_fft_amplitude(x, fs)


# --------------------------------------------------
# 4. Detect dominant frequency peaks
# --------------------------------------------------
peak_freqs, peak_amps, _ = detect_frequency_peaks(
    freqs,
    amplitude,
    min_height=0.1,
)


print("\nDetected frequency peaks:")
for f, a in zip(peak_freqs, peak_amps):
    print(f"Frequency = {f:.2f} Hz, Amplitude = {a:.3f}")


# --------------------------------------------------
# 5. Print sampling and resolution information
# --------------------------------------------------
df = frequency_resolution(fs, N)

print(f"\nNumber of samples N = {N}")
print(f"Sampling frequency fs = {fs} Hz")
print(f"Frequency resolution df = {df:.3f} Hz")


# --------------------------------------------------
# 6. Plot time signal
# --------------------------------------------------
plt.figure(figsize=(10, 4))
plt.plot(t, x)
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.title("Synthetic Vibration Signal")
plt.grid(True)
plt.tight_layout()
plt.savefig(FIGURE_DIR / "exp01_time_signal.png", dpi=300)
plt.show()


# --------------------------------------------------
# 7. Plot amplitude spectrum
# --------------------------------------------------
plt.figure(figsize=(10, 4))
plt.stem(freqs, amplitude)
plt.plot(peak_freqs, peak_amps, "ro", label="Detected peaks")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Amplitude")
plt.title("Single-Sided FFT Amplitude Spectrum")
plt.grid(True)
plt.xlim(0, 30)
plt.legend()
plt.tight_layout()
plt.savefig(FIGURE_DIR / "exp01_amplitude_spectrum.png", dpi=300)
plt.show()
