import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import windows

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
# 2. Non-bin-centered signal
# --------------------------------------------------
f = 5.3
A = 1.0

x = A * np.sin(2 * np.pi * f * t)


# --------------------------------------------------
# 3. FFT without window
# --------------------------------------------------
freqs, amplitude_raw = single_sided_fft_amplitude(x, fs)


# --------------------------------------------------
# 4. FFT with Hann window
# --------------------------------------------------
window = windows.hann(N)
x_windowed = x * window

freqs_w, amplitude_windowed = single_sided_fft_amplitude(x_windowed, fs)

# Approximate amplitude correction for Hann window.
# The Hann window reduces sinusoidal amplitude by its coherent gain.
coherent_gain = np.mean(window)
amplitude_windowed = amplitude_windowed / coherent_gain


# --------------------------------------------------
# 5. Plot comparison
# --------------------------------------------------
plt.figure(figsize=(10, 4))
plt.plot(freqs, amplitude_raw, label="No window")
plt.plot(freqs_w, amplitude_windowed, label="Hann window")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Amplitude")
plt.title("Spectral Leakage and Windowing")
plt.grid(True)
plt.xlim(0, 15)
plt.legend()
plt.tight_layout()
plt.savefig(FIGURE_DIR / "exp06_spectral_leakage_windowing.png", dpi=300)
plt.show()


# --------------------------------------------------
# 6. Print explanation
# --------------------------------------------------
df = fs / N

print(f"Signal frequency = {f:.2f} Hz")
print(f"FFT frequency spacing df = {df:.3f} Hz")
print("\nBecause 5.3 Hz does not fall exactly on an FFT bin,")
print("energy spreads into neighboring frequency bins.")
print("This spreading is called spectral leakage.")
print("\nThe Hann window reduces leakage away from the main peak,")
print("but it also broadens the main lobe.")
