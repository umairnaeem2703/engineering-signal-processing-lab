import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from stage_02_fft_frequency_analysis.src.spectral_analysis import (
    single_sided_fft_amplitude,
    detect_frequency_peaks,
    detect_harmonics,
)


FIGURE_DIR = Path(__file__).resolve().parents[1] / "report" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 1. Signal settings
# --------------------------------------------------
fs = 200
duration = 5
dt = 1 / fs

t = np.arange(0, duration, dt)


# --------------------------------------------------
# 2. Harmonic-rich signal
# --------------------------------------------------
f0 = 5

x = (
    1.0 * np.sin(2 * np.pi * f0 * t)
    + 0.6 * np.sin(2 * np.pi * 2 * f0 * t)
    + 0.3 * np.sin(2 * np.pi * 3 * f0 * t)
    + 0.2 * np.sin(2 * np.pi * 4 * f0 * t)
)


# --------------------------------------------------
# 3. FFT amplitude spectrum
# --------------------------------------------------
freqs, amplitude = single_sided_fft_amplitude(x, fs)


# --------------------------------------------------
# 4. Detect peaks
# --------------------------------------------------
peak_freqs, peak_amps, _ = detect_frequency_peaks(
    freqs,
    amplitude,
    min_height=0.1,
)


# --------------------------------------------------
# 5. Harmonic detection
# --------------------------------------------------
harmonics = detect_harmonics(
    peak_freqs,
    peak_amps,
    fundamental=None,
    tolerance=0.05,
)


print("\nDetected harmonic components:")
for f, a, h in harmonics:
    print(
        f"Harmonic {h}: "
        f"Frequency = {f:.2f} Hz, "
        f"Amplitude = {a:.3f}"
    )


# --------------------------------------------------
# 6. Plot spectrum
# --------------------------------------------------
plt.figure(figsize=(10, 4))
plt.stem(freqs, amplitude)
plt.plot(peak_freqs, peak_amps, "ro", label="Detected peaks")

for f, a, h in harmonics:
    plt.text(
        f,
        a,
        f"{h}f₀",
        ha="center",
        va="bottom",
    )

plt.xlabel("Frequency [Hz]")
plt.ylabel("Amplitude")
plt.title("Harmonic Detection using FFT")
plt.grid(True)
plt.xlim(0, 40)
plt.legend()
plt.tight_layout()
plt.savefig(FIGURE_DIR / "exp05_harmonic_detection.png", dpi=300)
plt.show()
