import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from stage_02_fft_frequency_analysis.src.spectral_analysis import (
    single_sided_fft_amplitude,
    single_sided_fft_phase,
    detect_frequency_peaks,
)


FIGURE_DIR = Path(__file__).resolve().parents[1] / "report" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 1. Signal settings
# --------------------------------------------------
fs = 100
duration = 5
dt = 1 / fs

t = np.arange(0, duration, dt)


# --------------------------------------------------
# 2. Synthetic signal with phase shift
# --------------------------------------------------
f1 = 5
f2 = 15

A1 = 1.5
A2 = 0.8

phi1 = np.pi / 6      # 30 degrees
phi2 = np.pi / 3      # 60 degrees

x = (
    A1 * np.sin(2 * np.pi * f1 * t + phi1)
    + A2 * np.sin(2 * np.pi * f2 * t + phi2)
)


# --------------------------------------------------
# 3. Amplitude and phase spectra
# --------------------------------------------------
freqs, amplitude = single_sided_fft_amplitude(x, fs)
_, phase = single_sided_fft_phase(x, fs)

peak_freqs, peak_amps, _ = detect_frequency_peaks(
    freqs,
    amplitude,
    min_height=0.1,
)


# --------------------------------------------------
# 4. Print detected peak phase values
# --------------------------------------------------
print("\nDetected frequency peaks with phase:")

for f, a in zip(peak_freqs, peak_amps):
    idx = np.argmin(np.abs(freqs - f))
    phase_rad = phase[idx]
    phase_deg = np.degrees(phase_rad)

    print(
        f"Frequency = {f:.2f} Hz, "
        f"Amplitude = {a:.3f}, "
        f"Phase = {phase_rad:.3f} rad "
        f"({phase_deg:.2f} deg)"
    )


# --------------------------------------------------
# 5. Plot amplitude spectrum
# --------------------------------------------------
plt.figure(figsize=(10, 4))
plt.stem(freqs, amplitude)
plt.plot(peak_freqs, peak_amps, "ro", label="Detected peaks")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Amplitude")
plt.title("FFT Amplitude Spectrum")
plt.grid(True)
plt.xlim(0, 30)
plt.legend()
plt.tight_layout()
plt.savefig(FIGURE_DIR / "exp02_amplitude_spectrum.png", dpi=300)
plt.show()


# --------------------------------------------------
# 6. Plot phase spectrum
# --------------------------------------------------
plt.figure(figsize=(10, 4))
plt.stem(freqs, phase)
plt.xlabel("Frequency [Hz]")
plt.ylabel("Phase [rad]")
plt.title("FFT Phase Spectrum")
plt.grid(True)
plt.xlim(0, 30)
plt.tight_layout()
plt.savefig(FIGURE_DIR / "exp02_phase_spectrum.png", dpi=300)
plt.show()
