import numpy as np
import matplotlib.pyplot as plt


def generate_signal(fs, T, components, noise_std=0.0, seed=1):
    """
    Generate a synthetic vibration signal.

    Parameters
    ----------
    fs : float
        Sampling frequency in Hz.
    T : float
        Signal duration in seconds.
    components : list of tuples
        Each tuple is (amplitude, frequency).
    noise_std : float
        Standard deviation of Gaussian noise.
    seed : int
        Random seed for repeatable noise.

    Returns
    -------
    t : ndarray
        Time vector.
    x : ndarray
        Signal.
    """
    dt = 1 / fs
    t = np.arange(0, T, dt)

    x = np.zeros_like(t)

    for A, f in components:
        x += A * np.sin(2 * np.pi * f * t)

    rng = np.random.default_rng(seed)
    noise = rng.normal(0, noise_std, size=len(t))

    x = x + noise

    return t, x


def compute_fft(x, fs):
    """
    Compute single-sided amplitude spectrum.
    """
    N = len(x)

    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(N, d=1/fs)

    amplitude = np.abs(X) / N
    amplitude[1:-1] *= 2

    return freqs, amplitude


def plot_time_and_frequency(t, x, freqs, amplitude, title):
    """
    Plot time-domain signal and frequency-domain spectrum.
    """
    plt.figure(figsize=(10, 4))
    plt.plot(t, x)
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title(title + " — Time Domain")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(freqs, amplitude)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitude")
    plt.title(title + " — Frequency Domain")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# -----------------------------
# Experiment 1: clean signal
# -----------------------------

fs = 100
T = 10
components = [
    (1.0, 2.0),
    (0.4, 7.0)
]
noise_std = 0.0

t, x = generate_signal(fs, T, components, noise_std)
freqs, amplitude = compute_fft(x, fs)

plot_time_and_frequency(
    t,
    x,
    freqs,
    amplitude,
    title="Clean synthetic vibration signal"
)

print(f"Sampling frequency fs = {fs} Hz")
print(f"Nyquist frequency = {fs/2} Hz")
print(f"Duration T = {T} s")
print(f"Frequency resolution Δf = {1/T} Hz")
print(f"Number of samples N = {len(x)}")