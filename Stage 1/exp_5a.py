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

def print_dominant_frequencies(freqs, amplitude, n_peaks=5):
    """
    Print dominant frequency peaks from the amplitude spectrum.
    """
    sorted_indices = np.argsort(amplitude)[::-1]

    printed = 0
    for idx in sorted_indices:
        f = freqs[idx]
        a = amplitude[idx]

        if f == 0:
            continue

        print(f"Peak at {f:.3f} Hz with amplitude {a:.3f}")
        printed += 1

        if printed == n_peaks:
            break

def compute_fft_windowed(x, fs, window_type="hann"):
    """
    Compute single-sided amplitude spectrum with optional windowing.

    Parameters
    ----------
    x : ndarray
        Time signal.
    fs : float
        Sampling frequency.
    window_type : str
        "hann" or "rectangular".

    Returns
    -------
    freqs : ndarray
        Frequency vector.
    amplitude : ndarray
        Single-sided amplitude spectrum.
    """
    N = len(x)

    if window_type == "hann":
        window = np.hanning(N)
    elif window_type == "rectangular":
        window = np.ones(N)
    else:
        raise ValueError("window_type must be 'hann' or 'rectangular'")

    x_windowed = x * window

    X = np.fft.rfft(x_windowed)
    freqs = np.fft.rfftfreq(N, d=1/fs)

    # Coherent gain correction
    coherent_gain = np.mean(window)

    amplitude = np.abs(X) / (N * coherent_gain)
    amplitude[1:-1] *= 2

    return freqs, amplitude

# -----------------------------
# Experiment 5A: no leakage, frequency on FFT bin
# -----------------------------

fs = 100
T = 10

components = [
    (1.0, 2.0)
]

t, x = generate_signal(
    fs=fs,
    T=T,
    components=components,
    noise_std=0.0,
    seed=1
)

freqs, amplitude = compute_fft(x, fs)

plot_time_and_frequency(
    t,
    x,
    freqs,
    amplitude,
    title="No leakage: f = 2.0 Hz, T = 10 s"
)

print(f"Frequency resolution Δf = {1/T} Hz")
print_dominant_frequencies(freqs, amplitude, n_peaks=5)