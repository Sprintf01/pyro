import numpy as np
import librosa


N_FFT = 2048
HOP_LENGTH = 512
WIN_LENGTH = 2048


def compute_stft(
    waveform: np.ndarray,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    win_length: int = WIN_LENGTH,
):
    """
    Compute complex STFT, magnitude spectrogram,
    and dB spectrogram.
    """

    stft = librosa.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window="hann",
    )

    magnitude = np.abs(stft)

    magnitude_db = librosa.amplitude_to_db(
        magnitude,
        ref=np.max,
    )

    return {
        "stft": stft,
        "magnitude": magnitude,
        "magnitude_db": magnitude_db,
    }
