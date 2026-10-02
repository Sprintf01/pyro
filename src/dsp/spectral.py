import numpy as np
import librosa


def compute_spectral_features(
    waveform: np.ndarray,
    sample_rate: int,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> dict:
    """
    Compute frame-level spectral features.
    """

    centroid = librosa.feature.spectral_centroid(
        y=waveform,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
    )[0]

    bandwidth = librosa.feature.spectral_bandwidth(
        y=waveform,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
    )[0]

    rolloff = librosa.feature.spectral_rolloff(
        y=waveform,
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
    )[0]

    flatness = librosa.feature.spectral_flatness(
        y=waveform,
        n_fft=n_fft,
        hop_length=hop_length,
    )[0]

    return {
        "spectral_centroid": centroid,
        "spectral_bandwidth": bandwidth,
        "spectral_rolloff": rolloff,
        "spectral_flatness": flatness,
    }


def compute_chroma(
    waveform: np.ndarray,
    sample_rate: int,
    hop_length: int = 512,
) -> np.ndarray:
    """
    Compute frame-level chroma representation.
    """

    chroma = librosa.feature.chroma_stft(
        y=waveform,
        sr=sample_rate,
        hop_length=hop_length,
    )

    return chroma