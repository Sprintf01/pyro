import numpy as np
import librosa


def compute_rms(
    waveform: np.ndarray,
    frame_length: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    """
    Compute frame-level RMS energy.
    """

    rms = librosa.feature.rms(
        y=waveform,
        frame_length=frame_length,
        hop_length=hop_length,
    )

    return rms[0]


def compute_zero_crossing_rate(
    waveform: np.ndarray,
    frame_length: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    """
    Compute frame-level zero-crossing rate.
    """

    zcr = librosa.feature.zero_crossing_rate(
        waveform,
        frame_length=frame_length,
        hop_length=hop_length,
    )

    return zcr[0]
