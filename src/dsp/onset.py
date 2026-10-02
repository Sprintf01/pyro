import numpy as np
import librosa


def compute_onset_strength(
    waveform: np.ndarray,
    sample_rate: int,
    hop_length: int = 512,
) -> np.ndarray:
    onset_strength = librosa.onset.onset_strength(
        y=waveform,
        sr=sample_rate,
        hop_length=hop_length,
    )

    return onset_strength


def detect_onsets(
    waveform: np.ndarray,
    sample_rate: int,
    hop_length: int = 512,
) -> dict:
    """
    Detect discrete onset events and return their timestamps
    together with the onset-strength envelope.
    """

    onset_strength = librosa.onset.onset_strength(
        y=waveform,
        sr=sample_rate,
        hop_length=hop_length,
    )

    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_strength,
        sr=sample_rate,
        hop_length=hop_length,
    )

    onset_times = librosa.frames_to_time(
        onset_frames,
        sr=sample_rate,
        hop_length=hop_length,
    )

    return {
        "onset_strength": onset_strength,
        "onset_frames": onset_frames,
        "onset_times": onset_times.astype(
            np.float64
        ),
    }