import numpy as np
import librosa


# Reasonable musical tempo range for this project.
MIN_TEMPO_BPM = 40.0
MAX_TEMPO_BPM = 240.0


def estimate_tempo(
    waveform: np.ndarray,
    sample_rate: int,
    hop_length: int = 512,
) -> dict:
    """
    Estimate the global tempo of an audio track.

    Returns:
        tempo_bpm:
            Estimated global tempo in beats per minute.

        tempo_candidates:
            Alternative plausible tempo estimates derived
            from the tempogram.
    """

    # ---------------------------------------------------------
    # Onset envelope
    # ---------------------------------------------------------

    onset_envelope = librosa.onset.onset_strength(
        y=waveform,
        sr=sample_rate,
        hop_length=hop_length,
    )

    # ---------------------------------------------------------
    # Global tempo estimate
    # ---------------------------------------------------------

    tempo = librosa.feature.tempo(
        onset_envelope=onset_envelope,
        sr=sample_rate,
        hop_length=hop_length,
        aggregate=np.median,
    )

    tempo_bpm = float(tempo[0])

    # ---------------------------------------------------------
    # Tempogram
    # ---------------------------------------------------------

    tempogram = librosa.feature.tempogram(
        onset_envelope=onset_envelope,
        sr=sample_rate,
        hop_length=hop_length,
    )

    tempo_frequencies = librosa.tempo_frequencies(
        tempogram.shape[0],
        hop_length=hop_length,
        sr=sample_rate,
    )

    # Average tempogram strength over time.
    strength = np.mean(
        tempogram,
        axis=1,
    )

    # ---------------------------------------------------------
    # Filter to plausible musical tempos
    # ---------------------------------------------------------

    valid = (
        np.isfinite(tempo_frequencies)
        & np.isfinite(strength)
        & (tempo_frequencies >= MIN_TEMPO_BPM)
        & (tempo_frequencies <= MAX_TEMPO_BPM)
    )

    frequencies = tempo_frequencies[valid]
    strengths = strength[valid]

    if len(frequencies) == 0:
        tempo_candidates = []

    else:
        # Sort by tempogram strength.
        strongest_indices = np.argsort(
            strengths
        )[::-1]

        tempo_candidates = []

        for index in strongest_indices:
            candidate_bpm = float(frequencies[index])

            tempo_candidates.append(
                {
                    "bpm": candidate_bpm,
                    "strength": float(strengths[index]),
                }
            )

            if len(tempo_candidates) >= 5:
                break

    return {
        "tempo_bpm": tempo_bpm,
        "tempo_candidates": tempo_candidates,
    }