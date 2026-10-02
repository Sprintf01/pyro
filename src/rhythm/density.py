import numpy as np
import librosa


DEFAULT_WINDOW_SECONDS = 2.0
DEFAULT_HOP_SECONDS = 0.5


def compute_rhythmic_density(
    onset_strength: np.ndarray,
    onset_times: np.ndarray,
    beat_times: np.ndarray,
    duration_seconds: float,
    sample_rate: int = 44100,
    hop_length: int = 512,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    hop_seconds: float = DEFAULT_HOP_SECONDS,
) -> dict:
    """
    Compute local rhythmic-density features over time.

    Features:
        local_onset_rate:
            Number of detected onsets per second.

        local_onset_strength:
            Mean onset strength inside the local window.

        onsets_per_beat:
            Number of detected onsets divided by the number
            of beats in the local window.

        beat_interval_variation:
            Standard deviation of beat intervals inside
            the local window.

    The output is time-indexed using the center of each
    analysis window.
    """

    onset_strength = np.asarray(
        onset_strength,
        dtype=np.float64,
    )

    onset_times = np.asarray(
        onset_times,
        dtype=np.float64,
    )

    beat_times = np.asarray(
        beat_times,
        dtype=np.float64,
    )

    if duration_seconds <= 0:
        raise ValueError(
            "duration_seconds must be greater than zero."
        )

    if window_seconds <= 0:
        raise ValueError(
            "window_seconds must be greater than zero."
        )

    if hop_seconds <= 0:
        raise ValueError(
            "hop_seconds must be greater than zero."
        )

    if len(onset_times) > 1:
        if np.any(np.diff(onset_times) < 0):
            raise ValueError(
                "onset_times must be sorted."
            )

    if len(beat_times) > 1:
        if np.any(np.diff(beat_times) < 0):
            raise ValueError(
                "beat_times must be sorted."
            )

    # The onset-strength envelope is frame-based.
    # Reconstruct the timestamp of each onset-strength frame.
    onset_strength_times = librosa.frames_to_time(
        np.arange(len(onset_strength)),
        sr=sample_rate,
        hop_length=hop_length,
    )

    window_centers = []
    local_onset_rate = []
    local_onset_strength = []
    onsets_per_beat = []
    beat_interval_variation = []

    start_time = 0.0

    while start_time < duration_seconds:
        end_time = min(
            start_time + window_seconds,
            duration_seconds,
        )

        center_time = (
            start_time + end_time
        ) / 2.0

        # Detected onset events.
        onset_mask = (
            (onset_times >= start_time)
            & (onset_times < end_time)
        )

        # Onset-strength frames.
        strength_mask = (
            (onset_strength_times >= start_time)
            & (onset_strength_times < end_time)
        )

        # Beat events.
        beat_mask = (
            (beat_times >= start_time)
            & (beat_times < end_time)
        )

        window_onset_times = onset_times[
            onset_mask
        ]

        window_onset_strength = onset_strength[
            strength_mask
        ]

        window_beat_times = beat_times[
            beat_mask
        ]

        window_duration = (
            end_time - start_time
        )

        onset_count = len(
            window_onset_times
        )

        beat_count = len(
            window_beat_times
        )

        onset_rate = (
            onset_count / window_duration
            if window_duration > 0
            else 0.0
        )

        if len(window_onset_strength) > 0:
            mean_onset_strength = float(
                np.mean(
                    window_onset_strength
                )
            )
        else:
            mean_onset_strength = 0.0

        if beat_count > 0:
            onset_per_beat = float(
                onset_count / beat_count
            )
        else:
            onset_per_beat = 0.0

        if beat_count >= 3:
            beat_intervals = np.diff(
                window_beat_times
            )

            interval_variation = float(
                np.std(beat_intervals)
            )
        else:
            interval_variation = 0.0

        window_centers.append(
            center_time
        )

        local_onset_rate.append(
            onset_rate
        )

        local_onset_strength.append(
            mean_onset_strength
        )

        onsets_per_beat.append(
            onset_per_beat
        )

        beat_interval_variation.append(
            interval_variation
        )

        start_time += hop_seconds

    return {
        "times": np.asarray(
            window_centers,
            dtype=np.float64,
        ),
        "local_onset_rate": np.asarray(
            local_onset_rate,
            dtype=np.float64,
        ),
        "local_onset_strength": np.asarray(
            local_onset_strength,
            dtype=np.float64,
        ),
        "onsets_per_beat": np.asarray(
            onsets_per_beat,
            dtype=np.float64,
        ),
        "beat_interval_variation": np.asarray(
            beat_interval_variation,
            dtype=np.float64,
        ),
    }