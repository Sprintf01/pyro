import numpy as np
import librosa


def track_beats(
    waveform: np.ndarray,
    sample_rate: int,
    hop_length: int = 512,
) -> dict:
    """
    Estimate beat positions and beat intervals.

    Returns:
        beat_times:
            Beat timestamps in seconds.

        beat_frames:
            Beat positions in feature frames.

        beat_intervals:
            Time between consecutive beats in seconds.

        mean_beat_interval:
            Average time between beats.

        estimated_bpm:
            BPM calculated from the detected beat intervals.
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
    # Beat tracking
    # ---------------------------------------------------------

    tempo, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_envelope,
        sr=sample_rate,
        hop_length=hop_length,
    )

    # librosa may return tempo as a scalar array.
    tempo = float(np.asarray(tempo).reshape(-1)[0])

    # ---------------------------------------------------------
    # Convert beat frames to seconds
    # ---------------------------------------------------------

    beat_times = librosa.frames_to_time(
        beat_frames,
        sr=sample_rate,
        hop_length=hop_length,
    )

    beat_times = beat_times.astype(np.float64)

    # ---------------------------------------------------------
    # Beat intervals
    # ---------------------------------------------------------

    if len(beat_times) >= 2:
        beat_intervals = np.diff(beat_times)

        mean_beat_interval = float(
            np.mean(beat_intervals)
        )

        estimated_bpm = float(
            60.0 / mean_beat_interval
        )

    else:
        beat_intervals = np.array(
            [],
            dtype=np.float64,
        )

        mean_beat_interval = None
        estimated_bpm = tempo

    return {
        "beat_times": beat_times,
        "beat_frames": beat_frames,
        "beat_intervals": beat_intervals,
        "mean_beat_interval": mean_beat_interval,
        "estimated_bpm": estimated_bpm,
        "tracker_tempo_bpm": tempo,
    }