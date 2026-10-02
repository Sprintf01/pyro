import numpy as np

from madmom.features.downbeats import (
    RNNDownBeatProcessor,
    DBNDownBeatTrackingProcessor,
)


def track_downbeats(
    audio_path: str,
    fps: int = 100,
    beats_per_bar: int = 4,
) -> dict:
    """
    Detect beats and downbeats from an audio file.

    Returns:
        beat_times:
            Timestamp of every detected beat in seconds.

        beat_positions:
            Beat position within the bar.
            Position 1 represents the downbeat.

        downbeat_times:
            Timestamp of every detected downbeat.

        num_beats:
            Number of detected beats.

        num_downbeats:
            Number of detected downbeats.
    """

    # ---------------------------------------------------------
    # 1. RNN downbeat activation
    # ---------------------------------------------------------

    processor = RNNDownBeatProcessor()

    activations = processor(audio_path)

    # ---------------------------------------------------------
    # 2. DBN beat/downbeat tracking
    # ---------------------------------------------------------

    tracker = DBNDownBeatTrackingProcessor(
        beats_per_bar=[beats_per_bar],
        fps=fps,
    )

    tracking = tracker(activations)

    tracking = np.asarray(
        tracking,
        dtype=np.float64,
    )

    if tracking.size == 0:
        return {
            "beat_times": np.array([], dtype=np.float64),
            "beat_positions": np.array([], dtype=np.int64),
            "downbeat_times": np.array([], dtype=np.float64),
            "num_beats": 0,
            "num_downbeats": 0,
        }

    # ---------------------------------------------------------
    # 3. Extract timestamps and beat positions
    # ---------------------------------------------------------

    beat_times = tracking[:, 0]

    beat_positions = tracking[:, 1].astype(np.int64)

    # Position 1 = downbeat.
    downbeat_mask = beat_positions == 1

    downbeat_times = beat_times[
        downbeat_mask
    ]

    return {
        "beat_times": beat_times,
        "beat_positions": beat_positions,
        "downbeat_times": downbeat_times,
        "num_beats": int(len(beat_times)),
        "num_downbeats": int(len(downbeat_times)),
    }