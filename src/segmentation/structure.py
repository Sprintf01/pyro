import numpy as np
import librosa


DEFAULT_HOP_LENGTH = 512

DEFAULT_MIN_SEGMENT_SECONDS = 8.0
DEFAULT_MAX_SEGMENT_SECONDS = 45.0

DEFAULT_BOUNDARY_THRESHOLD = 0.45


def _normalize_feature(feature: np.ndarray) -> np.ndarray:
    """
    Normalize a feature matrix across time.

    Each feature dimension is standardized independently.
    """

    feature = np.asarray(
        feature,
        dtype=np.float64,
    )

    mean = np.mean(
        feature,
        axis=1,
        keepdims=True,
    )

    std = np.std(
        feature,
        axis=1,
        keepdims=True,
    )

    std = np.where(
        std < 1e-8,
        1.0,
        std,
    )

    return (feature - mean) / std


def build_structure_features(
    chroma: np.ndarray,
    rms: np.ndarray,
    spectral_centroid: np.ndarray,
    spectral_bandwidth: np.ndarray,
    spectral_rolloff: np.ndarray,
    spectral_flatness: np.ndarray,
    onset_strength: np.ndarray,
) -> np.ndarray:
    """
    Combine low-level musical features into one
    time-aligned feature matrix.

    Output shape:
        (num_features, num_frames)
    """

    features = [
        chroma,
        rms[np.newaxis, :],
        spectral_centroid[np.newaxis, :],
        spectral_bandwidth[np.newaxis, :],
        spectral_rolloff[np.newaxis, :],
        spectral_flatness[np.newaxis, :],
        onset_strength[np.newaxis, :],
    ]

    lengths = [
        feature.shape[1]
        for feature in features
    ]

    num_frames = min(lengths)

    aligned_features = [
        feature[:, :num_frames]
        for feature in features
    ]

    normalized_features = [
        _normalize_feature(feature)
        for feature in aligned_features
    ]

    return np.vstack(
        normalized_features
    )


def compute_self_similarity(
    feature_matrix: np.ndarray,
) -> np.ndarray:
    """
    Compute a cosine-similarity matrix between
    feature frames.

    Returns:
        Square self-similarity matrix.
    """

    feature_matrix = np.asarray(
        feature_matrix,
        dtype=np.float64,
    )

    return librosa.segment.recurrence_matrix(
        feature_matrix,
        mode="affinity",
        metric="cosine",
        sym=True,
    )


def compute_novelty_curve(
    feature_matrix: np.ndarray,
) -> np.ndarray:
    """
    Compute a structural novelty curve.

    Peaks indicate locations where the musical
    characteristics change significantly.
    """

    self_similarity = compute_self_similarity(
        feature_matrix
    )

    novelty = np.zeros(
        self_similarity.shape[0],
        dtype=np.float64,
    )

    if len(novelty) >= 3:
        for index in range(
            1,
            len(novelty) - 1,
        ):
            previous_similarity = (
                self_similarity[
                    index - 1,
                    index
                ]
            )

            next_similarity = (
                self_similarity[
                    index,
                    index + 1
                ]
            )

            novelty[index] = abs(
                next_similarity
                - previous_similarity
            )

    # Smooth the novelty curve to reduce
    # frame-level noise.
    if len(novelty) >= 9:
        kernel = (
            np.ones(
                9,
                dtype=np.float64,
            )
            / 9.0
        )

        novelty = np.convolve(
            novelty,
            kernel,
            mode="same",
        )

    # Normalize to [0, 1].
    maximum = np.max(novelty)

    if maximum > 0:
        novelty = novelty / maximum

    return novelty

def detect_boundaries(
    novelty_curve: np.ndarray,
    sample_rate: int,
    hop_length: int = DEFAULT_HOP_LENGTH,
    threshold: float = DEFAULT_BOUNDARY_THRESHOLD,
    min_segment_seconds: float = DEFAULT_MIN_SEGMENT_SECONDS,
    max_segment_seconds: float = DEFAULT_MAX_SEGMENT_SECONDS,
) -> np.ndarray:
    """
    Detect structural boundaries from local peaks
    in the novelty curve.

    A boundary is accepted when:
        1. It is a local novelty peak.
        2. Its normalized novelty exceeds the threshold.
        3. It is sufficiently separated from the previous boundary.

    Long segments are split at the strongest internal
    novelty peak.
    """

    novelty_curve = np.asarray(
        novelty_curve,
        dtype=np.float64,
    )

    if len(novelty_curve) == 0:
        return np.array(
            [],
            dtype=np.float64,
        )

    min_distance_frames = max(
        1,
        int(
            min_segment_seconds
            * sample_rate
            / hop_length
        ),
    )

    # Find actual local peaks instead of taking the
    # first frame that crosses the threshold.
    candidate_frames = librosa.util.peak_pick(
        novelty_curve,
        pre_max=max(
            1,
            min_distance_frames // 2,
        ),
        post_max=max(
            1,
            min_distance_frames // 2,
        ),
        pre_avg=max(
            1,
            min_distance_frames // 2,
        ),
        post_avg=max(
            1,
            min_distance_frames // 2,
        ),
        delta=threshold,
        wait=min_distance_frames,
    )

    candidate_frames = np.asarray(
        candidate_frames,
        dtype=np.int64,
    )

    candidate_times = librosa.frames_to_time(
        candidate_frames,
        sr=sample_rate,
        hop_length=hop_length,
    )

    boundaries = []

    for time in candidate_times:
        time = float(time)

        if not boundaries:
            boundaries.append(time)
            continue

        if (
            time - boundaries[-1]
            >= min_segment_seconds
        ):
            boundaries.append(time)

    # Split excessively long segments using the
    # strongest internal novelty peak.
    final_boundaries = []

    for boundary in boundaries:

        if not final_boundaries:
            final_boundaries.append(boundary)
            continue

        previous = final_boundaries[-1]

        if (
            boundary - previous
            <= max_segment_seconds
        ):
            final_boundaries.append(boundary)
            continue

        start_frame = int(
            librosa.time_to_frames(
                previous,
                sr=sample_rate,
                hop_length=hop_length,
            )
        )

        end_frame = int(
            librosa.time_to_frames(
                boundary,
                sr=sample_rate,
                hop_length=hop_length,
            )
        )

        start_frame = max(
            0,
            start_frame,
        )

        end_frame = min(
            len(novelty_curve),
            end_frame,
        )

        if end_frame <= start_frame:
            final_boundaries.append(boundary)
            continue

        local_novelty = novelty_curve[
            start_frame:end_frame
        ]

        strongest_offset = int(
            np.argmax(local_novelty)
        )

        split_frame = (
            start_frame
            + strongest_offset
        )

        split_time = float(
            librosa.frames_to_time(
                split_frame,
                sr=sample_rate,
                hop_length=hop_length,
            )
        )

        if (
            split_time - previous
            >= min_segment_seconds
        ):
            final_boundaries.append(
                split_time
            )

        if (
            boundary - split_time
            >= min_segment_seconds
        ):
            final_boundaries.append(
                boundary
            )

    return np.asarray(
        sorted(
            set(final_boundaries)
        ),
        dtype=np.float64,
    )

def create_segments(
    boundaries: np.ndarray,
    duration_seconds: float,
) -> list[dict]:
    """
    Convert boundary timestamps into explicit
    start/end segments.

    The first segment always starts at 0.
    The final segment always ends at the track duration.
    """

    boundaries = np.asarray(
        boundaries,
        dtype=np.float64,
    )

    boundaries = boundaries[
        (boundaries > 0.0)
        & (boundaries < duration_seconds)
    ]

    boundaries = np.unique(
        np.sort(boundaries)
    )

    points = np.concatenate(
        [
            np.array([0.0]),
            boundaries,
            np.array([duration_seconds]),
        ]
    )

    segments = []

    for index in range(
        len(points) - 1
    ):

        start = float(
            points[index]
        )

        end = float(
            points[index + 1]
        )

        if end <= start:
            continue

        segments.append(
            {
                "segment_id": index + 1,
                "start_time": start,
                "end_time": end,
                "duration": end - start,
            }
        )

    return segments


def segment_audio_structure(
    chroma: np.ndarray,
    rms: np.ndarray,
    spectral_centroid: np.ndarray,
    spectral_bandwidth: np.ndarray,
    spectral_rolloff: np.ndarray,
    spectral_flatness: np.ndarray,
    onset_strength: np.ndarray,
    duration_seconds: float,
    sample_rate: int,
    hop_length: int = DEFAULT_HOP_LENGTH,
    threshold: float = DEFAULT_BOUNDARY_THRESHOLD,
    min_segment_seconds: float = DEFAULT_MIN_SEGMENT_SECONDS,
    max_segment_seconds: float = DEFAULT_MAX_SEGMENT_SECONDS,
) -> dict:
    """
    Complete structural segmentation pipeline.

    Returns:
        structure_features
        self_similarity
        novelty_curve
        boundaries
        segments
    """

    structure_features = build_structure_features(
        chroma=chroma,
        rms=rms,
        spectral_centroid=spectral_centroid,
        spectral_bandwidth=spectral_bandwidth,
        spectral_rolloff=spectral_rolloff,
        spectral_flatness=spectral_flatness,
        onset_strength=onset_strength,
    )

    self_similarity = compute_self_similarity(
        structure_features
    )

    novelty_curve = compute_novelty_curve(
        structure_features
    )

    boundaries = detect_boundaries(
        novelty_curve=novelty_curve,
        sample_rate=sample_rate,
        hop_length=hop_length,
        threshold=threshold,
        min_segment_seconds=min_segment_seconds,
        max_segment_seconds=max_segment_seconds,
    )

    segments = create_segments(
        boundaries=boundaries,
        duration_seconds=duration_seconds,
    )

    return {
        "structure_features": structure_features,
        "self_similarity": self_similarity,
        "novelty_curve": novelty_curve,
        "boundaries": boundaries,
        "segments": segments,
    }