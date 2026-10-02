from pathlib import Path
import json

import numpy as np
import pandas as pd

from src.preprocessing.audio import load_audio, save_standardized_wav
from src.preprocessing.metadata import extract_metadata

from src.dsp.stft import compute_stft
from src.dsp.energy import (
    compute_rms,
    compute_zero_crossing_rate,
)
from src.dsp.spectral import (
    compute_spectral_features,
    compute_chroma,
)
from src.dsp.onset import detect_onsets

from src.rhythm.tempo import estimate_tempo
from src.rhythm.beats import track_beats
from src.rhythm.downbeats import track_downbeats
from src.rhythm.density import compute_rhythmic_density


HOP_LENGTH = 512


def analyze_audio(
    input_path: str | Path,
    processed_dir: str | Path,
    output_dir: str | Path,
) -> dict:

    input_path = Path(input_path)
    processed_dir = Path(processed_dir)
    output_dir = Path(output_dir)

    processed_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # 1. Load and standardize audio
    # ---------------------------------------------------------

    audio = load_audio(input_path)
    metadata = extract_metadata(audio)

    standardized_path = (
        processed_dir
        / f"{input_path.stem}_standardized.wav"
    )

    save_standardized_wav(
        audio,
        standardized_path,
    )

    # ---------------------------------------------------------
    # 2. STFT
    # ---------------------------------------------------------

    stft_features = compute_stft(
        audio.waveform
    )

    # ---------------------------------------------------------
    # 3. Time axis
    # ---------------------------------------------------------

    times = (
        np.arange(
            stft_features["magnitude"].shape[1]
        )
        * HOP_LENGTH
        / audio.sample_rate
    )

    # ---------------------------------------------------------
    # 4. Energy
    # ---------------------------------------------------------

    rms = compute_rms(
        audio.waveform,
        hop_length=HOP_LENGTH,
    )

    zcr = compute_zero_crossing_rate(
        audio.waveform,
        hop_length=HOP_LENGTH,
    )

    # ---------------------------------------------------------
    # 5. Spectral features
    # ---------------------------------------------------------

    spectral = compute_spectral_features(
        audio.waveform,
        audio.sample_rate,
        hop_length=HOP_LENGTH,
    )

    # ---------------------------------------------------------
    # 6. Chroma
    # ---------------------------------------------------------

    chroma = compute_chroma(
        audio.waveform,
        audio.sample_rate,
        hop_length=HOP_LENGTH,
    )

    # ---------------------------------------------------------
    # 7. Onset detection
    #
    # This provides both:
    # - frame-level onset strength
    # - discrete onset timestamps
    # ---------------------------------------------------------

    onset_data = detect_onsets(
        audio.waveform,
        audio.sample_rate,
        hop_length=HOP_LENGTH,
    )

    onset_strength = onset_data[
        "onset_strength"
    ]

    onset_times = onset_data[
        "onset_times"
    ]

    # ---------------------------------------------------------
    # 8. Tempo estimation
    # ---------------------------------------------------------

    rhythm = estimate_tempo(
        audio.waveform,
        audio.sample_rate,
        hop_length=HOP_LENGTH,
    )

    # ---------------------------------------------------------
    # 9. Librosa beat tracking
    #
    # This remains separate from madmom downbeat tracking.
    # Do not merge the two beat sequences.
    # ---------------------------------------------------------

    beats = track_beats(
        audio.waveform,
        audio.sample_rate,
        hop_length=HOP_LENGTH,
    )

    # ---------------------------------------------------------
    # 10. Madmom beat/downbeat tracking
    #
    # This produces:
    # - beat timestamps
    # - beat positions within the bar
    # - downbeat timestamps
    #
    # It intentionally remains separate from the librosa
    # beat tracker above.
    # ---------------------------------------------------------

    downbeats = track_downbeats(
        str(input_path)
    )

    # ---------------------------------------------------------
    # 11. Rhythmic density
    #
    # Uses:
    # - detected onset events
    # - onset-strength envelope
    # - librosa beat timestamps
    #
    # This remains a separate time-indexed analysis layer.
    # ---------------------------------------------------------

    duration_seconds = (
        len(audio.waveform)
        / audio.sample_rate
    )

    density = compute_rhythmic_density(
        onset_strength=onset_strength,
        onset_times=onset_times,
        beat_times=beats["beat_times"],
        duration_seconds=duration_seconds,
        sample_rate=audio.sample_rate,
        hop_length=HOP_LENGTH,
    )

    # ---------------------------------------------------------
    # 12. Align frame-level features
    # ---------------------------------------------------------

    lengths = {
        "times": len(times),
        "rms": len(rms),
        "zcr": len(zcr),
        "spectral_centroid": len(
            spectral["spectral_centroid"]
        ),
        "spectral_bandwidth": len(
            spectral["spectral_bandwidth"]
        ),
        "spectral_rolloff": len(
            spectral["spectral_rolloff"]
        ),
        "spectral_flatness": len(
            spectral["spectral_flatness"]
        ),
        "onset_strength": len(
            onset_strength
        ),
        "chroma": chroma.shape[1],
    }

    num_frames = min(
        lengths.values()
    )

    frame_features = pd.DataFrame(
        {
            "time": times[:num_frames],
            "rms": rms[:num_frames],
            "zcr": zcr[:num_frames],
            "spectral_centroid": spectral[
                "spectral_centroid"
            ][:num_frames],
            "spectral_bandwidth": spectral[
                "spectral_bandwidth"
            ][:num_frames],
            "spectral_rolloff": spectral[
                "spectral_rolloff"
            ][:num_frames],
            "spectral_flatness": spectral[
                "spectral_flatness"
            ][:num_frames],
            "onset_strength": onset_strength[
                :num_frames
            ],
        }
    )

    for i in range(12):
        frame_features[f"chroma_{i}"] = (
            chroma[i, :num_frames]
        )

    # ---------------------------------------------------------
    # 13. Save frame-level feature bundle
    # ---------------------------------------------------------

    feature_path = (
        output_dir
        / f"{input_path.stem}_features.csv"
    )

    frame_features.to_csv(
        feature_path,
        index=False,
    )

    # ---------------------------------------------------------
    # 14. Save STFT bundle
    # ---------------------------------------------------------

    stft_path = (
        output_dir
        / f"{input_path.stem}_stft.npz"
    )

    np.savez_compressed(
        stft_path,
        stft=stft_features["stft"],
        magnitude=stft_features["magnitude"],
        magnitude_db=stft_features["magnitude_db"],
        chroma=chroma[:, :num_frames],
        times=times[:num_frames],
    )

    # ---------------------------------------------------------
    # 15. Add generated-file metadata
    # ---------------------------------------------------------

    metadata["standardized_audio"] = str(
        standardized_path
    )

    metadata["feature_file"] = str(
        feature_path
    )

    metadata["stft_file"] = str(
        stft_path
    )

    metadata["num_feature_frames"] = (
        num_frames
    )

    # ---------------------------------------------------------
    # 16. Save global rhythm analysis
    # ---------------------------------------------------------

    rhythm_path = (
        output_dir
        / f"{input_path.stem}_rhythm.json"
    )

    with open(
        rhythm_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            rhythm,
            f,
            indent=4,
        )

    metadata["rhythm_file"] = str(
        rhythm_path
    )

    # ---------------------------------------------------------
    # 17. Save librosa beat analysis
    #
    # This is intentionally kept separate from the madmom
    # beat/downbeat analysis.
    # ---------------------------------------------------------

    beats_path = (
        output_dir
        / f"{input_path.stem}_beats.json"
    )

    beats_json = {
        "tracker_tempo_bpm": (
            beats["tracker_tempo_bpm"]
        ),
        "estimated_bpm": (
            beats["estimated_bpm"]
        ),
        "mean_beat_interval": (
            beats["mean_beat_interval"]
        ),
        "num_beats": len(
            beats["beat_times"]
        ),
        "beat_times": (
            beats["beat_times"].tolist()
        ),
        "beat_frames": (
            beats["beat_frames"].tolist()
        ),
    }

    with open(
        beats_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            beats_json,
            f,
            indent=4,
        )

    metadata["beats_file"] = str(
        beats_path
    )

    # ---------------------------------------------------------
    # 18. Save madmom beat/downbeat analysis
    # ---------------------------------------------------------

    downbeats_path = (
        output_dir
        / f"{input_path.stem}_downbeats.json"
    )

    downbeats_json = {
        "num_beats": (
            downbeats["num_beats"]
        ),
        "num_downbeats": (
            downbeats["num_downbeats"]
        ),
        "beat_times": (
            downbeats["beat_times"].tolist()
        ),
        "beat_positions": (
            downbeats["beat_positions"].tolist()
        ),
        "downbeat_times": (
            downbeats["downbeat_times"].tolist()
        ),
    }

    with open(
        downbeats_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            downbeats_json,
            f,
            indent=4,
        )

    metadata["downbeats_file"] = str(
        downbeats_path
    )

    # ---------------------------------------------------------
    # 19. Save rhythmic density analysis
    # ---------------------------------------------------------

    density_path = (
        output_dir
        / f"{input_path.stem}_density.json"
    )

    density_json = {
        "window_seconds": 2.0,
        "hop_seconds": 0.5,
        "num_windows": len(
            density["times"]
        ),
        "times": (
            density["times"].tolist()
        ),
        "local_onset_rate": (
            density["local_onset_rate"].tolist()
        ),
        "local_onset_strength": (
            density["local_onset_strength"].tolist()
        ),
        "onsets_per_beat": (
            density["onsets_per_beat"].tolist()
        ),
        "beat_interval_variation": (
            density["beat_interval_variation"].tolist()
        ),
    }

    with open(
        density_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            density_json,
            f,
            indent=4,
        )

    metadata["density_file"] = str(
        density_path
    )

    # ---------------------------------------------------------
    # 20. Return complete analysis result
    # ---------------------------------------------------------

    return {
        "metadata": metadata,
        "features": frame_features,
        "stft": stft_features,
        "times": times[:num_frames],
        "waveform": audio.waveform,
        "rhythm": rhythm,
        "beats": beats,
        "downbeats": downbeats,
        "density": density,
    }