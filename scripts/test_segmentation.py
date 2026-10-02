import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import librosa

from src.preprocessing.audio import load_audio
from src.dsp.spectral import compute_spectral_features
from src.dsp.energy import compute_rms
from src.dsp.onset import compute_onset_strength
from src.dsp.spectral import compute_chroma

from src.segmentation.structure import (
    segment_audio_structure,
)


INPUT_AUDIO = (
    Path("data")
    / "raw"
    / "test_song.mp3"
)

OUTPUT_DIR = (
    Path("outputs")
    / "analysis"
    / "plots"
)

HOP_LENGTH = 512


def main():

    print()
    print("=" * 60)
    print("STRUCTURAL SEGMENTATION TEST")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. Load audio
    # ---------------------------------------------------------

    audio = load_audio(
        INPUT_AUDIO
    )

    print(
        f"\nAudio duration: "
        f"{len(audio.waveform) / audio.sample_rate:.3f} s"
    )

    # ---------------------------------------------------------
    # 2. Compute required features
    # ---------------------------------------------------------

    print("\nComputing features...")

    rms = compute_rms(
        audio.waveform,
        hop_length=HOP_LENGTH,
    )

    onset_strength = compute_onset_strength(
        audio.waveform,
        audio.sample_rate,
        hop_length=HOP_LENGTH,
    )

    spectral = compute_spectral_features(
        audio.waveform,
        audio.sample_rate,
        hop_length=HOP_LENGTH,
    )

    chroma = compute_chroma(
        audio.waveform,
        audio.sample_rate,
        hop_length=HOP_LENGTH,
    )

    # ---------------------------------------------------------
    # 3. Structural segmentation
    # ---------------------------------------------------------

    print("Running structural segmentation...")

    duration_seconds = (
        len(audio.waveform)
        / audio.sample_rate
    )

    result = segment_audio_structure(
        chroma=chroma,
        rms=rms,
        spectral_centroid=spectral[
            "spectral_centroid"
        ],
        spectral_bandwidth=spectral[
            "spectral_bandwidth"
        ],
        spectral_rolloff=spectral[
            "spectral_rolloff"
        ],
        spectral_flatness=spectral[
            "spectral_flatness"
        ],
        onset_strength=onset_strength,
        duration_seconds=duration_seconds,
        sample_rate=audio.sample_rate,
        hop_length=HOP_LENGTH,
    )

    # ---------------------------------------------------------
    # 4. Print boundaries
    # ---------------------------------------------------------

    print(
        f"\nDetected boundaries: "
        f"{len(result['boundaries'])}"
    )

    for index, boundary in enumerate(
        result["boundaries"],
        start=1,
    ):
        print(
            f"  Boundary {index:02d}: "
            f"{boundary:.3f} s"
        )

    # ---------------------------------------------------------
    # 5. Print segments
    # ---------------------------------------------------------

    print(
        f"\nDetected segments: "
        f"{len(result['segments'])}"
    )

    for segment in result["segments"]:
        print(
            f"  Segment "
            f"{segment['segment_id']:02d}: "
            f"{segment['start_time']:.3f} s "
            f"-> "
            f"{segment['end_time']:.3f} s "
            f"("
            f"{segment['duration']:.3f} s"
            f")"
        )

    # ---------------------------------------------------------
    # 6. Plot novelty curve
    # ---------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    novelty = result["novelty_curve"]

    novelty_times = librosa.frames_to_time(
        range(len(novelty)),
        sr=audio.sample_rate,
        hop_length=HOP_LENGTH,
    )

    plt.figure(figsize=(16, 6))

    plt.plot(
        novelty_times,
        novelty,
    )

    for boundary in result["boundaries"]:
        plt.axvline(
            boundary,
            linestyle="--",
            linewidth=1.0,
            alpha=0.7,
        )

    plt.title(
        "Structural Novelty Curve with Detected Boundaries"
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Novelty")

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "test_song_structure_novelty.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
    )

    plt.close()

    print(
        f"\nPlot saved to:"
        f"\n  {output_path}"
    )

    print("\nStructural segmentation test complete.")


if __name__ == "__main__":
    main()