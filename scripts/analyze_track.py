import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json

import matplotlib.pyplot as plt
import librosa.display

from src.pipeline import analyze_audio


def create_plots(
    result: dict,
    output_dir: Path,
    track_name: str,
) -> None:
    features = result["features"]
    stft = result["stft"]
    times = result["times"]
    waveform = result["waveform"]

    beats = result["beats"]
    downbeats = result["downbeats"]
    density = result["density"]

    sample_rate = result["metadata"]["sample_rate"]

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # 1. Waveform with detected beats
    # ---------------------------------------------------------

    plt.figure(figsize=(14, 5))

    librosa.display.waveshow(
        waveform,
        sr=sample_rate,
    )

    for beat_time in beats["beat_times"]:
        plt.axvline(
            beat_time,
            alpha=0.30,
            linewidth=0.8,
            linestyle="--",
        )

    plt.title("Waveform with Detected Beats")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()

    plt.savefig(
        plot_dir / f"{track_name}_waveform_beats.png",
        dpi=150,
    )

    plt.close()

    # ---------------------------------------------------------
    # 2. Madmom beat positions and downbeats
    # ---------------------------------------------------------

    beat_times = downbeats["beat_times"]
    beat_positions = downbeats["beat_positions"]

    # Show only the first 30 seconds for readable inspection.
    display_duration = min(
        30.0,
        len(waveform) / sample_rate,
    )

    waveform_samples = int(
        display_duration * sample_rate
    )

    display_waveform = waveform[:waveform_samples]

    display_time = list(range(len(display_waveform)))

    display_time = [
        sample / sample_rate
        for sample in display_time
    ]

    plt.figure(figsize=(16, 6))

    plt.plot(
        display_time,
        display_waveform,
        alpha=0.6,
    )

    for beat_time, beat_position in zip(
        beat_times,
        beat_positions,
    ):
        if beat_time > display_duration:
            break

        if beat_position == 1:
            # Start of a new bar.
            plt.axvline(
                beat_time,
                linestyle="-",
                linewidth=2.0,
                alpha=0.9,
            )

            plt.text(
                beat_time,
                0.95,
                "1",
                transform=plt.gca().get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=10,
                fontweight="bold",
            )

        else:
            # Other beats.
            plt.axvline(
                beat_time,
                linestyle="--",
                linewidth=0.8,
                alpha=0.35,
            )

            plt.text(
                beat_time,
                0.95,
                str(int(beat_position)),
                transform=plt.gca().get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=8,
            )

    plt.title(
        "Madmom Beat Positions and Downbeats"
    )
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    plt.xlim(
        0,
        display_duration,
    )

    plt.tight_layout()

    plt.savefig(
        plot_dir
        / f"{track_name}_madmom_beats_downbeats.png",
        dpi=150,
    )

    plt.close()

    # ---------------------------------------------------------
    # 3. Spectrogram
    # ---------------------------------------------------------

    plt.figure(figsize=(14, 6))

    librosa.display.specshow(
        stft["magnitude_db"],
        sr=sample_rate,
        hop_length=512,
        x_axis="time",
        y_axis="hz",
    )

    plt.colorbar(format="%+2.0f dB")
    plt.title("STFT Magnitude Spectrogram")
    plt.tight_layout()

    plt.savefig(
        plot_dir / f"{track_name}_spectrogram.png",
        dpi=150,
    )

    plt.close()

    # ---------------------------------------------------------
    # 4. RMS
    # ---------------------------------------------------------

    if "rms" in features.columns:

        plt.figure(figsize=(14, 5))

        plt.plot(
            times,
            features["rms"],
        )

        plt.title("RMS Energy")
        plt.xlabel("Time (s)")
        plt.ylabel("RMS")
        plt.tight_layout()

        plt.savefig(
            plot_dir / f"{track_name}_rms.png",
            dpi=150,
        )

        plt.close()

    # ---------------------------------------------------------
    # 5. Spectral features
    # ---------------------------------------------------------

    spectral_columns = [
        column
        for column in [
            "spectral_centroid",
            "spectral_bandwidth",
            "spectral_rolloff",
            "spectral_flatness",
        ]
        if column in features.columns
    ]

    if spectral_columns:

        plt.figure(figsize=(14, 6))

        for column in spectral_columns:
            plt.plot(
                times,
                features[column],
                label=column,
            )

        plt.title("Spectral Features")
        plt.xlabel("Time (s)")
        plt.ylabel("Value")
        plt.legend()

        plt.tight_layout()

        plt.savefig(
            plot_dir / f"{track_name}_spectral.png",
            dpi=150,
        )

        plt.close()

    # ---------------------------------------------------------
    # 6. Onset strength
    # ---------------------------------------------------------

    if "onset_strength" in features.columns:

        plt.figure(figsize=(14, 5))

        plt.plot(
            times,
            features["onset_strength"],
        )

        plt.title("Onset Strength")
        plt.xlabel("Time (s)")
        plt.ylabel("Onset Strength")

        plt.tight_layout()

        plt.savefig(
            plot_dir / f"{track_name}_onset.png",
            dpi=150,
        )

        plt.close()

    # ---------------------------------------------------------
    # 7. Rhythmic density
    # ---------------------------------------------------------

    density_times = density["times"]

    plt.figure(figsize=(14, 8))

    plt.plot(
        density_times,
        density["local_onset_rate"],
        label="Local Onset Rate",
    )

    plt.plot(
        density_times,
        density["local_onset_strength"],
        label="Local Onset Strength",
    )

    plt.plot(
        density_times,
        density["onsets_per_beat"],
        label="Onsets per Beat",
    )

    plt.plot(
        density_times,
        density["beat_interval_variation"],
        label="Beat Interval Variation",
    )

    plt.title("Rhythmic Density Features")
    plt.xlabel("Time (s)")
    plt.ylabel("Value")
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        plot_dir
        / f"{track_name}_rhythmic_density.png",
        dpi=150,
    )

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze a music track using the "
            "Stage 1 DSP and Stage 2 rhythm pipeline."
        )
    )

    parser.add_argument(
        "audio",
        type=str,
        help="Path to the input audio file.",
    )

    args = parser.parse_args()

    input_path = Path(args.audio)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Audio file not found: {input_path}"
        )

    output_dir = Path("outputs") / "analysis"
    processed_dir = Path("data") / "processed"

    print()
    print("=" * 60)
    print("MUSIC CUE SYSTEM — STAGE 1 DSP ANALYSIS")
    print("=" * 60)

    print(f"\nInput: {input_path}")

    result = analyze_audio(
        input_path=input_path,
        processed_dir=processed_dir,
        output_dir=output_dir,
    )

    # ---------------------------------------------------------
    # Print metadata
    # ---------------------------------------------------------

    print("\nAudio metadata:")

    for key, value in result["metadata"].items():
        print(f"  {key}: {value}")

    # ---------------------------------------------------------
    # Print rhythm analysis
    # ---------------------------------------------------------

    print("\nRhythm analysis:")

    print(
        f"  Estimated BPM: "
        f"{result['rhythm']['tempo_bpm']:.2f}"
    )

    print("  Tempo candidates:")

    for candidate in result["rhythm"]["tempo_candidates"]:
        print(
            f"    {candidate['bpm']:.2f} BPM "
            f"(strength={candidate['strength']:.4f})"
        )

    # ---------------------------------------------------------
    # Print beat tracking results
    # ---------------------------------------------------------

    beats = result["beats"]

    print("\nBeat tracking:")

    print(
        f"  Tracker BPM: "
        f"{beats['tracker_tempo_bpm']:.2f}"
    )

    print(
        f"  Detected beats: "
        f"{len(beats['beat_times'])}"
    )

    if beats["mean_beat_interval"] is not None:

        print(
            f"  Mean beat interval: "
            f"{beats['mean_beat_interval']:.4f} s"
        )

        print(
            f"  BPM from beat intervals: "
            f"{beats['estimated_bpm']:.2f}"
        )

    print("  First 20 beat timestamps:")

    for i, beat_time in enumerate(
        beats["beat_times"][:20],
        start=1,
    ):
        print(
            f"    Beat {i:02d}: "
            f"{beat_time:.3f} s"
        )

    # ---------------------------------------------------------
    # Print downbeat tracking results
    # ---------------------------------------------------------

    downbeats = result["downbeats"]

    print("\nDownbeat tracking:")

    print(
        f"  Detected beats: "
        f"{downbeats['num_beats']}"
    )

    print(
        f"  Detected downbeats: "
        f"{downbeats['num_downbeats']}"
    )

    print("  First 20 downbeats:")

    for i, downbeat_time in enumerate(
        downbeats["downbeat_times"][:20],
        start=1,
    ):
        print(
            f"    Downbeat {i:02d}: "
            f"{downbeat_time:.3f} s"
        )

    # ---------------------------------------------------------
    # Create plots
    # ---------------------------------------------------------

    print("\nGenerating diagnostic plots...")

    create_plots(
        result=result,
        output_dir=output_dir,
        track_name=input_path.stem,
    )

    # ---------------------------------------------------------
    # Save metadata
    # ---------------------------------------------------------

    metadata_path = (
        output_dir
        / f"{input_path.stem}_metadata.json"
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            result["metadata"],
            f,
            indent=4,
        )

    # ---------------------------------------------------------
    # Complete
    # ---------------------------------------------------------

    print("\nAnalysis complete.")

    print("\nGenerated:")

    print(
        f"  Standardized WAV: "
        f"{result['metadata']['standardized_audio']}"
    )

    print(
        f"  Features:         "
        f"{result['metadata']['feature_file']}"
    )

    print(
        f"  STFT:             "
        f"{result['metadata']['stft_file']}"
    )

    print(
        f"  Rhythm:           "
        f"{result['metadata']['rhythm_file']}"
    )

    print(
        f"  Beats:            "
        f"{result['metadata']['beats_file']}"
    )

    print(
        f"  Downbeats:        "
        f"{result['metadata']['downbeats_file']}"
    )

    print(
        f"  Density:          "
        f"{result['metadata']['density_file']}"
    )

    print(
        f"  Metadata:         "
        f"{metadata_path}"
    )

    print(
        f"  Plots:            "
        f"{output_dir / 'plots'}"
    )


if __name__ == "__main__":
    main()