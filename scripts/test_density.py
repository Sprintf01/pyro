from pathlib import Path
import sys

import numpy as np
import librosa


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.preprocessing.audio import load_audio
from src.dsp.onset import detect_onsets
from src.rhythm.beats import track_beats
from src.rhythm.density import compute_rhythmic_density


AUDIO_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "test_song.mp3"
)


def main():
    print("=" * 60)
    print("RHYTHMIC DENSITY TEST")
    print("=" * 60)

    audio = load_audio(AUDIO_PATH)

    onset_data = detect_onsets(
        audio.waveform,
        audio.sample_rate,
    )

    beat_data = track_beats(
        audio.waveform,
        audio.sample_rate,
    )

    duration_seconds = (
        len(audio.waveform)
        / audio.sample_rate
    )

    density = compute_rhythmic_density(
        onset_strength=onset_data[
            "onset_strength"
        ],
        onset_times=onset_data[
            "onset_times"
        ],
        beat_times=beat_data[
            "beat_times"
        ],
        duration_seconds=duration_seconds,
    )

    print()
    print("Audio:")
    print(
        f"  Duration: "
        f"{duration_seconds:.3f} s"
    )

    print()
    print("Onsets:")
    print(
        f"  Detected onsets: "
        f"{len(onset_data['onset_times'])}"
    )

    print()
    print("Rhythmic density:")
    print(
        f"  Windows: "
        f"{len(density['times'])}"
    )

    print(
        f"  Mean onset rate: "
        f"{np.mean(density['local_onset_rate']):.3f} "
        f"onsets/s"
    )

    print(
        f"  Mean onset strength: "
        f"{np.mean(density['local_onset_strength']):.3f}"
    )

    print(
        f"  Mean onsets/beat: "
        f"{np.mean(density['onsets_per_beat']):.3f}"
    )

    print(
        f"  Mean beat interval variation: "
        f"{np.mean(density['beat_interval_variation']):.6f} s"
    )

    print()
    print("First 10 windows:")

    for index in range(
        min(10, len(density["times"]))
    ):
        print(
            f"  {density['times'][index]:7.2f} s | "
            f"onset_rate="
            f"{density['local_onset_rate'][index]:6.2f} | "
            f"onset_strength="
            f"{density['local_onset_strength'][index]:6.3f} | "
            f"onsets/beat="
            f"{density['onsets_per_beat'][index]:5.2f} | "
            f"beat_variation="
            f"{density['beat_interval_variation'][index]:.4f}"
        )

    print()
    print("Rhythmic density test complete.")


if __name__ == "__main__":
    main()