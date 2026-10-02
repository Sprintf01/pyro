
from pathlib import Path
import sys
import json
import time

import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons

import sounddevice as sd

# -------------------------------------------------------------
# Project root
# -------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.audio import load_audio
from src.segmentation.structure import segment_audio_structure


# -------------------------------------------------------------
# Configuration
# -------------------------------------------------------------

HOP_LENGTH = 512
UPDATE_INTERVAL_MS = 40


# -------------------------------------------------------------
# Audio playback controller
# -------------------------------------------------------------

class AudioController:
    def __init__(
        self,
        waveform: np.ndarray,
        sample_rate: int,
    ):
        self.waveform = waveform
        self.sample_rate = sample_rate

        self.position = 0.0
        self.playing = False
        self.started_at = None

    def current_time(self) -> float:
        if not self.playing:
            return self.position

        elapsed = time.perf_counter() - self.started_at
        current = self.position + elapsed

        duration = len(self.waveform) / self.sample_rate

        if current >= duration:
            self.position = duration
            self.playing = False
            return duration

        return current

    def play(self):
        if self.playing:
            return

        self.position = self.current_time()
        self.started_at = time.perf_counter()
        self.playing = True

        start_sample = int(
            self.position * self.sample_rate
        )

        if start_sample >= len(self.waveform):
            self.stop()
            return

        sd.stop()

        sd.play(
            self.waveform[start_sample:],
            self.sample_rate,
        )

    def pause(self):
        if not self.playing:
            return

        self.position = self.current_time()

        sd.stop()

        self.playing = False
        self.started_at = None

    def stop(self):
        sd.stop()

        self.position = 0.0
        self.playing = False
        self.started_at = None

    def seek(self, seconds: float):
        duration = (
            len(self.waveform)
            / self.sample_rate
        )

        seconds = max(
            0.0,
            min(seconds, duration),
        )

        was_playing = self.playing

        sd.stop()

        self.position = seconds
        self.playing = False
        self.started_at = None

        if was_playing:
            self.play()


# -------------------------------------------------------------
# Data loading
# -------------------------------------------------------------

def load_analysis_data(
    audio_path: Path,
):
    stem = audio_path.stem

    analysis_dir = (
        PROJECT_ROOT
        / "outputs"
        / "analysis"
    )

    features_path = (
        analysis_dir
        / f"{stem}_features.csv"
    )

    stft_path = (
        analysis_dir
        / f"{stem}_stft.npz"
    )

    beats_path = (
        analysis_dir
        / f"{stem}_beats.json"
    )

    downbeats_path = (
        analysis_dir
        / f"{stem}_downbeats.json"
    )

    density_path = (
        analysis_dir
        / f"{stem}_density.json"
    )

    metadata_path = (
        analysis_dir
        / f"{stem}_metadata.json"
    )

    required = [
        features_path,
        stft_path,
        beats_path,
        downbeats_path,
    ]

    for path in required:
        if not path.exists():
            raise FileNotFoundError(
                f"Required analysis file not found:\n{path}\n\n"
                "Run analyze_track.py first."
            )

    features = pd.read_csv(
        features_path
    )

    stft = np.load(
        stft_path
    )

    with open(
        beats_path,
        "r",
        encoding="utf-8",
    ) as f:
        beats = json.load(f)

    with open(
        downbeats_path,
        "r",
        encoding="utf-8",
    ) as f:
        downbeats = json.load(f)

    density = None

    if density_path.exists():
        with open(
            density_path,
            "r",
            encoding="utf-8",
        ) as f:
            density = json.load(f)

    metadata = {}

    if metadata_path.exists():
        with open(
            metadata_path,
            "r",
            encoding="utf-8",
        ) as f:
            metadata = json.load(f)

    return (
        features,
        stft,
        beats,
        downbeats,
        density,
        metadata,
    )


# -------------------------------------------------------------
# Plot inspector
# -------------------------------------------------------------

class AnalysisInspector:

    def __init__(
        self,
        audio_path: Path,
    ):
        self.audio_path = audio_path

        print("Loading audio...")

        audio = load_audio(
            audio_path
        )

        self.waveform = audio.waveform
        self.sample_rate = audio.sample_rate

        self.duration = (
            len(self.waveform)
            / self.sample_rate
        )

        print(
            f"Audio duration: "
            f"{self.duration:.3f} s"
        )

        print()
        print("Loading analysis data...")

        (
            self.features,
            self.stft,
            self.beats,
            self.downbeats,
            self.density,
            self.metadata,
        ) = load_analysis_data(
            audio_path
        )

        self.times = (
            self.features["time"]
            .to_numpy()
        )

        self.audio = AudioController(
            self.waveform,
            self.sample_rate,
        )

        self.current_plot = "Waveform + Beats"

        self.fig = None
        self.ax = None
        self.cursor = None

        self.radio = None

        self._build_interface()

    # ---------------------------------------------------------
    # Interface
    # ---------------------------------------------------------

    def _build_interface(self):

        self.fig = plt.figure(
            figsize=(16, 9)
        )

        self.fig.canvas.manager.set_window_title(
            "Music Cue System — Analysis Inspector"
        )

        self.ax = self.fig.add_axes(
            [0.08, 0.17, 0.72, 0.73]
        )

        radio_ax = self.fig.add_axes(
            [0.82, 0.32, 0.16, 0.48]
        )

        self.radio = RadioButtons(
            radio_ax,
            [
                "Waveform + Beats",
                "Beats + Downbeats",
                "RMS Energy",
                "Onset Strength",
                "Spectral Features",
                "Spectrogram",
                "Segmentation",
            ],
        )

        self.radio.on_clicked(
            self.change_plot
        )

        # -----------------------------------------------------
        # Buttons
        # -----------------------------------------------------

        play_ax = self.fig.add_axes(
            [0.08, 0.06, 0.12, 0.05]
        )

        pause_ax = self.fig.add_axes(
            [0.22, 0.06, 0.12, 0.05]
        )

        stop_ax = self.fig.add_axes(
            [0.36, 0.06, 0.12, 0.05]
        )

        self.play_button = Button(
            play_ax,
            "Play",
        )

        self.pause_button = Button(
            pause_ax,
            "Pause",
        )

        self.stop_button = Button(
            stop_ax,
            "Stop",
        )

        self.play_button.on_clicked(
            lambda event: self.audio.play()
        )

        self.pause_button.on_clicked(
            lambda event: self.audio.pause()
        )

        self.stop_button.on_clicked(
            lambda event: self.audio.stop()
        )

        self.fig.canvas.mpl_connect(
            "key_press_event",
            self.handle_key,
        )

        self.fig.canvas.mpl_connect(
            "button_press_event",
            self.handle_mouse,
        )

        self.draw_plot()

        self.timer = self.fig.canvas.new_timer(
            interval=UPDATE_INTERVAL_MS
        )

        self.timer.add_callback(
            self.update_cursor
        )

        self.timer.start()

    # ---------------------------------------------------------
    # Plot switching
    # ---------------------------------------------------------

    def change_plot(
        self,
        label,
    ):
        self.current_plot = label
        self.draw_plot()

    # ---------------------------------------------------------
    # Clear plot
    # ---------------------------------------------------------

    def clear_axis(self):
        self.ax.clear()

    # ---------------------------------------------------------
    # Waveform + beats
    # ---------------------------------------------------------

    def plot_waveform_beats(self):

        librosa.display.waveshow(
            self.waveform,
            sr=self.sample_rate,
            ax=self.ax,
        )

        for beat_time in self.beats[
            "beat_times"
        ]:

            self.ax.axvline(
                beat_time,
                linestyle="--",
                linewidth=0.8,
                alpha=0.30,
            )

        self.ax.set_title(
            "Waveform with Detected Beats"
        )

        self.ax.set_xlabel(
            "Time (s)"
        )

        self.ax.set_ylabel(
            "Amplitude"
        )

    # ---------------------------------------------------------
    # Beats + downbeats
    # ---------------------------------------------------------

    def plot_beats_downbeats(self):

        display_duration = min(
            30.0,
            self.duration,
        )

        samples = int(
            display_duration
            * self.sample_rate
        )

        display_waveform = (
            self.waveform[:samples]
        )

        display_time = (
            np.arange(samples)
            / self.sample_rate
        )

        self.ax.plot(
            display_time,
            display_waveform,
            alpha=0.55,
        )

        beat_times = np.asarray(
            self.downbeats[
                "beat_times"
            ]
        )

        beat_positions = np.asarray(
            self.downbeats[
                "beat_positions"
            ]
        )

        for (
            beat_time,
            beat_position,
        ) in zip(
            beat_times,
            beat_positions,
        ):

            if beat_time > display_duration:
                break

            if int(beat_position) == 1:

                self.ax.axvline(
                    beat_time,
                    linestyle="-",
                    linewidth=2.0,
                    alpha=0.9,
                )

                self.ax.text(
                    beat_time,
                    0.95,
                    "1",
                    transform=(
                        self.ax.get_xaxis_transform()
                    ),
                    ha="center",
                    va="top",
                    fontsize=10,
                    fontweight="bold",
                )

            else:

                self.ax.axvline(
                    beat_time,
                    linestyle="--",
                    linewidth=0.8,
                    alpha=0.35,
                )

                self.ax.text(
                    beat_time,
                    0.95,
                    str(int(beat_position)),
                    transform=(
                        self.ax.get_xaxis_transform()
                    ),
                    ha="center",
                    va="top",
                    fontsize=8,
                )

        self.ax.set_xlim(
            0,
            display_duration,
        )

        self.ax.set_title(
            "Madmom Beat Positions and Downbeats"
        )

        self.ax.set_xlabel(
            "Time (s)"
        )

        self.ax.set_ylabel(
            "Amplitude"
        )

    # ---------------------------------------------------------
    # RMS
    # ---------------------------------------------------------

    def plot_rms(self):

        if "rms" not in self.features:
            self.ax.text(
                0.5,
                0.5,
                "RMS data unavailable",
                ha="center",
                va="center",
                transform=self.ax.transAxes,
            )
            return

        self.ax.plot(
            self.times,
            self.features["rms"],
        )

        self.ax.set_title(
            "RMS Energy"
        )

        self.ax.set_xlabel(
            "Time (s)"
        )

        self.ax.set_ylabel(
            "RMS"
        )

    # ---------------------------------------------------------
    # Onset strength
    # ---------------------------------------------------------

    def plot_onset(self):

        if "onset_strength" not in self.features:
            self.ax.text(
                0.5,
                0.5,
                "Onset data unavailable",
                ha="center",
                va="center",
                transform=self.ax.transAxes,
            )
            return

        self.ax.plot(
            self.times,
            self.features[
                "onset_strength"
            ],
        )

        self.ax.set_title(
            "Onset Strength"
        )

        self.ax.set_xlabel(
            "Time (s)"
        )

        self.ax.set_ylabel(
            "Onset Strength"
        )

    # ---------------------------------------------------------
    # Spectral features
    # ---------------------------------------------------------

    def plot_spectral(self):

        columns = [
            "spectral_centroid",
            "spectral_bandwidth",
            "spectral_rolloff",
            "spectral_flatness",
        ]

        for column in columns:

            if column in self.features:

                self.ax.plot(
                    self.times,
                    self.features[column],
                    label=column,
                )

        self.ax.set_title(
            "Spectral Features"
        )

        self.ax.set_xlabel(
            "Time (s)"
        )

        self.ax.set_ylabel(
            "Value"
        )

        self.ax.legend()

    # ---------------------------------------------------------
    # Spectrogram
    # ---------------------------------------------------------

    def plot_spectrogram(self):

        magnitude_db = self.stft[
            "magnitude_db"
        ]

        librosa.display.specshow(
            magnitude_db,
            sr=self.sample_rate,
            hop_length=HOP_LENGTH,
            x_axis="time",
            y_axis="hz",
            ax=self.ax,
        )

        self.ax.set_title(
            "STFT Magnitude Spectrogram"
        )

        self.fig.colorbar(
            self.ax.images[0],
            ax=self.ax,
            format="%+2.0f dB",
        )

    # ---------------------------------------------------------
    # Segmentation
    # ---------------------------------------------------------

    def plot_segmentation(self):

        print(
            "\nComputing structural segmentation..."
        )

        result = segment_audio_structure(
            waveform=self.waveform,
            sample_rate=self.sample_rate,
            hop_length=HOP_LENGTH,
        )

        novelty = np.asarray(
            result["novelty_curve"]
        )

        novelty_times = np.asarray(
            result["novelty_times"]
        )

        boundaries = np.asarray(
            result["boundaries"]
        )

        # -----------------------------------------------------
        # IMPORTANT:
        #
        # Boundaries must already be expressed in seconds.
        # We do NOT use frame indices here.
        # -----------------------------------------------------

        self.ax.plot(
            novelty_times,
            novelty,
            linewidth=1.0,
            label="Novelty",
        )

        for boundary in boundaries:

            self.ax.axvline(
                float(boundary),
                linestyle="--",
                linewidth=1.2,
                alpha=0.75,
            )

        self.ax.set_title(
            "Structural Novelty and Detected Boundaries"
        )

        self.ax.set_xlabel(
            "Time (s)"
        )

        self.ax.set_ylabel(
            "Novelty"
        )

        self.ax.set_xlim(
            0,
            self.duration,
        )

        self.ax.legend()

    # ---------------------------------------------------------
    # Draw selected plot
    # ---------------------------------------------------------

    def draw_plot(self):

        self.clear_axis()

        if self.current_plot == "Waveform + Beats":

            self.plot_waveform_beats()

        elif self.current_plot == "Beats + Downbeats":

            self.plot_beats_downbeats()

        elif self.current_plot == "RMS Energy":

            self.plot_rms()

        elif self.current_plot == "Onset Strength":

            self.plot_onset()

        elif self.current_plot == "Spectral Features":

            self.plot_spectral()

        elif self.current_plot == "Spectrogram":

            self.plot_spectrogram()

        elif self.current_plot == "Segmentation":

            self.plot_segmentation()

        self.create_cursor()

        self.fig.canvas.draw_idle()

    # ---------------------------------------------------------
    # Cursor
    # ---------------------------------------------------------

    def create_cursor(self):

        self.cursor = self.ax.axvline(
            0.0,
            linewidth=2.0,
            alpha=0.9,
        )

    def update_cursor(self):

        if self.cursor is None:
            return

        current_time = (
            self.audio.current_time()
        )

        self.cursor.set_xdata(
            [current_time, current_time]
        )

        self.fig.canvas.draw_idle()

    # ---------------------------------------------------------
    # Keyboard controls
    # ---------------------------------------------------------

    def handle_key(
        self,
        event,
    ):

        if event.key == " ":

            if self.audio.playing:
                self.audio.pause()
            else:
                self.audio.play()

        elif event.key == "escape":

            self.audio.stop()

        elif event.key == "left":

            current = (
                self.audio.current_time()
            )

            self.audio.seek(
                current - 5.0
            )

        elif event.key == "right":

            current = (
                self.audio.current_time()
            )

            self.audio.seek(
                current + 5.0
            )

    # ---------------------------------------------------------
    # Mouse seeking
    # ---------------------------------------------------------

    def handle_mouse(
        self,
        event,
    ):

        if event.inaxes != self.ax:
            return

        if event.xdata is None:
            return

        self.audio.seek(
            float(event.xdata)
        )


# -------------------------------------------------------------
# Main
# -------------------------------------------------------------

def main():

    if len(sys.argv) != 2:

        print(
            "Usage:\n"
            "  python scripts\\inspect_analysis.py "
            "<audio_file>"
        )

        sys.exit(1)

    input_path = Path(
        sys.argv[1]
    )

    if not input_path.exists():

        raise FileNotFoundError(
            f"Audio file not found: "
            f"{input_path}"
        )

    print()
    print("=" * 60)
    print(
        "INTERACTIVE MUSIC ANALYSIS INSPECTOR"
    )
    print("=" * 60)
    print()

    inspector = AnalysisInspector(
        input_path
    )

    print()
    print("Available plots:")
    print("  1. Waveform + Beats")
    print("  2. Beats + Downbeats")
    print("  3. RMS Energy")
    print("  4. Onset Strength")
    print("  5. Spectral Features")
    print("  6. Spectrogram")
    print("  7. Segmentation")

    print()
    print("Controls:")
    print("  SPACE       Play / Pause")
    print("  ESC         Stop")
    print("  LEFT        -5 seconds")
    print("  RIGHT       +5 seconds")
    print("  Mouse       Click plot to seek")
    print()
    print(
        "Select plots using the buttons on the right."
    )
    print()
    print(
        "Close the plot window to exit."
    )

    plt.show()

    inspector.audio.stop()


if __name__ == "__main__":
    main()
