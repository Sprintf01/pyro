from __future__ import annotations

import time
from typing import Callable

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button
import numpy as np
import sounddevice as sd


ANIMATION_INTERVAL_MS = 33


class AudioPlayback:
    """
    Synchronize audio playback with matplotlib plots.

    The playback time is maintained using a wall-clock timer,
    while matplotlib is responsible only for visualization.
    """

    def __init__(
        self,
        waveform: np.ndarray,
        sample_rate: int,
        duration_seconds: float,
    ):
        self.waveform = waveform
        self.sample_rate = sample_rate
        self.duration_seconds = duration_seconds

        self.playing = False
        self.current_time = 0.0

        self.start_wall_time: float | None = None
        self.start_audio_time = 0.0

    def play(self) -> None:
        if self.playing:
            return

        if self.current_time >= self.duration_seconds:
            self.current_time = 0.0

        start_sample = int(
            self.current_time * self.sample_rate
        )

        self.start_audio_time = self.current_time
        self.start_wall_time = time.perf_counter()
        self.playing = True

        sd.play(
            self.waveform[start_sample:],
            samplerate=self.sample_rate,
            blocking=False,
        )

    def pause(self) -> None:
        if not self.playing:
            return

        self.current_time = self.get_time()

        self.playing = False
        self.start_wall_time = None

        sd.stop()

    def stop(self) -> None:
        sd.stop()

        self.playing = False
        self.current_time = 0.0
        self.start_wall_time = None
        self.start_audio_time = 0.0

    def get_time(self) -> float:
        if not self.playing:
            return self.current_time

        if self.start_wall_time is None:
            return self.current_time

        elapsed = (
            time.perf_counter()
            - self.start_wall_time
        )

        current_time = (
            self.start_audio_time
            + elapsed
        )

        if current_time >= self.duration_seconds:
            self.current_time = self.duration_seconds
            self.playing = False
            self.start_wall_time = None
            sd.stop()
        else:
            self.current_time = current_time

        return self.current_time


def add_playback_controls(
    figure,
    playback: AudioPlayback,
    on_time_update: Callable[[float], None] | None = None,
):
    """
    Add Play/Pause/Stop buttons, keyboard controls,
    and a synchronized animation timer.

    Keyboard:
        SPACE = Play/Pause
        ESC   = Stop
    """

    play_axis = figure.add_axes(
        [0.35, 0.025, 0.08, 0.05]
    )

    pause_axis = figure.add_axes(
        [0.46, 0.025, 0.08, 0.05]
    )

    stop_axis = figure.add_axes(
        [0.57, 0.025, 0.08, 0.05]
    )

    play_button = Button(
        play_axis,
        "Play",
    )

    pause_button = Button(
        pause_axis,
        "Pause",
    )

    stop_button = Button(
        stop_axis,
        "Stop",
    )

    play_button.on_clicked(
        lambda event: playback.play()
    )

    pause_button.on_clicked(
        lambda event: playback.pause()
    )

    stop_button.on_clicked(
        lambda event: playback.stop()
    )

    def on_key(event):
        if event.key == " ":
            if playback.playing:
                playback.pause()
            else:
                playback.play()

        elif event.key == "escape":
            playback.stop()

    figure.canvas.mpl_connect(
        "key_press_event",
        on_key,
    )

    def update(frame):
        current_time = playback.get_time()

        if on_time_update is not None:
            on_time_update(current_time)

        return ()

    animation = FuncAnimation(
        figure,
        update,
        interval=ANIMATION_INTERVAL_MS,
        blit=False,
        cache_frame_data=False,
    )

    figure._audio_playback_animation = animation

    return animation