from pathlib import Path

import numpy as np

from .audio import AudioData


def extract_metadata(audio: AudioData) -> dict:
    waveform = audio.waveform

    peak_amplitude = float(np.max(np.abs(waveform)))
    rms = float(np.sqrt(np.mean(np.square(waveform))))

    clipped_samples = int(
        np.sum(np.abs(waveform) >= 0.999)
    )

    near_silent_samples = int(
        np.sum(np.abs(waveform) < 1e-4)
    )

    return {
        "filename": Path(audio.source_path).name,
        "source_path": audio.source_path,
        "sample_rate": audio.sample_rate,
        "original_sample_rate": audio.original_sample_rate,
        "num_samples": int(len(waveform)),
        "duration_seconds": float(
            len(waveform) / audio.sample_rate
        ),
        "channels": 1,
        "dtype": str(waveform.dtype),
        "peak_amplitude": peak_amplitude,
        "rms": rms,
        "clipped_samples": clipped_samples,
        "near_silent_samples": near_silent_samples,
    }
