from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


TARGET_SAMPLE_RATE = 44_100


class AudioData:
    def __init__(
        self,
        waveform: np.ndarray,
        sample_rate: int,
        original_sample_rate: int,
        source_path: str,
    ):
        self.waveform = waveform
        self.sample_rate = sample_rate
        self.original_sample_rate = original_sample_rate
        self.source_path = source_path


def load_audio(
    path: str | Path,
    target_sample_rate: int = TARGET_SAMPLE_RATE,
) -> AudioData:
    """
    Load an audio file, convert it to mono, and resample it.

    The original file is never modified.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")

    try:
        waveform, original_sr = librosa.load(
            path,
            sr=None,
            mono=True,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not decode audio file: {path}"
        ) from exc

    if waveform.size == 0:
        raise ValueError("Audio file contains no samples.")

    if not np.all(np.isfinite(waveform)):
        raise ValueError("Audio contains NaN or Inf values.")

    if original_sr != target_sample_rate:
        waveform = librosa.resample(
            waveform,
            orig_sr=original_sr,
            target_sr=target_sample_rate,
        )

    waveform = waveform.astype(np.float32)

    duration = len(waveform) / target_sample_rate

    if duration <= 0:
        raise ValueError("Audio duration must be greater than zero.")

    return AudioData(
        waveform=waveform,
        sample_rate=target_sample_rate,
        original_sample_rate=original_sr,
        source_path=str(path),
    )


def save_standardized_wav(
    audio: AudioData,
    output_path: str | Path,
) -> None:
    """
    Save the standardized mono float32 waveform.

    No peak normalization is applied.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sf.write(
        output_path,
        audio.waveform,
        audio.sample_rate,
        subtype="FLOAT",
    )
