from collections.abc import Iterable
from numpy import asarray, ndarray
from torch import Tensor, from_numpy, float32, mean
from torchaudio.functional import resample

from .constants import BASE_SAMPLE_RATE


def prepare_audio(audio: Tensor | ndarray | Iterable, sample_rate: int = BASE_SAMPLE_RATE) -> Tensor:
    """
    Prepares audio for ASR model inference.
    Converts audio to float32 PyTorch tensor, resamples it to the base sample rate, and normalizes it if needed.
    """

    if not isinstance(audio, Tensor):
        if not isinstance(audio, ndarray):
            try:
                audio = asarray(audio)
            except ValueError as exc:
                raise ValueError(
                    "Passed audio content is not convertible to numpy.ndarray!"
                    f"Expected Iterable Python object or numpy.ndarray or torch.Tensor, got {type(audio)}"
                ) from exc

        assert "float" in audio.dtype.name or "int" in audio.dtype.name, f"Audio should be a float or int array, got {audio.dtype} array"

        audio = from_numpy(audio)

    if not audio.dtype.is_floating_point and audio.abs().max() > 1:
        audio = audio.float() / 32768.0

    if audio.ndim != 1:
        audio = mean(audio, dim=0)

    audio = resample(audio, orig_freq=sample_rate, new_freq=BASE_SAMPLE_RATE)

    if audio.dtype != float32:
        audio = audio.to(dtype=float32)

    return audio