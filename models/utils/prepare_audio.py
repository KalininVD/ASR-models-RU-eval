from collections.abc import Iterable
from numpy import asarray, ndarray
from torch import Tensor, from_numpy, int32, float32, log, mean
from torchaudio.functional import resample

from .constants import BASE_SAMPLE_RATE


def prepare_audio(audio: Tensor | ndarray | Iterable, sample_rate: int = BASE_SAMPLE_RATE,
                  out_dtype=float32, out_sample_rate=BASE_SAMPLE_RATE) -> Tensor:
    """
    Prepares audio for ASR model inference.
    Converts audio to int32 or float32 PyTorch tensor, resamples it to the desired sample rate, and normalizes it if needed.
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

    assert out_dtype in (int32, float32), "Only int32 and float32 are supported as output Tensor's DType"

    audio = audio.to(float32)

    if audio.ndim != 1:
        audio = mean(audio, dim=0)

    audio = resample(audio, orig_freq=sample_rate, new_freq=out_sample_rate)

    if out_dtype == int32:
        if log(audio.abs().max()).item() * 2 < log(Tensor([1, 32768])).sum().item():
            audio = audio * 32768.0

        audio = audio.to(int32)

        audio[audio <= -32769] = -32768
        audio[audio >= 32768] = 32767
    else:
        if log(audio.abs().max()).item() * 2 > log(Tensor([1, 32768])).sum().item():
            audio = audio / 32768.0

        audio = audio.to(float32)

        audio[audio <= -1.0] = 1.0
        audio[audio >= 1.0] = 1.0

    return audio