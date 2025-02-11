from numpy import asarray, ndarray
from torch import Tensor, from_numpy, float32, mean
from torchaudio.functional import resample


def prepare_audio(audio: Tensor | ndarray | list | tuple, sample_rate: int = 16_000) -> Tensor:
    if not isinstance(audio, Tensor):
        if not isinstance(audio, ndarray):
            try:
                audio = asarray(audio)
            except ValueError:
                raise ValueError("Passed audio content is not convertible to numpy.ndarray!"
                                f"Expected 1D Python list or tuple or numpy.ndarray or torch.Tensor, got {type(audio)}")

        assert "float" in audio.dtype.name or "int" in audio.dtype.name, f"Audio should be a float or int array, got {audio.dtype} array"

        audio = from_numpy(audio)

    if not audio.dtype.is_floating_point:
        audio = audio.float() / 32768.0

    if audio.ndim != 1:
        audio = mean(audio, dim=0)

    audio = resample(audio, orig_freq=sample_rate, new_freq=16_000)

    if audio.dtype != float32:
        audio = audio.to(dtype=float32)

    return audio