from silero_vad import load_silero_vad, get_speech_timestamps
from torch import Tensor, device
from torch.cuda import is_available as is_cuda_available, empty_cache as clear_cuda_cache
from numpy import ndarray

from .constants import BASE_SAMPLE_RATE
from .prepare_audio import prepare_audio


class VADModel:
    def __init__(self) -> None:
        self.__silero_model = load_silero_vad().to("cpu")


    def get_timestamps(self, wav: Tensor | ndarray, sample_rate: int,
                       max_duration: float = 22.0, min_duration: float = 15.0, new_chunk_threshold: float = 0.2,
                       use_device: str | device = "cpu",
        ) -> list[tuple[float, float]]:

        wav = prepare_audio(wav, sample_rate)

        wav = wav.to(use_device)
        self.move_to_device(use_device)

        speech_timestamps = get_speech_timestamps(
            wav,
            self.__silero_model,
            return_seconds=True, # Return speech timestamps in seconds (default is samples)
        )

        curr_duration = 0.0
        curr_start = -1.0
        curr_end = 0.0
        boundaries: list[tuple[float, float]] = []

        # Concat segments into chunks for asr according to max/min duration
        for segment in speech_timestamps:
            start, end = segment['start'], segment['end']

            if int(curr_start) == -1:
                curr_start, curr_end, curr_duration = start, end, end - start
                continue

            if (
                curr_duration > min_duration and start - curr_end > new_chunk_threshold
            ) or (
                curr_duration + (end - curr_end) > max_duration
            ):
                boundaries.append((curr_start, curr_end))
                curr_start = start

            curr_end = end
            curr_duration = curr_end - curr_start

        if curr_duration != 0:
            boundaries.append((curr_start, curr_end))

        return boundaries
    

    def split_wav_tensor(self, wav: Tensor | ndarray, sample_rate: int,
                         max_duration: float = 22.0, min_duration: float = 15.0, new_chunk_threshold: float = 0.2,
                         use_device: str | device = "cpu",
        ) -> list[Tensor] | list[ndarray]:

        wav = prepare_audio(wav, sample_rate)

        wav = wav.to(use_device)
        self.move_to_device(use_device)

        speech_timestamps = get_speech_timestamps(
            wav,
            self.__silero_model,
        )

        curr_duration = 0
        curr_start = -1
        curr_end = 0

        chunks = []

        # Concat segments into chunks for asr according to max/min duration
        for segment in speech_timestamps:
            start, end = segment['start'], segment['end']

            if curr_start == -1:
                curr_start, curr_end, curr_duration = start, end, end - start
                continue

            if (
                curr_duration > min_duration * BASE_SAMPLE_RATE and start - curr_end > new_chunk_threshold * BASE_SAMPLE_RATE
            ) or (
                curr_duration + (end - curr_end) > max_duration * BASE_SAMPLE_RATE
            ):
                chunks.append(wav[curr_start : curr_end + 1])
                curr_start = start

            curr_end = end
            curr_duration = curr_end - curr_start

        if curr_duration != 0:
            chunks.append(wav[curr_start : curr_end + 1])

        return chunks


    def move_to_device(self, new_device: str | device) -> None:
        if is_cuda_available():
            clear_cuda_cache()

        if not isinstance(new_device, device):
            new_device = device(new_device)

        self.__silero_model.to(new_device)

        if is_cuda_available():
            clear_cuda_cache()