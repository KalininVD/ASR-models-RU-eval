from collections.abc import Iterable
from numpy import ndarray
from torch import Tensor, int32, device
from torchaudio import load as load_audio

from .utils.constants import BASE_SAMPLE_RATE
from .utils.prepare_audio import prepare_audio
from .utils.tone import StreamingCTCPipeline
from .asr_model import ASRModel


class TOne(ASRModel):
    def __init__(self, model_name: str) -> None:
        assert model_name.lower() in ("t-one", "tone"), f"Invalid model name: {model_name}. Use another class for the model!"

        self.name = "T-One"
        self.__pipeline: StreamingCTCPipeline = StreamingCTCPipeline.from_hugging_face()


    def transcribe_file(self, file_path: str) -> str:
        audio, orig_freq = load_audio(file_path)

        return self.transcribe_wav(audio, orig_freq)


    def transcribe_wav(self, wav: Tensor | ndarray | Iterable, sample_rate: int = BASE_SAMPLE_RATE) -> str:
        wav = prepare_audio(wav, sample_rate=sample_rate, out_dtype=int32, out_sample_rate=8_000)

        result = self.__pipeline.forward_offline(wav.cpu().numpy())
        transcription = " ".join(textphrase.text for textphrase in result)

        del wav, result

        assert isinstance(transcription, str), f"Unexpected transcription return type: expected str, got {type(transcription)}"

        return transcription


    def move_to_device(self, new_device: str | device) -> None:
        # raise NotImplementedError()
        pass


    def get_current_device(self) -> str:
        return "cpu"