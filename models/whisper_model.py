from numpy import ndarray
from torch import Tensor, device
from torch.cuda import is_available as is_cuda_available, empty_cache as clear_cuda_cache
from whisper import load_model, transcribe

from .asr_model import ASRModel
from .utils.prepare_audio import prepare_audio


WHISPER_MODELS_MAP = {
    "Whisper Tiny": "tiny",
    "Whisper Base": "base",
    "Whisper Small": "small",
    "Whisper Medium": "medium",
    "Whisper Large-v1": "large-v1",
    "Whisper Large-v2": "large-v2",
    "Whisper Large-v3": "large-v3",
    "Whisper Turbo": "turbo",
}


class WhisperModel(ASRModel):
    def __init__(self, model_name: str) -> None:
        assert model_name in WHISPER_MODELS_MAP.keys(), f"Invalid model name: {model_name}. Use another class for the model!"

        self.name: str = model_name
        self.__model = load_model(WHISPER_MODELS_MAP[model_name], device="cpu")


    def transcribe_file(self, file_path: str) -> str:
        result = transcribe(self.__model, file_path, task="transcribe", language="ru")

        if is_cuda_available():
            clear_cuda_cache()

        transcription = result["text"]
        assert isinstance(transcription, str), f"Unexpected transcription return type: expected str, got {type(transcription)}"

        return transcription


    def transcribe_wav(self, wav: Tensor | ndarray, sample_rate: int = 16_000) -> str:
        wav = prepare_audio(wav, sample_rate)

        if wav.device != self.__model.device:
            wav = wav.to(self.__model.device)

        result = transcribe(self.__model, wav, task="transcribe", language="ru")

        del wav

        if is_cuda_available():
            clear_cuda_cache()

        transcription = result["text"]
        assert isinstance(transcription, str), f"Unexpected transcription return type: expected str, got {type(transcription)}"

        return transcription


    def move_to_device(self, new_device: str | device) -> None:
        if not isinstance(new_device, device):
            new_device = device(new_device)

        if self.__model.device != new_device:
            self.__model.to(new_device)


    def get_current_device(self) -> str:
        return self.__model.device.type