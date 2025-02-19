from warnings import filterwarnings
filterwarnings("ignore", category=UserWarning)
filterwarnings("ignore", category=FutureWarning)

from collections.abc import Iterable
from numpy import ndarray
from torch import Tensor, device
from torch.cuda import is_available as is_cuda_available, empty_cache as clear_cuda_cache

from .asr_model import ASRModel
from .utils.prepare_audio import prepare_audio
from .utils.gigaam import GigaAMASR, load_model


GIGAAM_MODELS_MAP = {
    "GigaAM CTC-1": "v1_ctc",
    "GigaAM RNNT-1": "v1_rnnt",
    "GigaAM CTC-2": "v2_ctc",
    "GigaAM RNNT-2": "v2_rnnt",
}


class GigaAM(ASRModel):
    def __init__(self, model_name: str) -> None:
        assert model_name in GIGAAM_MODELS_MAP.keys(), f"Invalid model name: {model_name}. Use another class for the model!"

        self.name = model_name
        self.__model: GigaAMASR = load_model(
            model_name=GIGAAM_MODELS_MAP[model_name],
            fp16_encoder=False,
            device="cpu",
        )


    def transcribe_file(self, file_path: str) -> str:
        transcription = self.__model.transcribe(file_path)

        if is_cuda_available():
            clear_cuda_cache()

        assert isinstance(transcription, str), f"Unexpected transcription return type: expected str, got {type(transcription)}"

        return transcription


    def transcribe_wav(self, wav: Tensor | ndarray | Iterable, sample_rate: int = 16_000) -> str:
        wav = prepare_audio(wav, sample_rate)

        transcription = self.__model.transcribe(wav)

        del wav

        if is_cuda_available():
            clear_cuda_cache()

        assert isinstance(transcription, str), f"Unexpected transcription return type: expected str, got {type(transcription)}"

        return transcription


    def move_to_device(self, new_device: str | device) -> None:
        if not isinstance(new_device, device):
            new_device = device(new_device)

        if self.__model._device != new_device:
            self.__model.to(new_device)


    def get_current_device(self) -> str:
        return self.__model._device.type