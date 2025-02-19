import nemo.utils
nemo.utils.logging.setLevel('CRITICAL')

from numpy import ndarray
from torch import Tensor, device
from torch.cuda import is_available as is_cuda_available, empty_cache as clear_cuda_cache
from nemo.collections.asr.models import EncDecCTCModelBPE, EncDecHybridRNNTCTCBPEModel, ASRModel as NeMoASRModel

from .asr_model import ASRModel
from .utils.prepare_audio import prepare_audio


NVIDIA_MODELS_MAP = {
    "NVIDIA STT Multilingual FastConformer Hybrid Transducer-CTC Large P&C": (
        "stt_multilingual_fastconformer_hybrid_large_pc",
        EncDecHybridRNNTCTCBPEModel,
    ),

    "NVIDIA STT Ru Conformer-CTC Large": (
        "nvidia/stt_ru_conformer_ctc_large",
        EncDecCTCModelBPE,
    ),

    "NVIDIA FastConformer-Hybrid Large (ru)": (
        "nvidia/stt_ru_fastconformer_hybrid_large_pc",
        EncDecHybridRNNTCTCBPEModel,
    ),

    "NVIDIA FastConformer-Hybrid Large (kk-ru)": (
        "nvidia/stt_kk_ru_fastconformer_hybrid_large",
        EncDecHybridRNNTCTCBPEModel,
    ),
}


class NvidiaModel(ASRModel):
    def __init__(self, model_name: str) -> None:
        assert model_name in NVIDIA_MODELS_MAP.keys(), f"Invalid model name: {model_name}. Use another class for the model!"

        self.name = model_name

        model = NVIDIA_MODELS_MAP[model_name][1].from_pretrained(NVIDIA_MODELS_MAP[model_name][0], map_location="cpu")

        if not isinstance(model, NeMoASRModel):
            raise ValueError(f"Unexpected model type: {type(model)}")

        self.__model = model


    def transcribe_file(self, file_path: str) -> str:
        result = self.__model.transcribe(file_path, verbose=False)

        if is_cuda_available():
            clear_cuda_cache()

        if isinstance(result, (tuple, list)):
            result = result[0]
        if isinstance(result, (tuple, list)):
            result = result[0]

        transcription = result

        assert isinstance(transcription, str), f"Unexpected transcription return type: expected str, got {type(transcription)}"

        return transcription


    def transcribe_wav(self, wav: Tensor | ndarray, sample_rate: int = 16_000) -> str:
        wav = prepare_audio(wav, sample_rate)

        result = self.__model.transcribe(wav, verbose=False)

        del wav

        if is_cuda_available():
            clear_cuda_cache()

        if isinstance(result, (tuple, list)):
            result = result[0]
        if isinstance(result, (tuple, list)):
            result = result[0]

        transcription = result

        assert isinstance(transcription, str), f"Unexpected transcription return type: expected str, got {type(transcription)}"

        return transcription


    def move_to_device(self, new_device: str | device) -> None:
        if not isinstance(new_device, device):
            new_device = device(new_device)

        if self.__model.device != new_device:
            self.__model.to(new_device)


    def get_current_device(self) -> str:
        return self.__model.device.type