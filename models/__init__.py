from warnings import filterwarnings
filterwarnings("ignore", category=UserWarning)
filterwarnings("ignore", category=FutureWarning)
# Hide warnings from used libraries

from .asr_model import ASRModel
from .seamless_model import SeamlessM4T, SEAMLESS_MODELS_MAP
from .mms_model import MMSModel, MMS_MODELS_MAP
from .whisper_model import WhisperModel, WHISPER_MODELS_MAP
from .nvidia_model import NvidiaModel, NVIDIA_MODELS_MAP
from .gigaam_model import GigaAM, GIGAAM_MODELS_MAP


ALL_ASR_MODELS = [
    *list(SEAMLESS_MODELS_MAP.keys()),
    *list(MMS_MODELS_MAP.keys()),
    *list(WHISPER_MODELS_MAP.keys()),
    *list(NVIDIA_MODELS_MAP.keys()),
    *list(GIGAAM_MODELS_MAP.keys()),
]


def get_supported_models() -> list[str]:
    return ALL_ASR_MODELS


def construct_asr_model(model_name: str) -> ASRModel:
    if model_name in SEAMLESS_MODELS_MAP:
        return SeamlessM4T(model_name)

    if model_name in MMS_MODELS_MAP:
        return MMSModel(model_name)

    if model_name in WHISPER_MODELS_MAP:
        return WhisperModel(model_name)

    if model_name in NVIDIA_MODELS_MAP:
        return NvidiaModel(model_name)

    if model_name in GIGAAM_MODELS_MAP:
        return GigaAM(model_name)

    raise ValueError(
        f"Invalid model name: {model_name}.\n"
        "Choose another model! You can use `get_supported_models()` method to get the list of supported models."
    )


__all__ = [
    "ASRModel",
    "SeamlessM4T",
    "MMSModel",
    "WhisperModel",
    "NvidiaModel",
    "GigaAM",
    "get_supported_models",
    "construct_asr_model",
]