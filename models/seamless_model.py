from .asr_model import ASRModel

from transformers import SeamlessM4TProcessor, SeamlessM4TModel, SeamlessM4Tv2Model
from numpy import ndarray
from torch import Tensor, device
from torch.cuda import is_available as is_cuda_available, empty_cache as clear_cuda_cache
from torchaudio import load as load_audio
from .utils.prepare_audio import prepare_audio


SEAMLESS_MODELS_MAP = {
    "SeamlessM4T Medium": "facebook/hf-seamless-m4t-medium",
    "SeamlessM4T Large-v1": "facebook/hf-seamless-m4t-large",
    "SeamlessM4T Large-v2": "facebook/seamless-m4t-v2-large",
}


class SeamlessM4T(ASRModel):
    def __init__(self, model_name: str) -> None:
        assert model_name in SEAMLESS_MODELS_MAP.keys(), f"Invalid model name: {model_name}. Use another class for the model!"

        self.name = model_name

        processor_or_tuple = SeamlessM4TProcessor.from_pretrained(SEAMLESS_MODELS_MAP[model_name])

        if isinstance(processor_or_tuple, SeamlessM4TProcessor):
            self.__processor = processor_or_tuple
        elif isinstance(processor_or_tuple, tuple):
            self.__processor = processor_or_tuple[0]
        else:
            raise ValueError(f"Unexpected processor type: {type(processor_or_tuple)}")

        if "v2" in model_name:
            self.__model = SeamlessM4Tv2Model.from_pretrained(SEAMLESS_MODELS_MAP[model_name]).to("cpu")
        else:
            self.__model = SeamlessM4TModel.from_pretrained(SEAMLESS_MODELS_MAP[model_name]).to("cpu")


    def transcribe_file(self, file_path: str) -> str:
        audio, orig_freq = load_audio(file_path)

        return self.transcribe_wav(audio, orig_freq)


    def transcribe_wav(self, wav: Tensor | ndarray, sample_rate: int = 16_000) -> str:
        wav = prepare_audio(wav, sample_rate)

        audio_inputs = self.__processor(audios=wav, sampling_rate=16_000, return_tensors="pt")
        audio_inputs = audio_inputs.to(self.__model.device)

        output_tokens = self.__model.generate(**audio_inputs, tgt_lang="rus", generate_speech=False)

        transcription = self.__processor.decode(output_tokens[0].tolist()[0], skip_special_tokens=True)

        del wav, audio_inputs, output_tokens

        if is_cuda_available():
            clear_cuda_cache()

        assert isinstance(transcription, str), f"Unexpected transcription return type: expected str, got {type(transcription)}"

        return transcription


    def move_to_device(self, new_device: str | device) -> None:
        if not isinstance(new_device, device):
            new_device = device(new_device)

        if self.__model.device != new_device:
            self.__model.to(new_device)


    def get_current_device(self) -> str:
        return self.__model.device.type