from collections.abc import Iterable
from numpy import ndarray
from torch import Tensor, device, no_grad as torch_no_grad, argmax as torch_argmax
from torch.cuda import is_available as is_cuda_available, empty_cache as clear_cuda_cache
from torchaudio import load as load_audio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from .utils.constants import BASE_SAMPLE_RATE
from .utils.prepare_audio import prepare_audio
from .asr_model import ASRModel


MMS_MODELS_MAP = {
    "MMS-1B FL102": "facebook/mms-1b-fl102",
    "MMS-1B L1107": "facebook/mms-1b-l1107",
    "MMS-1B All": "facebook/mms-1b-all",
}


class MMSModel(ASRModel):
    def __init__(self, model_name: str) -> None:
        assert model_name in MMS_MODELS_MAP.keys(), f"Invalid model name: {model_name}. Use another class for the model!"

        self.name = model_name

        processor_or_tuple = Wav2Vec2Processor.from_pretrained(MMS_MODELS_MAP[model_name])

        if isinstance(processor_or_tuple, Wav2Vec2Processor):
            self.__processor = processor_or_tuple
        elif isinstance(processor_or_tuple, tuple):
            self.__processor = processor_or_tuple[0]
        else:
            raise ValueError(f"Unexpected processor type: {type(processor_or_tuple)}")

        self.__model = Wav2Vec2ForCTC.from_pretrained(MMS_MODELS_MAP[model_name]).to("cpu")

        self.__processor.tokenizer.set_target_lang("rus")
        self.__model.load_adapter("rus")


    def transcribe_file(self, file_path: str) -> str:
        audio, orig_freq = load_audio(file_path)

        return self.transcribe_wav(audio, orig_freq)


    def transcribe_wav(self, wav: Tensor | ndarray | Iterable, sample_rate: int = BASE_SAMPLE_RATE) -> str:
        wav = prepare_audio(wav, sample_rate)

        inputs = self.__processor(
            wav, sampling_rate=BASE_SAMPLE_RATE, return_tensors="pt"
        ).to(self.__model.device)

        with torch_no_grad():
            outputs = self.__model(**inputs).logits

        ids = torch_argmax(outputs, dim=-1)[0]
        transcription = self.__processor.decode(ids)

        del wav, inputs, outputs, ids

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