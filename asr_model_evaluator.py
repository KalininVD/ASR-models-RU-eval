from datasets import Dataset, IterableDataset, Audio
from evaluate import Metric, load as load_metric
from tqdm.notebook import tqdm
from torch import device
from torch.cuda import is_available as is_cuda_available, empty_cache as clear_cuda_cache

from models import construct_asr_model
from models.utils.constants import BASE_SAMPLE_RATE, SUPPORTED_METRICS, LONG_AUDIO_THRESHOLD_SECONDS
from models.utils.normalize_text import normalize_text

from models.asr_model import ASRModel
from models.utils.vad_model import VADModel


class ASRModelEvaluator:
    def __init__(self):
        self.__vad_model: VADModel | None = None


    def _get_vad_model(self, device: str | device = "cpu") -> VADModel:
        if self.__vad_model is None:
            self.__vad_model = VADModel()

        self.__vad_model.move_to_device(device)

        return self.__vad_model


    def _prepare_dataset(self, data: Dataset | IterableDataset, use_device: str | device = "cpu") -> Dataset | IterableDataset:
        data = data.cast_column("audio", Audio(sampling_rate=BASE_SAMPLE_RATE))

        def split_audio_to_segments(sample: dict) -> dict:
            audio = sample["audio"]
            audio_array = audio["array"]
            audio_sample_rate = audio["sampling_rate"]

            if len(audio_array) < LONG_AUDIO_THRESHOLD_SECONDS * audio_sample_rate:
                segments = [audio_array]
            else:
                segments = self._get_vad_model(use_device).split_wav_tensor(audio_array, audio_sample_rate, use_device=use_device)

            del audio_array

            if is_cuda_available():
                clear_cuda_cache()

            return {
                **sample,
                "audio_segments": segments,
            }

        return data.map(
            function=split_audio_to_segments,
        )


    def evaluate_model(
            self,
            metric: SUPPORTED_METRICS,
            model: ASRModel | str,
            data: Dataset | IterableDataset,
            use_text_normalization: bool = True,
            use_device: str | device = "cpu",
            verbose: bool = False,
        ) -> float:

        loaded_metric: Metric = load_metric(metric)

        if "audio_segments" not in data.column_names:
            data = self._prepare_dataset(data, use_device=use_device)
            self._get_vad_model("cpu")

        if isinstance(model, str):
            model = construct_asr_model(model)

        predicted_transcriptions = []
        reference_transcriptions = []

        model_device = model.get_current_device()

        model.move_to_device(use_device)

        for sample in tqdm(data, disable=not verbose, desc="Dataset progress"):
            audio_segments = sample["audio_segments"]
            audio_sample_rate = sample["audio"]["sampling_rate"]
            reference_transcription = sample["transcription"]

            predicted_transcription = ""

            for segment in tqdm(audio_segments, disable=not verbose or len(audio_segments) == 1, desc="Audio progress"):
                predicted_transcription += model.transcribe_wav(segment, audio_sample_rate) + " "

            predicted_transcriptions.append(predicted_transcription)
            reference_transcriptions.append(reference_transcription)

        model.move_to_device(model_device)

        if is_cuda_available():
            clear_cuda_cache()

        if use_text_normalization:
            predicted_transcriptions = list(map(normalize_text, predicted_transcriptions))
            reference_transcriptions = list(map(normalize_text, reference_transcriptions))

        result = loaded_metric.compute(predictions=predicted_transcriptions, references=reference_transcriptions)

        assert isinstance(result, float), f"Metric computation result is not a float! Used metric is {metric}, got result {result} of type {type(result)}"

        return result


    def evaluate(
            self,
            metric: SUPPORTED_METRICS,
            models: ASRModel | str | list[ASRModel] | list[str],
            data: Dataset | IterableDataset,
            use_text_normalization: bool = True,
            use_device: str | device = "cpu",
            verbose: bool = False,
        ) -> list[float]:

        if "audio_segments" not in data.column_names:
            data = self._prepare_dataset(data, use_device=use_device)
            self._get_vad_model("cpu")

        if not isinstance(models, list):
            models = [models]

        result = []

        for model in models:
            if verbose:
                print(f"Evaluating {model.name if isinstance(model, ASRModel) else model}")

            value = self.evaluate_model(
                metric, model, data, use_text_normalization, use_device, verbose
            )

            if verbose:
                print(f"{metric} = {value}")

            result.append(value)

        return result