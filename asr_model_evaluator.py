from datasets import Dataset, Audio
from evaluate import Metric, load as load_metric
from torch import device
from torch.cuda import is_available as is_cuda_available, empty_cache as clear_cuda_cache

from tqdm.notebook import tqdm
from huggingface_hub.utils.tqdm import disable_progress_bars as disable_hf_progress_bars, enable_progress_bars as enable_hf_progress_bars, are_progress_bars_disabled as are_hf_progress_bars_disabled
from datasets.utils.tqdm import disable_progress_bars as disable_datasets_progress_bars, enable_progress_bars as enable_datasets_progress_bars, are_progress_bars_disabled as are_datasets_progress_bars_disabled
from transformers.utils.logging import disable_progress_bar as disable_transformers_progress_bar, enable_progress_bar as enable_transformers_progress_bar

from models import construct_asr_model
from models.utils.constants import BASE_SAMPLE_RATE, SUPPORTED_METRICS, LONG_AUDIO_THRESHOLD_SECONDS
from models.utils.normalize_text import normalize_text

from models.asr_model import ASRModel
from models.utils.vad_model import VADModel


def disable_progress_bars():
    disable_hf_progress_bars()
    disable_datasets_progress_bars()
    disable_transformers_progress_bar()

def enable_progress_bars():
    enable_hf_progress_bars()
    enable_datasets_progress_bars()
    enable_transformers_progress_bar()

def are_progress_bars_disabled():
    return are_hf_progress_bars_disabled() or are_datasets_progress_bars_disabled()


class ASRModelEvaluator:
    def __init__(self, *_, metric: SUPPORTED_METRICS):
        self.__vad_model: VADModel | None = None
        self.__metric: Metric = load_metric(metric)


    def _get_vad_model(self, device: str | device = "cpu") -> VADModel:
        if self.__vad_model is None:
            self.__vad_model = VADModel()

        self.__vad_model.move_to_device(device)

        return self.__vad_model


    def _prepare_dataset(self, data: Dataset, use_device: str | device = "cpu") -> Dataset:
        def split_audio_to_segments(batch: dict[str, list]) -> dict[str, list]:
            result = {
                "audio_segments": [],
                "audio_sample_rate": [],
                "transcription": batch["transcription"],
            }

            for audio in batch["audio"]:
                audio_array = audio["array"]
                audio_sample_rate = audio["sampling_rate"]

                if len(audio_array) < LONG_AUDIO_THRESHOLD_SECONDS * audio_sample_rate:
                    result["audio_segments"].append([audio_array])
                else:
                    result["audio_segments"].append(self._get_vad_model(use_device).split_wav_tensor(audio_array, audio_sample_rate, use_device=use_device))

                del audio_array

                result["audio_sample_rate"].append(audio_sample_rate)

                if is_cuda_available():
                    clear_cuda_cache()

            return result

        return data.cast_column(
            column="audio",
            feature=Audio(sampling_rate=BASE_SAMPLE_RATE),
        ).map(
            function=split_audio_to_segments,
            remove_columns=data.column_names,
            batched=True,
            batch_size=16,
            desc="Splitting each audio in the dataset to speech segments",
        )


    def evaluate_model(
            self,
            model: ASRModel | str,
            data: Dataset,
            use_text_normalization: bool = True,
            use_device: str | device = "cpu",
            verbose: bool | None = None,
        ) -> float:

        pr_bars_enabled = not are_progress_bars_disabled()

        if verbose and not pr_bars_enabled:
            enable_progress_bars()
        elif pr_bars_enabled:
            disable_progress_bars()

        if "audio_segments" not in data.column_names:
            data = self._prepare_dataset(data, use_device=use_device)
            self._get_vad_model("cpu")

        if isinstance(model, str):
            model = construct_asr_model(model)

        model_device = model.get_current_device()

        model.move_to_device(use_device)

        def _process_sample(sample: dict) -> dict:
            audio_segments = sample["audio_segments"]
            audio_sample_rate = sample["audio_sample_rate"]
            reference_transcription = sample["transcription"]

            predicted_transcription = ""

            for segment in tqdm(audio_segments, disable=not verbose or len(audio_segments) == 1, desc="Audio progress"):
                predicted_transcription += model.transcribe_wav(segment, audio_sample_rate) + " "

            return {
                "reference": reference_transcription,
                "predicted": predicted_transcription,
            }
        
        processed = data.map(
            function=_process_sample,
            remove_columns=data.column_names,
            new_fingerprint=model.name[:64],
            desc="Dataset progress",
        )

        model.move_to_device(model_device)

        if is_cuda_available():
            clear_cuda_cache()
        
        def normalize_sample(sample: dict) -> dict:
            return {
                "reference": normalize_text(sample["reference"]),
                "predicted": normalize_text(sample["predicted"]),
            }
        
        if use_text_normalization:
            processed = processed.map(
                function=normalize_sample,
                remove_columns=processed.column_names,
                desc="Normalizing transcriptions",
            )
        
        result = self.__metric.compute(predictions=processed["predicted"], references=processed["reference"])

        assert isinstance(result, float), f"Metric computation result is not a float! Used metric is {self.__metric.name}, got result {result} of type {type(result)}"

        if pr_bars_enabled and are_progress_bars_disabled():
            enable_progress_bars()
        elif not pr_bars_enabled and not are_progress_bars_disabled():
            disable_progress_bars()

        return result


    def evaluate(
            self,
            models: ASRModel | str | list[ASRModel] | list[str],
            data: Dataset,
            use_text_normalization: bool = True,
            use_device: str | device = "cpu",
            verbose: bool | None = None,
        ) -> list[float]:

        pr_bars_enabled = not are_progress_bars_disabled()

        if verbose and not pr_bars_enabled:
            enable_progress_bars()
        elif pr_bars_enabled:
            disable_progress_bars()

        if "audio_segments" not in data.column_names:
            data = self._prepare_dataset(data, use_device=use_device)
            self._get_vad_model("cpu")

        if not isinstance(models, list):
            models = [models]

        result = []

        for model in models:
            if verbose != False:
                print(f"Evaluating {model.name if isinstance(model, ASRModel) else model}")

            value = self.evaluate_model(
                model, data, use_text_normalization, use_device, verbose
            )

            if verbose != False:
                print(f"{self.__metric.name} = {value}")

            result.append(value)

        if pr_bars_enabled and are_progress_bars_disabled():
            enable_progress_bars()
        elif not pr_bars_enabled and not are_progress_bars_disabled():
            disable_progress_bars()

        return result