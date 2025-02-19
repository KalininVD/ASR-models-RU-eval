from abc import ABC, abstractmethod
from collections.abc import Iterable
from torch import Tensor, device
from numpy import ndarray


class ASRModel(ABC):
    name: str

    @abstractmethod
    def transcribe_file(self, file_path: str) -> str:
        raise NotImplementedError("transcribe_file method called from abstract class!")

    @abstractmethod
    def transcribe_wav(self, wav: Tensor | ndarray | Iterable, sample_rate: int) -> str:
        raise NotImplementedError("transcribe_wav method called from abstract class!")

    @abstractmethod
    def move_to_device(self, device: str | device) -> None:
        raise NotImplementedError("move_to_device method called from abstract class!")
    
    @abstractmethod
    def get_current_device(self) -> str:
        raise NotImplementedError("get_current_device method called from abstract class!")