from typing import Literal


SUPPORTED_METRICS = Literal[
    "wer",
    "cer",
]

LONG_AUDIO_THRESHOLD_SECONDS: int = 25

BASE_SAMPLE_RATE: int = 16_000