"""Base interfaces and protocols for Speech-to-Text (STT) engines."""
from __future__ import annotations

from typing import Protocol
import numpy as np


class STTResult:
    __slots__ = ("text", "language", "lang_prob", "avg_logprob", "no_speech_prob")

    def __init__(
        self,
        text: str,
        language: str,
        lang_prob: float,
        avg_logprob: float,
        no_speech_prob: float,
    ):
        self.text = text
        self.language = language
        self.lang_prob = lang_prob
        self.avg_logprob = avg_logprob
        self.no_speech_prob = no_speech_prob


class STTEngine(Protocol):
    def transcribe(self, audio: np.ndarray) -> STTResult:
        ...

    def warmup(self) -> None:
        ...
