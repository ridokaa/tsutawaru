"""Translation backend protocol."""
from __future__ import annotations

from typing import Protocol


class Translator(Protocol):
    #: Stable identifier for the backend actually doing the work ("google",
    #: "deepl"). The gloss cache is keyed on this so that results from
    #: different backends can never collide in the persistent DB. It reports
    #: what the backend *is*, which is not always what config asked for —
    #: provider="deepl" with no API key silently becomes Google.
    name: str

    def sentence(self, text: str) -> str:
        ...

    def batch(self, texts: list[str]) -> list[str]:
        ...
