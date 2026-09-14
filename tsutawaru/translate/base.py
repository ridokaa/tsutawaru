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

    #: `remember=False` asks a stateful backend not to keep this line as
    #: context for the next one. Only the compare lane passes it — that lane is
    #: a second transcript of audio the primary lane already sent, so keeping
    #: both would put the same utterance in the context twice. Stateless
    #: backends accept and ignore it.
    def sentence(self, text: str, remember: bool = True) -> str:
        ...

    def batch(self, texts: list[str], remember: bool = True) -> list[str]:
        ...
