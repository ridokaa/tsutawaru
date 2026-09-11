"""Core data types. Every pipeline stage speaks in these.

This module is the contract between stages and deliberately has no dependencies
beyond numpy — import it from anywhere without pulling in audio or ML stacks.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

SAMPLE_RATE = 16000

_ids = itertools.count(1)


@dataclass(slots=True)
class Utterance:
    """A VAD-delimited chunk of speech, ready for STT."""

    id: int
    audio: np.ndarray  # float32, mono, 16 kHz, range [-1, 1]
    t_start: float  # monotonic seconds
    t_end: float
    stream: str = "main"
    forced: bool = False  # True when emitted by the max-length flush

    @property
    def duration(self) -> float:
        return len(self.audio) / SAMPLE_RATE

    @staticmethod
    def new(**kw) -> "Utterance":
        return Utterance(id=next(_ids), **kw)


@dataclass(slots=True)
class Token:
    """One breakdown entry.

    After agglutination a Token may span several IPADIC morphemes: `surface` and
    `reading` are the concatenation, while `base_form` stays the *head*
    morpheme's dictionary form so gloss lookups still hit (食べ+た -> 食べる).
    """

    surface: str
    base_form: str
    reading: str = ""  # katakana
    romaji: str = ""
    pos: str = ""  # major POS, e.g. 名詞
    pos1: str = ""  # POS subtype, e.g. 終助詞 — required by agglutination
    gloss: Optional[str] = None  # None = pending, "" = unavailable
    # Grammar carried by the morphemes agglutination folded in, e.g.
    # ["negative", "past"] for 行けなかった. The gloss is looked up on base_form
    # (行ける -> "can go"), so without this the breakdown states the opposite of
    # what was said. Populated by agglutinate(), empty for simple tokens.
    infl: list[str] = field(default_factory=list)


@dataclass(slots=True)
class Segment:
    """A transcript line at any stage of completion. Mutated in place by workers."""

    id: int
    stream: str
    original: str
    romaji: str = ""
    english: str = ""
    tokens: list[Token] = field(default_factory=list)
    lang: str = "ja"
    confidence: float = 0.0  # Whisper avg_logprob; 0.0 means "not reported"
    # Whisper no_speech_prob. Unlike avg_logprob, 0.0 is a perfectly ordinary
    # real value here — clean speech scores near zero — so it cannot double as
    # the "unset" marker. -1.0 is impossible for a probability and is used
    # instead; without a distinct sentinel an engine that reports nothing would
    # read as maximally certain that speech is present.
    no_speech: float = -1.0
    truncated: int = 0  # tokens dropped by max_tokens; rendered as "… (+N tokens)"
    continued: bool = False  # follows a forced flush — same speaker turn
    t_audio_end: float = 0.0
    t_stt_done: float = 0.0
    t_complete: float = 0.0
    partial: bool = True
    # A second ASR model's reading of the *same* audio. Empty `alt_model` means
    # comparison is off. On Segment rather than in a parallel structure because
    # the two transcripts must stay paired — anything that can drift out of step
    # loses the property being tested. `model` names the engine behind
    # `original` and is set only while comparing, so an ordinary session's JSONL
    # is unchanged and the window reads its emptiness to pick the layout.
    model: str = ""
    alt_model: str = ""
    alt_original: str = ""
    alt_romaji: str = ""
    alt_english: str = ""

    @staticmethod
    def new(**kw) -> "Segment":
        return Segment(id=next(_ids), **kw)

    def timing(self) -> dict:
        """How long the line took to arrive, measured from the *end* of speech.

        The budget is what a listener waits after someone stops talking, so
        measuring from the start would penalise a long sentence for being long.
        `line_ms` ends at `t_complete`, which the gloss lane stamps once the
        breakdown is filled — time-to-full-breakdown, not time-to-English.
        With the breakdown tier switched off the gloss lane never runs and the
        sentence lane stamps it instead, so `line_ms` then means
        time-to-English. The two are not comparable: a session recorded with
        the tier off will show a lower `line_ms` for the same pipeline, because
        it is measuring a shorter thing. Check which mode a session ran in
        before putting its figures beside another's.
        0.0 is the unset marker, so a stage never reached reports None.
        """

        def ms(a: float, b: float) -> Optional[float]:
            return round((b - a) * 1000.0, 1) if a and b else None

        return {
            "stt_ms": ms(self.t_audio_end, self.t_stt_done),
            "line_ms": ms(self.t_audio_end, self.t_complete),
        }

    def to_dict(self) -> dict:
        """JSON-safe projection used by --record and the WebSocket sink."""
        return {
            "id": self.id,
            "stream": self.stream,
            "jp": self.original,
            "romaji": self.romaji,
            "en": self.english,
            "lang": self.lang,
            "partial": self.partial,
            "continued": self.continued,
            "truncated": self.truncated,
            # Whisper's avg_logprob. Carried on Segment since the STT stage was
            # built, but until now it stopped there — no sink read it and it was
            # absent from this projection, so every --record session and every
            # WebSocket client saw garbled and clean lines as indistinguishable.
            "confidence": self.confidence,
            "no_speech": self.no_speech,
            # The latency budget, per line. The stamps were already on Segment;
            # only this projection was missing them, so no recorded session
            # could be checked against the budget after the fact.
            "t": self.timing(),
            **(
                # Only present when a comparison actually ran; its absence is
                # the honest signal that nothing was compared.
                {"model": self.model,
                 "alt": {"model": self.alt_model,
                         "jp": self.alt_original,
                         "romaji": self.alt_romaji,
                         "en": self.alt_english}}
                if self.alt_model else {}
            ),
            "tokens": [
                {"s": t.surface, "r": t.romaji, "g": t.gloss, "p": t.pos,
                 "i": t.infl}
                for t in self.tokens
            ],
        }
