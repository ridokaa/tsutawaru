"""Hepburn romanization (token-driven with particle overrides)."""
from __future__ import annotations

from functools import lru_cache
import re
import pykakasi

from tsutawaru.models import Token

_kks = pykakasi.kakasi()

# Hepburn renders these particles by pronunciation, not by kana.
PARTICLE_ROMAJI = {"は": "wa", "へ": "e", "を": "o"}
_PUNCT = re.compile(r"^[^\w]+$", re.UNICODE)


@lru_cache(maxsize=50000)
def to_romaji(text: str) -> str:
    """Hepburn romaji for an arbitrary Japanese string."""
    parts = []
    for p in _kks.convert(text):
        h = p["hepburn"].strip()
        if h and not _PUNCT.match(h):  # pykakasi passes punctuation through
            parts.append(h)
    return " ".join(" ".join(parts).split())


def token_romaji(tok: Token, mode: str = "hepburn") -> str:
    """Romanize ONE (possibly merged) token. Prefer the katakana reading."""
    if mode == "hepburn" and tok.pos == "助詞" and tok.surface in PARTICLE_ROMAJI:
        return PARTICLE_ROMAJI[tok.surface]
    src = tok.reading or tok.surface
    # A trailing っ/ッ with nothing after it geminates nothing. Agglutination
    # rejoins it to the next token where one exists, but utterance-final cases
    # (ねっ, あっ, — a clipped or emphatic ending) have no next token, and
    # pykakasi then spells the marker out: ねっ -> "netsu" rather than "ne".
    while len(src) > 1 and src[-1] in ("っ", "ッ"):
        src = src[:-1]
    return to_romaji(src)


def sentence_romaji(tokens: list[Token], mode: str = "hepburn") -> str:
    """Build the sentence romaji from the tokens — never from the raw string.

    Skips tokens that romanise to nothing. The reading is built from the
    *unfiltered* token list so it covers every spoken word, which means
    punctuation reaches this function too — 「、」 has no pronunciation, and
    joining it in blindly leaves a double space in the middle of the line.
    """
    parts = (t.romaji or token_romaji(t, mode) for t in tokens)
    return " ".join(p for p in parts if p).strip()
