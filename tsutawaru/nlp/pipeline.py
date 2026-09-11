"""NLP pipeline orchestrator for a Segment."""
from __future__ import annotations

import unicodedata

from tsutawaru.config import NlpCfg
from tsutawaru.models import Segment
from tsutawaru.nlp.agglutinate import agglutinate
from tsutawaru.nlp.filters import dedupe, keep
from tsutawaru.nlp.romaji import sentence_romaji, token_romaji


def _reading_tokens(text: str, tok_engine, cfg: NlpCfg) -> list:
    """Normalize, tokenize, agglutinate, romanize. The reading list, unfiltered.

    Shared by `annotate` and `reading` so the two sides of an A/B cannot differ
    by romanization (は -> wa) rather than by what the models heard.
    """
    normalized = unicodedata.normalize("NFKC", text)
    toks = tok_engine.tokenize(normalized)
    if cfg.agglutinate:
        # Before filtering: agglutination is what makes a reading correct
        # (it rejoins dangling geminate markers), and it must see the real
        # morpheme sequence, not one with holes punched in it.
        toks = agglutinate(toks)
    for t in toks:
        t.romaji = token_romaji(t, cfg.particle_romaji)
    return toks


def reading(text: str, tok_engine, cfg: NlpCfg) -> str:
    """Romaji line for a bare transcript — the comparison lane, which arrives
    after the NLP stage is done with the segment and so cannot use `annotate`.
    """
    return sentence_romaji(_reading_tokens(text, tok_engine, cfg), cfg.particle_romaji)


def annotate(seg: Segment, tok_engine, cfg: NlpCfg) -> Segment:
    """Run the NLP stage: normalize -> tokenize -> agglutinate -> romaji + breakdown.

    The reading and the breakdown are built from *different* token lists on
    purpose. The breakdown is an edited view — fillers and punctuation removed,
    stutters collapsed, capped at max_tokens — because a word list is only
    useful if it is short. The reading is not a view of anything; it is how the
    Japanese line above it is pronounced, so it has to cover every word.

    Sharing one list made the reading inherit the breakdown's edits: なんか
    vanished from `tsu bo no ue de … ` while still being visible in the
    Japanese, and an utterance of nothing but あー rendered a blank romaji line.
    """
    reading_toks = _reading_tokens(seg.original, tok_engine, cfg)
    seg.romaji = sentence_romaji(reading_toks, cfg.particle_romaji)

    # Breakdown: the edited view. Token objects are shared with the reading
    # list, so romaji is already populated on every survivor.
    toks = dedupe([t for t in reading_toks if keep(t)])

    truncated = 0
    if len(toks) > cfg.max_tokens:
        truncated = len(toks) - cfg.max_tokens
        toks = toks[: cfg.max_tokens]
    seg.truncated = truncated

    seg.tokens = toks
    return seg
