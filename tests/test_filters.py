"""Tests for punctuation, POS filtering, and deduplication."""
from __future__ import annotations

from tsutawaru.models import Token
from tsutawaru.nlp.filters import PUNCT_RE, dedupe, keep


def test_punct_regex_matches_standard_punctuation():
    assert PUNCT_RE.match("、") is not None
    assert PUNCT_RE.match("。") is not None
    assert PUNCT_RE.match("…") is not None
    assert PUNCT_RE.match("！？") is not None
    assert PUNCT_RE.match("「」") is not None


def test_choonpu_preserved_in_words():
    """ー inside words like コーヒー must NOT match punct regex alone."""
    assert PUNCT_RE.match("コーヒー") is None
    tok = Token(surface="コーヒー", base_form="コーヒー", pos="名詞")
    assert keep(tok) is True


def test_dedupe_collapses_immediate_stutters():
    toks = [
        Token(surface="あ", base_form="あ"),
        Token(surface="あ", base_form="あ"),
        Token(surface="あの", base_form="あの"),
    ]
    res = dedupe(toks)
    assert len(res) == 2
    assert [t.surface for t in res] == ["あ", "あの"]
