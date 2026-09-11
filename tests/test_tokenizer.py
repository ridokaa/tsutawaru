"""Tests for Janome tokenizer and pos1 subtype extraction."""
from __future__ import annotations

from tsutawaru.nlp.tokenizer import JanomeTokenizer


def test_tokenizer_preserves_pos1():
    tok = JanomeTokenizer()
    tokens = tok.tokenize("おはようございます、いい天気ですね。")
    # All tokens should have pos and pos1 populated
    assert len(tokens) > 0
    for t in tokens:
        assert t.pos != ""

    # Check pos1 on sentence-final particle "ね"
    ne = [t for t in tokens if t.surface == "ね"]
    assert len(ne) == 1
    assert ne[0].pos == "助詞"
    assert ne[0].pos1 == "終助詞"


def test_tokenizer_raw_count_showcase():
    tok = JanomeTokenizer()
    tokens = tok.tokenize("おはようございます、いい天気ですね。")
    # Raw Janome segmentation produces 9 morphemes before filtering/agglutinating:
    # おはよう (感動詞) / ござい (助動詞) / ます (助動詞) / 、 (記号) / いい (形容詞) / 天気 (名詞) / です (助動詞) / ね (助詞) / 。 (記号)
    surfaces = [t.surface for t in tokens]
    assert surfaces == ["おはよう", "ござい", "ます", "、", "いい", "天気", "です", "ね", "。"]
