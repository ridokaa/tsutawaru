"""Tests for token-driven Hepburn romanization."""
from __future__ import annotations

from tsutawaru.config import NlpCfg
from tsutawaru.models import Segment
from tsutawaru.nlp.pipeline import annotate
from tsutawaru.nlp.tokenizer import JanomeTokenizer


def test_romaji_pairs():
    cfg = NlpCfg()
    tok_engine = JanomeTokenizer()

    pairs = [
        ("おはようございます、いい天気ですね。", "ohayougozaimasu ii tenki desune"),
        ("私は学生です", "watashi wa gakusei desu"),
        ("ちょっと待ってください", "chotto mattekudasai"),
        ("今日は日本語を勉強します", "kyou wa nihongo o benkyou shimasu"),
        ("コーヒーを飲みます", "koohii o nomimasu"),
        ("東京へ行きます", "toukyou e ikimasu"),
        ("昨日は友達と映画を見に行った", "kinou wa tomodachi to eiga o mi ni itta"),
    ]

    for jp, expected_romaji in pairs:
        seg = Segment.new(stream="test", original=jp)
        annotate(seg, tok_engine, cfg)
        assert seg.romaji == expected_romaji, f"Failed for {jp}: got '{seg.romaji}' expected '{expected_romaji}'"


def test_romaji_particle_pronunciation():
    """Hepburn overrides: は->wa, へ->e, を->o."""
    tok_engine = JanomeTokenizer()
    cfg = NlpCfg()

    seg = Segment.new(stream="test", original="学校へ行きます")
    annotate(seg, tok_engine, cfg)
    assert " e " in f" {seg.romaji} "
    assert " he " not in f" {seg.romaji} "
