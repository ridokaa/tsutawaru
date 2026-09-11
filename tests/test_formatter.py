"""Tests for transcript multi-tier formatter (§5a)."""
from __future__ import annotations

from tsutawaru.config import UiCfg
from tsutawaru.models import Segment, Token
from tsutawaru.ui.formatter import render_block


def test_render_block_golden_spec():
    cfg = UiCfg()
    seg = Segment.new(
        stream="Discord",
        original="おはようございます、いい天気ですね。",
        romaji="ohayougozaimasu ii tenki desune",
        english="Good morning, nice weather today",
    )
    seg.tokens = [
        Token(surface="おはようございます", base_form="おはよう", romaji="ohayougozaimasu", gloss="good morning"),
        Token(surface="いい", base_form="いい", romaji="ii", gloss="good"),
        Token(surface="天気", base_form="天気", romaji="tenki", gloss="weather"),
        Token(surface="ですね", base_form="です", romaji="desune", gloss="right / isn't it"),
    ]

    rendered = render_block(seg, cfg)
    expected = "\n".join([
        "[Discord]:",
        "1. Original (JP)  : おはようございます、いい天気ですね。",
        "2. Romaji         : ohayougozaimasu ii tenki desune",
        "3. English (Full) : Good morning, nice weather today",
        "4. Word Breakdown :",
        "   • おはようございます (ohayougozaimasu) : good morning",
        "   • いい (ii) : good",
        "   • 天気 (tenki) : weather",
        "   • ですね (desune) : right / isn't it",
        "-" * 50,
    ])
    assert rendered == expected


def test_render_block_pending_placeholders():
    cfg = UiCfg()
    seg = Segment.new(stream="main", original="こんにちは")
    seg.tokens = [Token(surface="こんにちは", base_form="こんにちは", romaji="", gloss=None)]

    rendered = render_block(seg, cfg)
    assert "2. Romaji         : …" in rendered
    assert "3. English (Full) : …" in rendered
    assert "• こんにちは () : …" in rendered


def test_render_block_handles_brackets_literally():
    cfg = UiCfg()
    seg = Segment.new(stream="main", original="[笑い]")
    rendered = render_block(seg, cfg)
    assert "1. Original (JP)  : [笑い]" in rendered
