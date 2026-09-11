"""Tests for static gloss dictionary."""
from __future__ import annotations

from tsutawaru.models import Token
from tsutawaru.translate.static_gloss import static_gloss


def test_static_gloss_merged_surface_beats_base_form():
    # ですね (surface) vs です (base_form)
    tok = Token(surface="ですね", base_form="です", pos="助動詞")
    gloss = static_gloss(tok)
    assert gloss == "right / isn't it"


def test_static_gloss_particles():
    assert static_gloss(Token(surface="は", base_form="は", pos="助詞")) == "topic marker"
    assert static_gloss(Token(surface="が", base_form="が", pos="助詞")) == "subject marker"
    assert static_gloss(Token(surface="を", base_form="を", pos="助詞")) == "object marker"


def test_static_gloss_unmapped_returns_none():
    tok = Token(surface="量子力学", base_form="量子力学", pos="名詞")
    assert static_gloss(tok) is None
