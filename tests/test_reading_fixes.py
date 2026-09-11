"""IPADIC readings that are right for the tag and wrong for conversation.

Found by reading a live session: 「この下とか潜る」 rendered as
"kono **moto** toka moguru". The romaji tier is one of the reasons this tool
exists, so a wrong reading is not cosmetic — it teaches the wrong word.

The trigger is a Viterbi artifact rather than a rule about any one neighbour:
下 reads シタ standing alone and in 「この下」, but flips to the bound-noun entry
下/名詞-非自立/モト once a determiner precedes it *and* a 並立助詞 follows. 後 is
the worse case, flipping to ノチ after any determiner at all, which makes the
everyday 「この後」 (kono ato) read as the literary kono nochi.
"""
from __future__ import annotations

import pytest

from tsutawaru.config import load
from tsutawaru.models import Segment, Token
from tsutawaru.nlp.pipeline import annotate
from tsutawaru.nlp.tokenizer import READING_FIXES, build_tokenizer, fix_reading


@pytest.fixture(scope="module")
def nlp():
    cfg = load().nlp
    return build_tokenizer(cfg), cfg


def _romaji(nlp, text: str) -> str:
    tk, cfg = nlp
    return annotate(Segment.new(stream="main", original=text), tk, cfg).romaji


@pytest.mark.parametrize("jp, want, must_not", [
    ("この下とか潜る", "shita", "moto"),
    ("海の下とか", "shita", "moto"),
    ("あの下とか見る", "shita", "moto"),
    ("この下や", "shita", "moto"),
    ("この後とか行く", "ato", "nochi"),
    ("この後どうする", "ato", "nochi"),
    ("ご飯の後とか", "ato", "nochi"),
])
def test_conversational_reading_wins(nlp, jp, want, must_not):
    r = _romaji(nlp, jp)
    assert want in r, f"{jp} -> {r}"
    assert must_not not in r, f"{jp} -> {r}"


@pytest.mark.parametrize("jp, want", [
    ("この下に潜る", "shita"),   # already correct — must stay correct
    ("下とか", "shita"),
    ("机の下", "shita"),
    ("その下", "shita"),
])
def test_already_correct_readings_are_untouched(nlp, jp, want):
    assert want in _romaji(nlp, jp)


def test_the_table_is_keyed_on_the_tag_not_the_surface():
    """A surface-only rule would rewrite every 下, including the correct ones.

    The bound-noun tag is not itself the defect — 上・中・事・物・所 all take
    非自立 with the right reading — so the key has to carry the tag that goes
    wrong, and a token wearing any other tag must pass through untouched.
    """
    assert all(len(k) == 3 for k in READING_FIXES)

    bound = fix_reading(Token(surface="下", base_form="下", reading="モト",
                              pos="名詞", pos1="非自立"))
    assert bound.reading == "シタ"

    ordinary = fix_reading(Token(surface="下", base_form="下", reading="シタ",
                                 pos="名詞", pos1="一般"))
    assert ordinary.reading == "シタ"

    # A different word carrying the same tag is none of this table's business.
    other = fix_reading(Token(surface="中", base_form="中", reading="ナカ",
                              pos="名詞", pos1="非自立"))
    assert other.reading == "ナカ"


def test_the_fix_lands_before_agglutination(nlp):
    """Agglutination concatenates readings, so a late fix would arrive too late.

    「この後とかさ」 merges the trailing 終助詞; if the correction ran after that
    merge it would have to unpick ノチ out of an already-joined reading.
    """
    r = _romaji(nlp, "この後とかさ")
    assert "ato" in r and "nochi" not in r
