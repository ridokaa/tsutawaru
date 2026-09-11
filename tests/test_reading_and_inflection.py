"""The reading line must cover the whole utterance; the breakdown must not lie.

Two defects found by reading a live session:

  * the romaji line was built from the *filtered* token list, so fillers that
    were visible in the Japanese silently vanished from the reading, and an
    utterance of nothing but あー rendered a blank line;
  * agglutinated tokens gloss their dictionary head, so 行けなかった
    ("couldn't go") was glossed "i can go" — an inversion, not a nuance.
"""
from __future__ import annotations

import pytest

from tsutawaru.config import UiCfg, load
from tsutawaru.models import Segment, Token
from tsutawaru.nlp.agglutinate import agglutinate
from tsutawaru.nlp.pipeline import annotate
from tsutawaru.nlp.tokenizer import build_tokenizer


@pytest.fixture(scope="module")
def nlp():
    cfg = load().nlp
    return build_tokenizer(cfg), cfg


def _ann(nlp, text: str) -> Segment:
    tk, cfg = nlp
    return annotate(Segment.new(stream="main", original=text), tk, cfg)


# ---------------------------------------------------------------- the reading

@pytest.mark.parametrize("jp, must_contain", [
    ("つぼの上でなんかステップ踏んでるけどさぁ", "nanka"),   # フィラー
    ("えーっと、そうですね", "eetto"),                      # フィラー
])
def test_reading_keeps_words_the_breakdown_drops(nlp, jp, must_contain):
    seg = _ann(nlp, jp)
    assert must_contain in seg.romaji
    # ...and the word is still absent from the breakdown, which stays an edit
    assert must_contain not in " ".join(t.romaji for t in seg.tokens)


def test_an_all_filler_utterance_still_has_a_reading(nlp):
    """あー used to render a blank romaji line under visible Japanese."""
    seg = _ann(nlp, "あー")
    assert seg.romaji == "aa"
    assert seg.tokens == []


def test_reading_has_no_double_spaces_from_punctuation(nlp):
    """Punctuation reaches the reading list now, and romanises to nothing."""
    seg = _ann(nlp, "おはようございます、いい天気ですね。")
    assert seg.romaji == "ohayougozaimasu ii tenki desune"
    assert "  " not in seg.romaji


def test_truncation_does_not_shorten_the_reading(nlp):
    """max_tokens caps the word list; it must not cut the pronunciation."""
    jp = "猫 犬 鳥 馬 牛 豚 羊 鹿 熊 狼 狐 兎 鼠 鯨 鮫 蛸 烏 亀 蛇 蟹 蜂 蝶 蟻 蚊 鮭 鯉 鮪"
    seg = _ann(nlp, jp)
    assert seg.truncated > 0, "fixture must exceed max_tokens to be meaningful"
    assert len(seg.tokens) < len(seg.romaji.split())


def test_agglutination_still_fixes_readings(nlp):
    """The reading is built pre-filter but post-agglutination, which is what
    rejoins dangling geminate markers (もっ -> 'motsu' without it)."""
    assert "netsu" not in _ann(nlp, "ねっ").romaji


# ------------------------------------------------------------- the breakdown

@pytest.mark.parametrize("jp, base, labels", [
    ("行けなかった", "行ける", ["negative", "past"]),
    ("食べた", "食べる", ["past"]),
    ("行きません", "行く", ["polite", "negative"]),
    ("分からん", "分かる", ["negative"]),          # ん as 助動詞
    ("帰ろう", "帰る", ["volitional"]),
    ("食べたい", "食べる", ["want to"]),
    ("見てる", "見る", ["progressive"]),
    ("踏んでる", "踏む", ["progressive"]),          # でる, not いる
])
def test_conjugation_is_reported_next_to_the_gloss(nlp, jp, base, labels):
    seg = _ann(nlp, jp)
    tok = seg.tokens[0]
    assert tok.base_form == base       # gloss still hits the dictionary
    assert tok.infl == labels          # ...and the grammar is not lost


def test_negation_is_not_invented_for_the_nominaliser_ん(nlp):
    """ん is negative in 分からん but a nominaliser in 入れるのある.

    Labelling by surface alone would invert a correct reading, so the lookup is
    gated on part of speech.
    """
    for tok in _ann(nlp, "何か入れるのある").tokens:
        assert tok.infl == []


def test_stacked_auxiliaries_are_all_reported(nlp):
    tok = _ann(nlp, "食べさせられた").tokens[0]
    assert tok.infl == ["causative", "passive/potential", "past"]


def test_simple_tokens_carry_no_labels(nlp):
    for tok in _ann(nlp, "天気").tokens:
        assert tok.infl == []


def test_labels_are_deduped():
    """Two past markers in one group should not print 'past, past'."""
    toks = [
        Token(surface="食べ", base_form="食べる", pos="動詞", pos1="自立"),
        Token(surface="て", base_form="て", pos="助詞", pos1="接続助詞"),
        Token(surface="い", base_form="いる", pos="動詞", pos1="非自立"),
        Token(surface="た", base_form="た", pos="助動詞", pos1="*"),
    ]
    assert agglutinate(toks)[0].infl == ["progressive", "past"]


def test_each_token_gets_its_own_label_list():
    """A shared mutable default would leak labels between tokens."""
    toks = [
        Token(surface="食べ", base_form="食べる", pos="動詞", pos1="自立"),
        Token(surface="た", base_form="た", pos="助動詞", pos1="*"),
        Token(surface="犬", base_form="犬", pos="名詞", pos1="一般"),
    ]
    out = agglutinate(toks)
    assert out[0].infl == ["past"]
    assert out[1].infl == []


# ----------------------------------------------------------------- rendering

def test_labels_reach_the_window():
    from tsutawaru.ui.window_qt import _fmt_block

    seg = Segment.new(stream="main", original="行けなかった", english="I couldn't go")
    seg.tokens = [Token(surface="行けなかった", base_form="行ける", romaji="ikenakatta",
                        pos="動詞", pos1="自立", gloss="i can go",
                        infl=["negative", "past"])]
    # The breakdown is collapsed by default, so the labels live behind the
    # toggle. Both states matter: the header must not leak the gloss, and
    # opening it must still carry the inflection labels through.
    collapsed = _fmt_block(seg, UiCfg())
    assert "negative, past" not in collapsed
    assert "tok:" in collapsed  # the toggle is there to open

    html = _fmt_block(seg, UiCfg(), expanded=True)
    assert "negative, past" in html


def test_labels_reach_the_console_and_log():
    from tsutawaru.ui.formatter import render_block

    seg = Segment.new(stream="main", original="行けなかった", english="I couldn't go")
    seg.tokens = [Token(surface="行けなかった", base_form="行ける", romaji="ikenakatta",
                        pos="動詞", pos1="自立", gloss="i can go",
                        infl=["negative", "past"])]
    assert "[negative, past]" in render_block(seg, UiCfg())


def test_labels_reach_the_websocket_payload():
    seg = Segment.new(stream="main", original="食べた")
    seg.tokens = [Token(surface="食べた", base_form="食べる", pos="動詞",
                        pos1="自立", infl=["past"])]
    assert seg.to_dict()["tokens"][0]["i"] == ["past"]
