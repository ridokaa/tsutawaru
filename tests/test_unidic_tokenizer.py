"""fugashi + UniDic, mapped onto the IPADIC tagset the pipeline speaks.

These test the *mapping*, not UniDic. UniDic's own analysis is not this
project's to verify; what is, is that its output arrives downstream shaped the
way agglutinate.py, drop_pos and romaji.py expect — because the whole point of
the backend is to be swappable, and a swap that changes two things at once
cannot be measured.
"""
from __future__ import annotations

import pytest

from tsutawaru.config import NlpCfg
from tsutawaru.nlp.tokenizer import (
    JanomeTokenizer,
    _unidic_lemma,
    _unidic_pos,
    build_tokenizer,
    resolve_unidic_dir,
)

pytestmark = pytest.mark.skipif(
    resolve_unidic_dir() is None,
    reason="no UniDic dictionary installed (see tools/tokdiff.py docstring)",
)


@pytest.fixture(scope="module")
def uni():
    pytest.importorskip("fugashi")
    return build_tokenizer(NlpCfg(tokenizer="unidic"))


def test_janome_is_still_the_default():
    """The candidate must never become the control by accident."""
    assert NlpCfg().tokenizer == "janome"
    assert isinstance(build_tokenizer(NlpCfg()), JanomeTokenizer)


# --- POS mapping ----------------------------------------------------------

def test_hijiritsu_kanou_does_not_become_hijiritsu():
    """非自立可能 means "may be auxiliary", 非自立 asserts that it is.

    Collapsing them would make agglutinate.py fold an independent verb into
    whatever preceded it: 行けなかった tags 行け as 動詞-非自立可能, and treating
    that as bound merges it backwards into the previous word.
    """
    assert _unidic_pos("動詞", "非自立可能") == ("動詞", "自立")
    assert _unidic_pos("形容詞", "非自立可能") == ("形容詞", "自立")


@pytest.mark.parametrize("unidic,ipadic", [
    (("名詞", "普通名詞"), ("名詞", "一般")),
    (("名詞", "数詞"), ("名詞", "数")),
    (("代名詞", "*"), ("名詞", "代名詞")),
    (("形状詞", "一般"), ("名詞", "形容動詞語幹")),
    (("接尾辞", "名詞的"), ("名詞", "接尾")),
    (("接尾辞", "動詞的"), ("動詞", "接尾")),
    (("感動詞", "フィラー"), ("フィラー", "*")),
])
def test_pos_pairs_map_onto_the_ipadic_tagset(unidic, ipadic):
    assert _unidic_pos(*unidic) == ipadic


def test_particle_subtypes_pass_through_untouched():
    """Agglutination keys on these two by name; renaming them breaks the merge."""
    assert _unidic_pos("助詞", "終助詞") == ("助詞", "終助詞")
    assert _unidic_pos("助詞", "接続助詞") == ("助詞", "接続助詞")


def test_absent_subtype_matches_janome_not_empty_string():
    """Janome passes IPADIC's literal "*" through. Differing here would report a
    difference on every auxiliary in the corpus and drown the real ones."""
    assert _unidic_pos("助動詞", "*")[1] == "*"
    assert _unidic_pos("連体詞", "")[1] == "*"


# --- lemma ----------------------------------------------------------------

@pytest.mark.parametrize("lemma,surface,want", [
    ("コーヒー-coffee", "コーヒー", "コーヒー"),  # loanword origin annotation
    ("行く", "行け", "行く"),
    ("*", "ほげ", "ほげ"),                        # unknown word
    ("", "ほげ", "ほげ"),
])
def test_lemma_annotation_is_stripped(lemma, surface, want):
    assert _unidic_lemma(lemma, surface) == want


# --- end to end -----------------------------------------------------------

def test_emits_ipadic_shaped_tokens(uni):
    toks = uni.tokenize("今日はいい天気ですね")
    assert [t.surface for t in toks] == ["今日", "は", "いい", "天気", "です", "ね"]
    by_surface = {t.surface: t for t in toks}
    assert by_surface["は"].pos == "助詞"
    assert by_surface["ね"].pos1 == "終助詞"      # agglutination depends on this
    assert by_surface["です"].pos == "助動詞"
    assert by_surface["天気"].reading == "テンキ"  # katakana, as IPADIC's 読み is


def test_reading_is_the_written_form_not_the_pronunciation(uni):
    """kana (仮名形), not pron (発音形).

    pron already applies は->ワ, and romaji.py applies its own Hepburn particle
    override on top — taking pron would mean the correction happens twice.
    """
    toks = {t.surface: t for t in uni.tokenize("今日はいい天気ですね")}
    assert toks["は"].reading == "ハ"


def test_no_silent_fallback_to_janome():
    """A missing dictionary must fail loudly.

    An experiment that quietly runs the control instead of the candidate is
    worse than one that crashes: the numbers look plausible and mean nothing.
    """
    with pytest.raises(RuntimeError, match="no UniDic found"):
        build_tokenizer(NlpCfg(tokenizer="unidic", unidic_dir="/nonexistent/unidic"))
