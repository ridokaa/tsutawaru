"""Tests for bound morpheme agglutination pass."""
from __future__ import annotations

import pytest

from tsutawaru.config import NlpCfg
from tsutawaru.nlp.agglutinate import agglutinate
from tsutawaru.nlp.filters import keep
from tsutawaru.nlp.tokenizer import (
    JanomeTokenizer,
    build_tokenizer,
    resolve_unidic_dir,
)


def test_agglutinate_showcase_4_tokens():
    """Showcase sentence MUST produce exactly 4 tokens (AutoTranslationPlan §4 Phase 3b)."""
    tok = JanomeTokenizer()
    raw = tok.tokenize("おはようございます、いい天気ですね。")
    filtered = [t for t in raw if keep(t)]
    merged = agglutinate(filtered)

    assert len(merged) == 4
    surfaces = [t.surface for t in merged]
    assert surfaces == ["おはようございます", "いい", "天気", "ですね"]

    # Base form of head morphemes preserved
    assert merged[0].base_form == "おはよう"
    assert merged[1].base_form == "いい"
    assert merged[2].base_form == "天気"
    assert merged[3].base_form == "です"


def test_agglutinate_copula_not_absorbed_into_noun():
    """です must NOT absorb into a preceding noun (e.g. 天気ですね -> 天気 / ですね)."""
    tok = JanomeTokenizer()
    raw = tok.tokenize("今日はいい天気です。")
    merged = agglutinate([t for t in raw if keep(t)])
    surfaces = [t.surface for t in merged]
    assert "天気です" not in surfaces
    assert "天気" in surfaces
    assert "です" in surfaces


def test_agglutinate_te_form_and_polite_request():
    """ちょっと待ってください -> ちょっと / 待ってください (base=待つ)."""
    tok = JanomeTokenizer()
    raw = tok.tokenize("ちょっと待ってください")
    merged = agglutinate([t for t in raw if keep(t)])
    assert len(merged) == 2
    assert merged[0].surface == "ちょっと"
    assert merged[1].surface == "待ってください"
    assert merged[1].base_form == "待つ"


def test_agglutinate_noun_suffix():
    """田中さん / 友達たち merges suffix into noun."""
    tok = JanomeTokenizer()
    raw = tok.tokenize("田中さんたち")
    merged = agglutinate([t for t in raw if keep(t)])
    assert len(merged) == 1
    assert merged[0].surface == "田中さんたち"


def test_agglutinate_verb_past_tense():
    """食べた -> surface 食べた, base 食べる."""
    tok = JanomeTokenizer()
    raw = tok.tokenize("ご飯を食べた")
    merged = agglutinate([t for t in raw if keep(t)])
    tabeta = [t for t in merged if t.surface == "食べた"]
    assert len(tabeta) == 1
    assert tabeta[0].base_form == "食べる"


# --- romaji-integrity regressions (found on live audio) --------------------

def _romaji_for(text):
    from tsutawaru.config import load
    from tsutawaru.nlp.agglutinate import agglutinate
    from tsutawaru.nlp.filters import keep
    from tsutawaru.nlp.romaji import sentence_romaji, token_romaji
    from tsutawaru.nlp.tokenizer import build_tokenizer

    cfg = load().nlp
    toks = agglutinate([t for t in build_tokenizer(cfg).tokenize(text) if keep(t)])
    for t in toks:
        t.romaji = token_romaji(t, cfg.particle_romaji)
    return toks, sentence_romaji(toks, cfg.particle_romaji)


def test_dangling_small_tsu_rejoins_next_token():
    """もっ + ち must merge: a bare っ has no consonant to geminate, so pykakasi
    spells it out as 'motsu' instead of the 'moc-' of 'motchi'."""
    toks, romaji = _romaji_for("かしわもっち")
    assert "motsu" not in romaji, romaji
    assert "motchi" in romaji, romaji
    assert [t.surface for t in toks] == ["かしわ", "もっち"]


def test_merge_keeps_morphemes_with_no_dictionary_reading():
    """妖 has an empty IPADIC reading. Plain reading concatenation dropped it,
    so 妖ちゃん romanised as 'chan' and the word vanished from the line."""
    toks, romaji = _romaji_for("妖ちゃん年下なんじゃない")
    assert romaji.startswith("you"), romaji
    assert toks[0].surface == "妖ちゃん"
    assert toks[0].reading, "merged reading must not be empty"


def test_small_tsu_fix_does_not_regress_known_good_cases():
    for text, expected in [
        ("おはようございます、いい天気ですね。", "ohayougozaimasu ii tenki desune"),
        ("ちょっと待ってください", "chotto mattekudasai"),
        ("私は学生です", "watashi wa gakusei desu"),
        ("今日は日本語を勉強します", "kyou wa nihongo o benkyou shimasu"),
    ]:
        _, romaji = _romaji_for(text)
        assert romaji == expected, f"{text}: {romaji!r} != {expected!r}"


def test_utterance_final_small_tsu_is_not_spelled_out():
    """ねっ has no following token to geminate into; pykakasi renders the bare
    marker as 'netsu'. The correct reading drops it: 'ne'."""
    _, romaji = _romaji_for("ねっ")
    assert romaji == "ne", romaji
    _, romaji = _romaji_for("あっ")
    assert romaji == "a", romaji


def test_numeral_takes_only_one_suffix():
    """IPADIC tags 家 as the profession suffix 〜家 (reading カ), so an unbounded
    suffix chain produced 一回家 'ichikaika' instead of 一回 / 家."""
    toks, romaji = _romaji_for("とりあえず荷物がいっぱいだから一回家帰ろう")
    surfaces = [t.surface for t in toks]
    assert "一回家" not in surfaces, surfaces
    assert "一回" in surfaces and "家" in surfaces, surfaces
    assert "ichikaika" not in romaji, romaji


def test_non_numeric_suffix_chains_still_merge():
    """The numeral guard must not stop ordinary suffixes like 〜たち."""
    toks, _ = _romaji_for("友達たちと行きました")
    assert "友達たち" in [t.surface for t in toks]


# --------------------------------------------------------------------------
# 補助動詞 vs ordinary verbs after て.
#
# Found 2026-08-24 by diffing janome against fugashi on a recorded session, not
# by reading output — the wrong answer is a real word with a plausible gloss
# sitting next to plausible Japanese, so it never looked wrong. 変えて続ける
# merged to one token glossed 変える, losing 続ける; 立ち上げて glossed 立つ,
# "to stand", for a word meaning "to launch".
#
# Every V-て-V case in the tests above is a true auxiliary (待ってください), so
# they exercised only the half of the rule that worked. These cover the other
# half, and run under both backends because the two dictionaries disagree about
# the tag the rule used to trust: IPADIC files 続ける under 非自立 alongside the
# real auxiliaries, and UniDic's 非自立可能 says only that a verb *may* be one.
# --------------------------------------------------------------------------

def _merge(backend: str, text: str):
    cfg = NlpCfg(tokenizer=backend)
    raw = build_tokenizer(cfg).tokenize(text)
    return agglutinate([t for t in raw if keep(t)])


BACKENDS = ["janome"] + (
    ["unidic"] if resolve_unidic_dir() is not None else []
)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("text,head_base", [
    ("見ている", "見る"),          # progressive
    ("見てる", "見る"),            # contraction: carries its own て
    ("なってしまう", "なる"),       # completive
    ("食べてください", "食べる"),    # request
    ("持っていく", "持つ"),         # directional
    ("やってみる", "やる"),         # attemptive
    ("置いてある", "置く"),         # resultative
    ("してくれる", "する"),         # benefactive
    # Spoken contractions, where the connective is fused into the auxiliary and
    # there is no て left for the gate to find. なっちゃいました segments as
    # なっ + ちゃ[接続助詞] + い(いる) + まし + た; やっとく as やっ + とく.
    ("なっちゃいました", "なる"),
    ("食べちゃう", "食べる"),
    ("読んじゃった", "読む"),
    ("やっとく", "やる"),
])
def test_true_auxiliaries_still_merge(backend, text, head_base):
    merged = _merge(backend, text)
    assert len(merged) == 1, [t.surface for t in merged]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("text,parts", [
    ("変えて続ける", ["変えて", "続ける"]),      # two actions, not one
    ("食べ始める", ["食べ", "始める"]),         # aspectual compound
    ("生まれ直します", ["生まれ", "直します"]),   # aspectual compound
])
def test_lexical_verbs_do_not_get_swallowed(backend, text, parts):
    """These carry meaning the head's dictionary form cannot stand in for.

    食べ始める folded into 食べる is "to eat" — the half that says "start" is
    simply gone, and the breakdown is the tier a learner reads for exactly that.
    """
    assert [t.surface for t in _merge(backend, text)] == parts


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("text", ["映画を見る", "荷物を上げる"])
def test_listed_verbs_stay_lexical_without_te(backend, text):
    """見る and 上げる are auxiliaries only in ～てみる and ～てあげる.

    The list is safe to widen precisely because the て gate holds: without it,
    every ordinary 見る in the corpus would fold into whatever preceded it.
    """
    assert len(_merge(backend, text)) == 3
