"""JMdict gloss lookup: ranking, POS handling, and the pool's use of it.

The lookup rules are the whole point of this module — which sense of a word the
breakdown shows is the difference between "safe / secure" and "great man" for
大丈夫. Those cases are pinned here against a small hand-built database so the
test runs without the 41 MB real one.
"""
from __future__ import annotations

import sqlite3

import pytest

from tsutawaru.models import Token
from tsutawaru.translate import jmdict


@pytest.fixture
def db(tmp_path, monkeypatch):
    """A miniature jmdict.db holding only the rows these cases turn on."""
    p = tmp_path / "jmdict.db"
    con = sqlite3.connect(str(p))
    con.execute("CREATE TABLE jm(k TEXT, pos TEXT, gloss TEXT, prio INT)")
    con.executemany("INSERT INTO jm VALUES(?,?,?,?)", [
        # 大丈夫: IPADIC calls it 名詞, JMdict files the everyday sense as adj-na.
        ("大丈夫", "adj", "safe / secure", 0),
        ("大丈夫", "n", "great man / fine figure of a man", 20),
        # 聞き取る is listed; its potential 聞き取れる is not, like most potentials.
        ("聞き取る", "v", "to catch (someone's words) / to make out", 0),
        # A word whose common noun sense should beat a rare verb sense even when
        # the token is tagged 動詞 — the POS penalty must not outweigh priority.
        ("話", "n", "talk / story", 0),
        ("話", "v", "archaic verb sense", 60),
    ])
    con.commit()
    con.close()
    monkeypatch.setattr(jmdict, "db_path", lambda: p)
    monkeypatch.setattr(jmdict, "_lookup", jmdict._Lookup())
    return p


def _tok(base, pos="名詞", pos1="一般"):
    return Token(surface=base, base_form=base, pos=pos, pos1=pos1)


def test_na_adjective_beats_the_archaic_noun(db):
    """The bug this ranking exists for: a hard `WHERE pos='n'` returns "great man"."""
    assert jmdict.gloss(_tok("大丈夫", "名詞", "形容動詞語幹")) == "safe / secure"


def test_pos_preference_does_not_outrank_priority(db):
    """POS is a tiebreak, not a filter — a rare sense in the right class loses."""
    assert jmdict.gloss(_tok("話", "動詞", "自立")) == "talk / story"


def test_potential_verb_falls_back_to_its_plain_form(db):
    got = jmdict.gloss(_tok("聞き取れる", "動詞", "自立"))
    assert got == "can catch (someone's words) / can make out"


def test_potential_fallback_only_fires_on_a_miss(db):
    """聞き取る is in the dictionary, so it must never be mangled into 聞き取う."""
    assert jmdict.gloss(_tok("聞き取る", "動詞", "自立")).startswith("to catch")


def test_potential_fallback_is_verbs_only(db):
    """A noun ending in える must not be deconjugated."""
    assert jmdict.gloss(_tok("聞き取れる", "名詞", "一般")) is None


def test_unknown_word_returns_none(db):
    assert jmdict.gloss(_tok("アボルスコ")) is None


def test_missing_database_degrades_instead_of_raising(tmp_path, monkeypatch):
    """No jmdict.db means the pool falls back to the online backend, not a crash."""
    monkeypatch.setattr(jmdict, "db_path", lambda: tmp_path / "absent.db")
    monkeypatch.setattr(jmdict, "_lookup", jmdict._Lookup())
    assert jmdict.gloss(_tok("猫")) is None


def test_depotential_rules():
    assert jmdict._depotential("会える") == "会う"
    assert jmdict._depotential("聞き取れる") == "聞き取る"
    assert jmdict._depotential("泳げる") == "泳ぐ"
    assert jmdict._depotential("話せる") == "話す"
    assert jmdict._depotential("猫") is None
    assert jmdict._depotential("する") is None  # too short to strip


def test_pool_consults_jmdict_before_the_backend(db, monkeypatch):
    """The gloss lane must not spend a network call on a word the dictionary has."""
    from tsutawaru.config import TranslateCfg
    from tsutawaru.models import Segment
    from tsutawaru.pipeline import queues
    from tsutawaru.translate.cache import GlossCache
    from tsutawaru.translate.pool import TranslationPool

    calls = []

    class Backend:
        name = "fake"

        def sentence(self, text):
            return "s"

        def batch(self, texts):
            calls.append(list(texts))
            return ["from-network" for _ in texts]

    cfg = TranslateCfg(persist_cache=False)
    pool = TranslationPool(Backend(), GlossCache(10, persist=False), cfg)
    seg = Segment.new(stream="main", original="大丈夫アボルスコ")
    seg.tokens = [_tok("大丈夫", "名詞", "形容動詞語幹"), _tok("アボルスコ")]
    try:
        pool._glosses(seg)
    finally:
        pool.shutdown()
        while not queues.ui_q.empty():
            queues.ui_q.get_nowait()

    assert seg.tokens[0].gloss == "safe / secure"
    assert seg.tokens[1].gloss == "from-network"
    assert calls == [["アボルスコ"]], "only the dictionary miss should reach the backend"
