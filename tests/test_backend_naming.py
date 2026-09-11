"""Backend identity: cache namespacing, transcript attribution, MPS safety.

What config asks for is not always what it gets — provider="deepl" with no API
key silently becomes Google — so the gloss cache and the UI both key off the
backend that actually answered rather than off TranslateCfg.
"""
from __future__ import annotations

import threading

import pytest

from tsutawaru.config import TranslateCfg, UiCfg
from tsutawaru.models import Segment, Token
from tsutawaru.pipeline.queues import reset_all
from tsutawaru.translate.cache import GlossCache
from tsutawaru.translate.pool import TranslationPool, _name


class FakeBackend:
    def __init__(self, name: str):
        self.name = name

    def sentence(self, text: str) -> str:
        return f"{self.name}:{text}"

    def batch(self, texts: list[str]) -> list[str]:
        return [f"{self.name}-gloss" for _ in texts]


class Nameless:
    def sentence(self, text: str) -> str:
        return text

    def batch(self, texts: list[str]) -> list[str]:
        return list(texts)


@pytest.fixture
def cfg() -> TranslateCfg:
    return TranslateCfg(provider="google", gloss_workers=1, persist_cache=False)


def _pool(backend, cfg, cache=None) -> TranslationPool:
    reset_all()
    return TranslationPool(backend, cache or GlossCache(100, persist=False), cfg)


def _seg(text: str = "犬") -> Segment:
    return Segment.new(
        stream="main", original=text,
        tokens=[Token(surface=text, base_form=text, pos="名詞", pos1="一般")],
    )


def _run(pool: TranslationPool, seg: Segment) -> Segment:
    pool.submit(seg)
    ev = threading.Event()
    for _ in range(200):
        if seg.english and not seg.partial:
            break
        ev.wait(0.01)
    return seg


# ------------------------------------------------------------- cache identity

def test_gloss_cache_is_namespaced_by_backend_not_config(cfg):
    """Two backends must never share a cache row, however cfg is set.

    The L2 cache is SQLite on disk, so a row written under the wrong name is
    wrong permanently — it outlives the process that made the mistake.
    """
    cache = GlossCache(100, persist=False)
    a = _pool(FakeBackend("google"), cfg, cache)
    b = _pool(FakeBackend("marian"), cfg, cache)
    try:
        _run(a, _seg("犬"))
        _run(b, _seg("犬"))
    finally:
        a.shutdown()
        b.shutdown()

    assert cfg.provider == "google"          # config never moved
    assert cache.get("google:犬:名詞") == "google-gloss"
    assert cache.get("marian:犬:名詞") == "marian-gloss"


def test_a_backend_reads_only_its_own_cached_glosses(cfg):
    cache = GlossCache(100, persist=False)
    cache.put("google:猫:名詞", "poisoned")
    p = _pool(FakeBackend("marian"), cfg, cache)
    try:
        assert _run(p, _seg("猫")).tokens[0].gloss == "marian-gloss"
    finally:
        p.shutdown()


def test_unnamed_backend_does_not_kill_the_lane(cfg):
    """A backend missing `name` should degrade, not take translation down."""
    p = _pool(Nameless(), cfg)
    try:
        seg = _run(p, _seg("鳥"))
    finally:
        p.shutdown()
    assert seg.english == "鳥"
    assert _name(Nameless()) == "unknown"


# ------------------------------------------------------------- backend identity

def test_online_backend_reports_google_when_deepl_has_no_key():
    """The whole reason identity comes from the backend and not from config."""
    from tsutawaru.translate.online import OnlineTranslator

    assert OnlineTranslator(TranslateCfg(provider="deepl", deepl_api_key="")).name == "google"
    assert OnlineTranslator(TranslateCfg(provider="deepl", deepl_api_key="k")).name == "deepl"
    assert OnlineTranslator(TranslateCfg(provider="google")).name == "google"


def test_block_header_is_just_the_stream_name():
    """No backend tag in the transcript — the header stays 'MAIN'."""
    from tsutawaru.ui.window_qt import _fmt_block

    html = _fmt_block(
        Segment.new(stream="main", original="大事だね!", english="It's important"), UiCfg()
    )
    assert '<div class="stream">main</div>' in html
    for word in ("google", "deepl", "GOOGLE", "·"):
        assert word not in html


# ------------------------------------------------------------ provider set

def test_retired_provider_is_rejected_on_the_cli():
    """`--provider marian` must not be silently accepted.

    The override path used to setattr without checking _CHOICES, so a retired
    value survived into the run and then resolved to whatever the builder
    happened to do with it — the process disagreeing with its own command line.
    """
    from tsutawaru.config import load

    assert load(overrides={"translate": {"provider": "marian"}}).translate.provider == "google"
    assert load(overrides={"translate": {"provider": "deepl"}}).translate.provider == "deepl"


def test_a_rejected_override_does_not_discard_the_rest(monkeypatch):
    """A bad flag keeps the current value; it must not reset unrelated ones."""
    from tsutawaru.config import load

    cfg = load(overrides={"translate": {"provider": "marian", "gloss_workers": 7}})
    assert cfg.translate.provider == "google"
    assert cfg.translate.gloss_workers == 7


def test_choice_checking_applies_to_every_flag_not_just_provider():
    from tsutawaru.config import load

    assert load(overrides={"stt": {"model": "huge"}}).stt.model == "kotoba"
    assert load(overrides={"ui": {"sink": "banana"}}).ui.sink == "window"
    assert load(overrides={"stt": {"model": "medium"}}).stt.model == "medium"
