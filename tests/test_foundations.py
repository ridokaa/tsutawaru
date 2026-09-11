"""Tests for the Phase 0 foundations: models, queues, config, logbus."""
from __future__ import annotations

import logging
import sys
import textwrap

import pytest

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from tsutawaru import config as cfgmod
from tsutawaru.logbus import Metrics, RateLimitedWarner
from tsutawaru.models import Segment, Token, Utterance
from tsutawaru.pipeline.queues import DropOldestQueue, reset_all


# ------------------------------------------------------------------ models

def test_segment_ids_are_unique_and_monotonic():
    a, b = Segment.new(stream="x", original="あ"), Segment.new(stream="x", original="い")
    assert b.id > a.id


def test_utterance_duration():
    import numpy as np

    u = Utterance.new(audio=np.zeros(16000, np.float32), t_start=0.0, t_end=1.0)
    assert u.duration == pytest.approx(1.0)


def test_segment_to_dict_is_json_safe():
    import json

    seg = Segment.new(stream="Discord", original="天気")
    seg.tokens = [Token(surface="天気", base_form="天気", romaji="tenki", gloss="weather")]
    json.dumps(seg.to_dict())  # must not raise


def test_token_defaults_gloss_pending():
    """None means pending, "" means unavailable — the formatter distinguishes them."""
    t = Token(surface="は", base_form="は")
    assert t.gloss is None


# ------------------------------------------------------------------ queues

def test_drop_oldest_evicts_oldest_not_newest():
    q = DropOldestQueue(maxsize=3, name="t")
    for i in range(5):
        q.put_latest(i)
    assert [q.get_nowait() for _ in range(3)] == [2, 3, 4]
    assert q.dropped == 2


def test_drop_oldest_never_blocks_when_full():
    q = DropOldestQueue(maxsize=1, name="t")
    for i in range(1000):
        q.put_latest(i)  # would deadlock with a plain Queue.put
    assert q.qsize() == 1


def test_reset_all_clears_and_zeroes_counters():
    from tsutawaru.pipeline.queues import raw_q

    for i in range(600):
        raw_q.put_latest(i)
    assert raw_q.dropped > 0
    reset_all()
    assert raw_q.qsize() == 0 and raw_q.dropped == 0


# ------------------------------------------------------------------ metrics

def test_percentiles_and_counts():
    m = Metrics()
    for v in range(1, 101):
        m.record("stt", float(v))
    s = m.snapshot()["stt"]
    assert s["n"] == 100
    assert s["max"] == 100.0
    assert 45 <= s["p50"] <= 55
    assert 90 <= s["p95"] <= 100


def test_window_bounds_memory():
    m = Metrics(window=10)
    for v in range(100):
        m.record("x", float(v))
    assert m.snapshot()["x"]["n"] == 100      # count is cumulative
    assert m.snapshot()["x"]["max"] == 99.0   # but only the last 10 are kept


def test_timer_context_manager():
    m = Metrics()
    with m.timer("nlp"):
        pass
    assert m.snapshot()["nlp"]["n"] == 1


def test_format_table_handles_empty():
    assert "no metrics" in Metrics().format_table()


def test_rate_limited_warner_fires_once_then_rearms(caplog):
    log = logging.getLogger("test.warner")
    w = RateLimitedWarner(log)
    with caplog.at_level(logging.WARNING):
        assert w.warn("boom") is True
        assert w.warn("boom") is False   # suppressed
        assert w.warn("boom") is False
        w.rearm()
        assert w.warn("boom") is True
    assert caplog.text.count("boom") == 2


# ------------------------------------------------------------------- config

def test_defaults_load_without_a_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = cfgmod.load()
    assert cfg.stt.model == "kotoba"
    assert cfg.vad.threshold == 0.55
    assert cfg.nlp.agglutinate is True
    assert cfg.source_path is None


def test_file_overrides_defaults(tmp_path, monkeypatch):
    (tmp_path / "config.toml").write_text(textwrap.dedent("""
        [stt]
        model = "base"
        [vad]
        threshold = 0.7
    """), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    cfg = cfgmod.load()
    assert cfg.stt.model == "base"
    assert cfg.vad.threshold == 0.7
    assert cfg.stt.beam_size == 1  # untouched keys keep their default


def test_cli_overrides_beat_file(tmp_path, monkeypatch):
    (tmp_path / "config.toml").write_text('[stt]\nmodel = "base"\n', encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    cfg = cfgmod.load(overrides={"stt": {"model": "medium"}})
    assert cfg.stt.model == "medium"


def test_none_override_does_not_clobber_file_value(tmp_path, monkeypatch):
    """Unset CLI flags arrive as None and must be ignored, not written through."""
    (tmp_path / "config.toml").write_text('[stt]\nmodel = "base"\n', encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    cfg = cfgmod.load(overrides={"stt": {"model": None}})
    assert cfg.stt.model == "base"


def test_unknown_key_warns_and_is_ignored(tmp_path, monkeypatch, caplog):
    (tmp_path / "config.toml").write_text('[stt]\nmdoel = "base"\n', encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    with caplog.at_level(logging.WARNING):
        cfg = cfgmod.load()
    assert "unknown key" in caplog.text and "mdoel" in caplog.text
    assert cfg.stt.model == "kotoba"


def test_invalid_choice_falls_back_to_default(tmp_path, monkeypatch, caplog):
    (tmp_path / "config.toml").write_text('[translate]\nprovider = "bing"\n', encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    with caplog.at_level(logging.WARNING):
        cfg = cfgmod.load()
    assert cfg.translate.provider == "google"


def test_missing_explicit_config_raises(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError):
        cfgmod.load("nope.toml")


def test_example_config_parses_and_matches_defaults(monkeypatch):
    """config.example.toml must stay in sync with the dataclass defaults."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    monkeypatch.chdir(root)
    from_file = cfgmod.load(str(root / "config.example.toml"))
    defaults = cfgmod.Config()
    for section in ("audio", "vad", "stt", "nlp", "translate", "ui"):
        assert getattr(from_file, section) == getattr(defaults, section), section


# --------------------------------------------------------------- validation

def test_validate_catches_inverted_utterance_bounds():
    cfg = cfgmod.Config()
    cfg.vad.min_utterance_ms, cfg.vad.max_utterance_ms = 5000, 1000
    assert any("min_utterance_ms" in e for e in cfg.validate())


def test_validate_catches_out_of_range_threshold():
    cfg = cfgmod.Config()
    cfg.vad.threshold = 1.5
    assert any("threshold" in e for e in cfg.validate())


def test_validate_catches_deepl_without_key():
    cfg = cfgmod.Config()
    cfg.translate.provider = "deepl"
    assert any("deepl_api_key" in e for e in cfg.validate())


@pytest.mark.skipif(sys.version_info < (3, 14), reason="3.14-only constraint")
def test_validate_rejects_webrtc_on_py314():
    cfg = cfgmod.Config()
    cfg.vad.backend = "webrtc"
    errs = cfg.validate()
    assert any("webrtc" in e and "3.14" in e for e in errs)


def test_clean_config_validates():
    assert cfgmod.Config().validate() == []


def test_warner_does_not_build_message_while_suppressed(caplog):
    """Regression: the silent-stream message shells out to system_profiler
    (~200 ms). Building it eagerly on every suppressed call starved the VAD
    thread and overflowed raw_q at ~600 dropped frames per 30 s."""
    log = logging.getLogger("test.lazy")
    w = RateLimitedWarner(log)
    calls = {"n": 0}

    def expensive() -> str:
        calls["n"] += 1
        return "costly message"

    with caplog.at_level(logging.WARNING):
        assert w.warn(expensive) is True
        assert calls["n"] == 1
        for _ in range(100):
            assert w.warn(expensive) is False
        assert calls["n"] == 1, "message was built while suppressed"
        w.rearm()
        assert w.warn(expensive) is True
        assert calls["n"] == 2


def test_warner_still_accepts_plain_strings():
    """Callable support must not break the simple case."""
    w = RateLimitedWarner(logging.getLogger("test.str"))
    assert w.warn("plain") is True
    assert w.warn("plain") is False


def test_arm_once_claims_without_emitting(caplog):
    """arm_once() lets a caller emit off-thread; it must not log by itself."""
    w = RateLimitedWarner(logging.getLogger("test.armonce"))
    with caplog.at_level(logging.WARNING):
        assert w.arm_once() is True
        assert w.arm_once() is False
        w.rearm()
        assert w.arm_once() is True
    assert caplog.text == ""
