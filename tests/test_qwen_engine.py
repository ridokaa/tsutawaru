"""Qwen3-ASR as an opt-in STT backend.

Two things are worth pinning here and they are not the obvious ones.

The first is that this change is *inert by default*. Every test below has to
opt in explicitly; a config nobody touched still builds kotoba-whisper. That is
the whole revert story, so it gets a test rather than a comment.

The second is what happens to the confidence filters. Qwen3-ASR reports no
avg_logprob and no no_speech_prob, and stt/filters.py has three branches plus a
decoder threshold that key on those numbers. The engine therefore reports
documented "not reported" sentinels — and a sentinel is only safe if every one
of those branches abstains on it. A value that happened to sit the wrong side
of a threshold would discard real speech on every line, silently, and the
transcript would simply be shorter than the conversation. That is the failure
these tests exist to make impossible.

No model is downloaded. Session is replaced with a stub, because what is under
test is the wiring and the sentinels, not Qwen's transcription.
"""
from __future__ import annotations

import importlib.machinery
import sys
import threading
import types

import numpy as np
import pytest

from tsutawaru.config import SttCfg, load


class _FakeSession:
    """Stands in for mlx_qwen3_asr.Session. Records what it was called with."""

    def __init__(self, repo, **kw):
        self.repo = repo
        self.calls = []
        self.thread = threading.get_ident()

    def transcribe(self, audio, **kw):
        self.calls.append(kw)
        return types.SimpleNamespace(
            text="  カフェテリアでブランを発見  ",
            language="ja",
            finish_reason="stop",
            truncated=False,
        )


@pytest.fixture
def engine(monkeypatch):
    """A QwenMLXEngine wired to the stub, with its Session reachable."""
    from tsutawaru.stt import qwen_mlx_engine as qme

    made = {}

    def _factory(repo, **kw):
        made["session"] = _FakeSession(repo, **kw)
        return made["session"]

    # A real ModuleType with a __spec__, not a SimpleNamespace: the factory
    # calls importlib.util.find_spec on this name, and find_spec raises rather
    # than returning None when an already-imported module has no spec.
    stub = types.ModuleType("mlx_qwen3_asr")
    stub.__spec__ = importlib.machinery.ModuleSpec("mlx_qwen3_asr", loader=None)
    stub.Session = _factory
    monkeypatch.setitem(sys.modules, "mlx_qwen3_asr", stub)

    def build(**over):
        cfg = SttCfg(model="qwen3", **over)
        eng = qme.QwenMLXEngine(cfg)
        # `made` fills in when the engine first loads, which is deliberately
        # not at construction — see _ensure_session.
        eng.made = made
        return eng

    return build


# ------------------------------------------------------------------ inertness


def test_the_default_config_still_builds_whisper():
    """The revert story: an untouched config never reaches this code path."""
    from tsutawaru.stt.qwen_mlx_engine import REPO as QWEN_MODELS

    assert SttCfg().model == "kotoba"
    assert SttCfg().model not in QWEN_MODELS


def test_the_new_names_are_accepted_by_config():
    for name in ("qwen3", "qwen3-small"):
        cfg = load(None, {"stt": {"model": name}})
        assert cfg.stt.model == name, f"{name} was rejected"
        assert not cfg.validate()


# ------------------------------------------------- the sentinels and filters


def test_the_sentinels_make_every_confidence_filter_abstain():
    """The load-bearing test.

    stt/filters.py tests avg_logprob against two floors and no_speech_prob
    against two ceilings. With no real values to give it, the engine must
    report numbers that fall on the *keep* side of all four — otherwise
    selecting this model throws away speech it never even mis-heard.
    """
    from tsutawaru.stt.filters import is_hallucination
    from tsutawaru.stt.qwen_mlx_engine import NO_LOGPROB, NO_SPEECH_PROB

    cfg = SttCfg(model="qwen3")
    res = types.SimpleNamespace(
        avg_logprob=NO_LOGPROB, no_speech_prob=NO_SPEECH_PROB
    )

    # An ordinary line, and a short one that only the confidence branch could
    # have killed (no peak information is passed, so the amplitude gate cannot
    # fire and the decision rests entirely on the sentinels).
    assert not is_hallucination("結構早いな", res, cfg)
    assert not is_hallucination("はい", res, cfg)

    # And the floors really are the thing being cleared, not a coincidence of
    # the defaults: even wound far tighter than any config would set them.
    assert not is_hallucination("はい", res, SttCfg(logprob_floor=-0.01,
                                                   no_speech_thresh=0.01))


def test_the_string_and_repetition_filters_still_work_without_confidence():
    """What survives the switch: everything that reads only the text."""
    from tsutawaru.stt.filters import is_hallucination
    from tsutawaru.stt.qwen_mlx_engine import NO_LOGPROB, NO_SPEECH_PROB

    cfg = SttCfg(model="qwen3")
    res = types.SimpleNamespace(
        avg_logprob=NO_LOGPROB, no_speech_prob=NO_SPEECH_PROB
    )
    assert is_hallucination("ご視聴ありがとうございました", res, cfg)
    assert is_hallucination("ああああああああ", res, cfg)
    assert is_hallucination("そうそうそうそうそう", res, cfg)


def test_the_amplitude_gate_still_fires_without_confidence():
    """The filter that actually removed junk on 2026-08-22 is engine-agnostic.

    It is computed from the waveform in the orchestrator, so it is unaffected
    by the model reporting no scores — which is the reason the confidence loss
    is acceptable rather than merely survivable.
    """
    from tsutawaru.stt.filters import is_hallucination
    from tsutawaru.stt.qwen_mlx_engine import NO_LOGPROB, NO_SPEECH_PROB

    res = types.SimpleNamespace(
        avg_logprob=NO_LOGPROB, no_speech_prob=NO_SPEECH_PROB
    )
    assert is_hallucination("はい", res, SttCfg(model="qwen3"),
                            peak=0.02, peak_ref=0.73)


def test_a_segment_built_from_this_engine_reports_not_reported():
    """The sentinels are the ones models.Segment already documents."""
    from tsutawaru.models import Segment
    from tsutawaru.stt.qwen_mlx_engine import NO_LOGPROB, NO_SPEECH_PROB

    assert Segment.new(stream="m", original="x").confidence == NO_LOGPROB
    assert Segment.new(stream="m", original="x").no_speech == NO_SPEECH_PROB


# ------------------------------------------------------------------- engine


def test_transcribe_returns_stripped_text_and_the_sentinels(engine):
    from tsutawaru.stt.qwen_mlx_engine import NO_LOGPROB, NO_SPEECH_PROB

    r = engine().transcribe(np.zeros(16000, dtype=np.float32))
    assert r.text == "カフェテリアでブランを発見"
    assert r.language == "ja"
    assert r.avg_logprob == NO_LOGPROB
    assert r.no_speech_prob == NO_SPEECH_PROB


def test_the_model_is_loaded_once_not_per_utterance(engine):
    """2.5 GB of weights; a reload per line would dwarf the latency budget."""
    eng = engine()
    assert eng.session is None, "construction must not load the model"
    eng.transcribe(np.zeros(16000, dtype=np.float32))
    first = eng.session
    for _ in range(3):
        eng.transcribe(np.zeros(16000, dtype=np.float32))
    assert eng.session is first
    assert len(eng.made["session"].calls) == 4


def test_initial_prompt_is_passed_as_qwens_biasing_context(engine):
    """Whisper's initial_prompt can be echoed into the transcript; this cannot.

    Same config key, better mechanism — this is where the mangled proper nouns
    (カベテリア, ディスコート) would be corrected if anywhere.
    """
    eng = engine(initial_prompt="Discord カフェテリア ブラン")
    eng.transcribe(np.zeros(16000, dtype=np.float32))
    assert eng.made["session"].calls[0]["context"] == "Discord カフェテリア ブラン"


def test_language_is_pinned_or_left_to_detection(engine):
    eng = engine(lang_mode="pinned", language="ja")
    eng.transcribe(np.zeros(16000, dtype=np.float32))
    assert eng.made["session"].calls[0]["language"] == "ja"

    eng = engine(lang_mode="detect")
    eng.transcribe(np.zeros(16000, dtype=np.float32))
    assert eng.made["session"].calls[0]["language"] is None


def test_the_confidence_loss_is_announced_at_startup(engine, caplog):
    """Discovering this from a thin transcript weeks later would be worse."""
    with caplog.at_level("WARNING"):
        engine()
    assert "no avg_logprob" in caplog.text
    assert "logprob_floor" in caplog.text


# ------------------------------------------------------- MLX thread affinity


def test_the_model_loads_on_the_thread_that_runs_inference(engine):
    """Regression: MLX streams are thread-local.

    build_engine() runs on the main thread during stage construction, and
    transcribe() is only ever called from the STT worker. A Session built at
    construction therefore lived on the wrong thread, and every utterance died
    with "There is no Stream(gpu, 1) in current thread" — caught per-utterance
    and logged, so the pipeline ran on looking healthy while dropping all of
    its output. A 7-minute live session recorded zero lines.

    The stub cannot raise that error, so this asserts the property that
    prevents it instead: the load happens on whichever thread first asks, not
    on whichever thread built the engine.
    """
    eng = engine()
    assert eng.session is None

    seen = {}

    def worker():
        seen["thread"] = threading.get_ident()
        eng.transcribe(np.zeros(16000, dtype=np.float32))
        seen["loaded_on"] = eng.made["session"].thread

    t = threading.Thread(target=worker)
    t.start()
    t.join()

    assert seen["loaded_on"] == seen["thread"], "model loaded on the wrong thread"
    assert seen["loaded_on"] != threading.get_ident(), "test did not cross a thread"


def test_warmup_is_what_moves_the_load_onto_the_worker(engine):
    """The worker calls warmup() first, so deferring costs no runtime latency.

    Without this the fix would merely relocate the stall: the weights would
    load on the first real utterance instead of at startup, putting a
    multi-second pause in front of the first thing anyone says.
    """
    eng = engine()
    eng.warmup()
    assert eng.session is not None


# ------------------------------------------------------------------ routing


def test_the_factory_routes_the_qwen_names_here(engine, monkeypatch):
    from tsutawaru.stt import factory
    from tsutawaru.stt.qwen_mlx_engine import QwenMLXEngine

    monkeypatch.setattr(factory.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(factory.platform, "machine", lambda: "arm64")
    assert isinstance(factory.build_engine(SttCfg(model="qwen3")), QwenMLXEngine)


def test_a_missing_runtime_raises_instead_of_running_another_model(monkeypatch):
    """Silent fallback would invalidate every comparison made afterwards.

    The point of switching models is to find out whether Qwen is better. A run
    that quietly used whisper-small because an import failed would answer that
    question with the wrong model's output and nothing would say so.
    """
    from tsutawaru.stt import factory

    monkeypatch.setattr(factory.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(factory.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(factory.importlib.util, "find_spec", lambda n: None)

    with pytest.raises(RuntimeError, match="mlx-qwen3-asr"):
        factory.build_engine(SttCfg(model="qwen3"))


def test_selecting_qwen_off_apple_silicon_raises(monkeypatch):
    from tsutawaru.stt import factory

    monkeypatch.setattr(factory.platform, "system", lambda: "Windows")
    with pytest.raises(RuntimeError, match="Apple Silicon"):
        factory.build_engine(SttCfg(model="qwen3"))


def test_whisper_models_are_untouched_by_the_new_branch(monkeypatch):
    """Regression: the branch must not capture names it does not own."""
    from tsutawaru.stt import factory

    monkeypatch.setattr(factory.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(factory.platform, "machine", lambda: "arm64")
    pytest.importorskip("mlx_whisper")
    from tsutawaru.stt.mlx_whisper_engine import MLXWhisperEngine

    eng = factory.build_engine(SttCfg(model="kotoba"))
    assert isinstance(eng, MLXWhisperEngine)
    assert eng.repo == "kaiinui/kotoba-whisper-v2.0-mlx"
