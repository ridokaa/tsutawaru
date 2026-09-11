"""End-to-end integration tests for the full tsutawaru pipeline."""
from __future__ import annotations

import time
import pytest

from tsutawaru.config import Config
from tsutawaru.models import Segment, Token
from tsutawaru.nlp.pipeline import annotate
from tsutawaru.nlp.tokenizer import JanomeTokenizer
from tsutawaru.pipeline.queues import reset_all, ui_q
from tsutawaru.translate.cache import GlossCache
from tsutawaru.translate.pool import TranslationPool


class StubTranslator:
    name = "stub"  # part of the Translator protocol; the gloss cache keys on it

    def __init__(self, batch_delay: float = 0.0):
        self.batch_delay = batch_delay

    def sentence(self, text: str) -> str:
        return f"EN: {text}"

    def batch(self, texts: list[str]) -> list[str]:
        if self.batch_delay > 0:
            time.sleep(self.batch_delay)
        return [f"gloss_{t}" for t in texts]


@pytest.fixture(autouse=True)
def clean_queues():
    reset_all()
    yield
    reset_all()


def test_nlp_and_translation_pipeline_integration(tmp_path):
    """Feed a segment through NLP annotation and TranslationPool to verify full flow."""
    cfg = Config()
    tok_engine = JanomeTokenizer()
    db_file = str(tmp_path / "e2e_gloss.db")
    cache = GlossCache(cap=100, persist=True, db_path=db_file)
    backend = StubTranslator()
    pool = TranslationPool(backend, cache, cfg.translate)

    seg = Segment.new(stream="Discord", original="おはようございます、いい天気ですね。")
    # Phase 3: NLP
    annotate(seg, tok_engine, cfg.nlp)
    assert len(seg.tokens) == 4
    assert seg.romaji == "ohayougozaimasu ii tenki desune"

    # Phase 4: Submit to translation pool
    pool.submit(seg)

    # Wait for completion events in ui_q
    deadline = time.monotonic() + 3.0
    received = {}
    while time.monotonic() < deadline and len(received) < 2:
        try:
            kind, s = ui_q.get(timeout=0.1)
            received[kind] = s
        except Exception:
            pass

    pool.shutdown()

    assert "english" in received
    assert "breakdown" in received
    assert seg.english == "EN: おはようございます、いい天気ですね。"
    assert seg.tokens[0].gloss == "good morning"  # From static table
    assert seg.tokens[1].gloss == "gloss_いい"
    assert seg.tokens[2].gloss == "gloss_天気"
    assert seg.tokens[3].gloss == "right / isn't it"  # From static table


def test_sentence_lane_isolation(tmp_path):
    """Sentence translation must NOT block behind a slow batch gloss worker (AutoTranslationPlan §4 Phase 4d)."""
    cfg = Config()
    db_file = str(tmp_path / "iso_gloss.db")
    cache = GlossCache(cap=100, persist=True, db_path=db_file)
    # Slow batch delay: 0.5s
    backend = StubTranslator(batch_delay=0.5)
    pool = TranslationPool(backend, cache, cfg.translate)

    seg = Segment.new(stream="Discord", original="今日は天気がいいです")
    seg.tokens = [
        Token(surface="今日", base_form="今日", pos="名詞"),
        Token(surface="天気", base_form="天気", pos="名詞"),
    ]

    t0 = time.perf_counter()
    pool.submit(seg)

    # Wait only for sentence translation ('english')
    english_received = False
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        try:
            kind, s = ui_q.get(timeout=0.05)
            if kind == "english":
                sentence_latency = time.perf_counter() - t0
                english_received = True
                break
        except Exception:
            pass

    pool.shutdown()

    assert english_received is True
    # Sentence translation must arrive well before the 0.5s gloss batch completes
    assert sentence_latency < 0.25


# ------------------------------------------------------ warmup is a gate


class _DeadEngine:
    """An engine that cannot transcribe anything, the way a real one fails."""

    def __init__(self):
        self.calls = 0

    def warmup(self):
        self.transcribe(None)

    def transcribe(self, audio):
        self.calls += 1
        raise RuntimeError("There is no Stream(gpu, 1) in current thread.")


def _orchestrator_with(engine):
    from tsutawaru.config import load
    from tsutawaru.pipeline.orchestrator import Orchestrator

    orch = Orchestrator(load())
    orch.stages.stt = engine
    return orch


def test_a_failed_warmup_stops_the_run():
    """Regression: a broken engine used to run a full session in silence.

    On 2026-09-04 an MLX thread-affinity fault made every Qwen transcription
    raise. Warmup caught it, logged it, and continued; the pipeline then
    captured audio, segmented it correctly, and emitted nothing for seven
    minutes. A crash would have been found in seconds — looking healthy is what
    made it expensive.
    """
    orch = _orchestrator_with(_DeadEngine())
    assert not orch.stop_evt.is_set()
    orch._stt_worker()
    assert orch.stop_evt.is_set(), "a run that cannot transcribe must not continue"


def test_a_failed_warmup_does_not_go_on_to_consume_utterances():
    """Stopping means stopping — not draining the queue into the same fault."""
    import numpy as np

    from tsutawaru.models import Utterance
    from tsutawaru.pipeline.queues import utt_q

    engine = _DeadEngine()
    utt_q.put_latest(Utterance.new(audio=np.zeros(1600, dtype=np.float32),
                                   t_start=0.0, t_end=0.1))
    _orchestrator_with(engine)._stt_worker()

    assert engine.calls == 1, "only warmup should have run"
    assert utt_q.qsize() == 1, "the queued utterance was consumed anyway"
    utt_q.get_nowait()


def test_the_failure_names_the_engine_and_says_what_to_do(caplog):
    """The message is the only thing standing between this and another lost hour."""
    with caplog.at_level("ERROR"):
        _orchestrator_with(_DeadEngine())._stt_worker()

    assert "_DeadEngine" in caplog.text
    assert "stopping" in caplog.text
    assert "--model kotoba" in caplog.text
    # The underlying fault has to survive into the log, not just our summary.
    assert "Stream(gpu, 1)" in caplog.text
