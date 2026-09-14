"""Provisional transcripts: a prefix shown while speech is still running.

The regressions worth guarding are the ones that silently corrupt the
transcript rather than crash: a provisional that evicts real audio from the
drop-oldest queue, a final that lands on a different line than its preview, and
a preview that reaches the translator as half a sentence.
"""
from __future__ import annotations

import collections
from types import SimpleNamespace

import numpy as np
import pytest

from tsutawaru.audio.vad import SileroVADGate, WIN
from tsutawaru.config import VadCfg
from tsutawaru.models import Segment, Utterance
from tsutawaru.pipeline.queues import reset_all, utt_q


@pytest.fixture(autouse=True)
def clean_queues():
    reset_all()
    yield
    reset_all()


def _gate(**kw):
    cfg = VadCfg(hangover_ms=64, min_utterance_ms=100, **kw)
    gate = SileroVADGate(cfg)
    return gate


def _speak(gate, frames, speech=True):
    gate._prob = lambda w: 0.9 if speech else 0.0
    for _ in range(frames):
        gate._window(np.ones(WIN, dtype=np.float32) * 0.5)


def _drain():
    out = []
    while not utt_q.empty():
        out.append(utt_q.get_nowait())
    return out


def test_provisional_fires_and_final_shares_the_line():
    # 32 ms per window, so 20 windows is 640 ms of speech.
    gate = _gate(provisional_after_ms=320)
    _speak(gate, 20)
    _speak(gate, 4, speech=False)  # hangover_ms=64 -> 2 windows closes it

    utts = _drain()
    prov = [u for u in utts if u.provisional]
    final = [u for u in utts if not u.provisional]
    assert prov, "no provisional emitted past the threshold"
    assert final, "the utterance never closed"
    assert prov[0].line_id == final[-1].line_id, "preview landed on another line"
    assert len(final[-1].audio) > len(prov[0].audio), "final must be the whole run"


def test_provisional_off_by_zero():
    gate = _gate(provisional_after_ms=0)
    _speak(gate, 40)
    _speak(gate, 4, speech=False)
    assert not [u for u in _drain() if u.provisional]


def test_provisional_never_evicts_real_audio():
    """utt_q is drop-oldest. A preview must yield to anything already queued."""
    gate = _gate(provisional_after_ms=320)
    utt_q.put_latest("a real utterance waiting for STT")
    _speak(gate, 30)
    assert not [u for u in _drain() if getattr(u, "provisional", False)]


def test_carried_remainder_starts_a_new_line():
    """A forced flush shows its line; what follows must not overwrite it."""
    gate = _gate(max_utterance_ms=320, provisional_after_ms=0)
    _speak(gate, 40)
    ids = {u.line_id for u in _drain() if not u.provisional}
    assert len(ids) > 1, f"forced flushes all reused one line id: {ids}"


class _Rec:
    def __init__(self):
        self.segs = []

    def add(self, seg):
        self.segs.append(seg)


class _Res:
    def __init__(self, text):
        self.text, self.language = text, "ja"


def _orch():
    """A bare Orchestrator: the two methods under test touch nothing else."""
    from tsutawaru.config import Config
    from tsutawaru.pipeline.orchestrator import Orchestrator

    o = Orchestrator.__new__(Orchestrator)
    o.cfg = Config()
    o.recorder = _Rec()
    o._provisional = {}
    o._peaks = collections.deque(maxlen=64)
    o.stages = SimpleNamespace(stt_compare=None)
    return o


def test_final_revises_the_preview_in_place():
    """One line, not two: the window keys on seg.id and the recorder holds it."""
    o = _orch()
    utt = Utterance.new(audio=np.zeros(16000, np.float32), t_start=0.0, t_end=1.0,
                        line_id=7, provisional=True)
    o._show_provisional(utt, "これは")
    assert len(o.recorder.segs) == 1
    preview = o.recorder.segs[0]
    assert preview.provisional is True and preview.original == "これは"

    final = Utterance.new(audio=np.ones(32000, np.float32) * 0.5, t_start=0.0,
                          t_end=2.0, line_id=7)
    o._emit_segment(final, _Res("これはペンです。"))

    assert len(o.recorder.segs) == 1, "the revision registered a second line"
    assert preview.id == o.recorder.segs[0].id
    assert preview.original == "これはペンです。", "final text did not replace it"
    assert preview.provisional is False, "line stayed marked provisional"
    assert preview.romaji == "" and preview.tokens == [], "stale annotation kept"
    assert not o._provisional, "line left in the pending map"


def test_final_without_a_preview_is_a_new_line():
    o = _orch()
    utt = Utterance.new(audio=np.ones(16000, np.float32) * 0.5, t_start=0.0,
                        t_end=1.0, line_id=9)
    o._emit_segment(utt, _Res("はい、そうです。"))
    assert len(o.recorder.segs) == 1
    assert o.recorder.segs[0].provisional is False
