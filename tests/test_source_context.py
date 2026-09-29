"""Conversational context follows the capture source.

Context is only true for one voice: on a stream the previous lines really are
the same speaker, on a call they are probably someone else and tsutawaru cannot
tell. Measured on 220 Discord utterances the context made things worse, so the
decision belongs to the source rather than to a global setting.
"""
from __future__ import annotations

import collections
import threading
from types import SimpleNamespace

import pytest

from tsutawaru.audio.sources import SOURCES, single_speaker
from tsutawaru.config import Config
from tsutawaru.translate.local_mlx import PROMPT, LocalTranslator, _prompt


def test_source_registry_marks_one_voice():
    assert single_speaker("youtube") is True
    assert single_speaker("discord") is False
    # An unrecognised source must answer "call": treating a call as a stream
    # asserts the wrong speaker into every line, the costlier direction.
    assert single_speaker("something-new") is False
    assert all(isinstance(s.single_speaker, bool) for s in SOURCES.values())


def _translator(n=3):
    t = LocalTranslator.__new__(LocalTranslator)   # no weights, no MLX
    t.max_context = n
    t.history = collections.deque(maxlen=0)
    t.lock = threading.Lock()
    return t


def test_context_off_by_default_and_toggles():
    t = _translator()
    assert t.history.maxlen == 0
    assert _prompt("ねこ", t.history) == PROMPT.format(src="ねこ")

    t.set_context(True)
    assert t.history.maxlen == 3
    t.history.extend(["A.", "B."])
    assert "A." in _prompt("ねこ", t.history)

    t.set_context(False)
    assert t.history.maxlen == 0
    assert not t.history, "a call inherited the stream's conversation"
    assert _prompt("ねこ", t.history) == PROMPT.format(src="ねこ")


def test_zero_config_disables_it_on_every_source():
    t = _translator(n=0)
    t.set_context(True)
    assert t.history.maxlen == 0


class _Backend:
    def __init__(self):
        self.on = None

    def set_context(self, on):
        self.on = on


def _orch(source):
    from tsutawaru.pipeline.orchestrator import Orchestrator

    o = Orchestrator.__new__(Orchestrator)
    o.cfg = Config()
    o.cfg.audio.source = source
    o.stages = SimpleNamespace(translator=_Backend())
    return o


@pytest.mark.parametrize("source,expected", [("youtube", True), ("discord", False)])
def test_orchestrator_applies_the_source_decision(source, expected):
    o = _orch(source)
    o._apply_context()
    assert o.stages.translator.on is expected


def test_apply_context_is_a_noop_for_a_backend_without_one():
    """The online providers keep no history and must not be asked to."""
    o = _orch("youtube")
    o.stages.translator = object()
    o._apply_context()      # must not raise
