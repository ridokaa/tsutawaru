"""The gate lowers its threshold only where speech is actually being masked.

Driven through `_retune` with synthetic probabilities rather than audio: the
decision is a function of the probability stream, and a WAV would make this a
slow test of Silero instead of a fast test of the rule.
"""
from __future__ import annotations

from tsutawaru.audio.vad import (AMBIG_HI, AMBIG_LO, AMBIG_OFF, AMBIG_ON,
                                 AMBIG_WINDOWS, SileroVADGate)
from tsutawaru.config import VadCfg


def _gate() -> SileroVADGate:
    """A gate without Silero: `_retune` never touches the model."""
    g = SileroVADGate.__new__(SileroVADGate)
    import collections
    g.cfg = VadCfg()
    g.thr = g.cfg.threshold
    g._ambig = collections.deque(maxlen=AMBIG_WINDOWS)
    g._noisy = False
    return g


def _feed(g, ambiguous_rate: float, windows: int = AMBIG_WINDOWS) -> None:
    mid = (AMBIG_LO + AMBIG_HI) / 2
    for i in range(windows):
        g._retune(mid if (i % 1000) / 1000 < ambiguous_rate else 0.99)


def test_a_clean_source_keeps_the_configured_threshold():
    g = _gate()
    _feed(g, 0.06)                       # measured: quiet clips sit at 4.5-6.5%
    assert not g._noisy
    assert g.thr == VadCfg().threshold


def test_a_masked_source_drops_to_the_noisy_threshold():
    g = _gate()
    _feed(g, 0.27)                       # measured: the loud game clip
    assert g._noisy
    assert g.thr == VadCfg().threshold_noisy


def test_it_waits_for_a_full_window_before_deciding():
    """A verdict off three seconds of audio would flap on the first pause."""
    g = _gate()
    _feed(g, 0.9, windows=AMBIG_WINDOWS - 1)
    assert g.thr == VadCfg().threshold, "decided before it had a minute"


def test_hysteresis_holds_through_a_quiet_stretch():
    """A game that goes briefly quiet must not bounce the threshold."""
    g = _gate()
    _feed(g, 0.27)
    assert g._noisy
    _feed(g, (AMBIG_ON + AMBIG_OFF) / 2)   # between the two edges
    assert g._noisy, "dropped out on a rate that only the ON edge excludes"
    _feed(g, 0.02)
    assert not g._noisy and g.thr == VadCfg().threshold


def test_a_source_change_forgets_the_previous_background():
    """reset() runs on a source switch: a game stream's verdict must not be
    carried into the call that follows it."""
    g = _gate()
    _feed(g, 0.27)
    assert g._noisy
    g.buf, g.preroll, g.silence_run = [], __import__("collections").deque(), 0
    g.in_speech, g._prov_sent = False, 0
    import numpy as np
    g._resid = np.zeros(0, dtype=np.float32)
    g.model = type("M", (), {"reset_states": staticmethod(lambda: None)})()
    g.reset()
    assert not g._noisy and g.thr == VadCfg().threshold and not g._ambig
