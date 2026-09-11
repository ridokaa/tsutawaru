"""Tests for VAD gating, pre-roll, hangover, forced flush, and Silero integration."""
from __future__ import annotations

from unittest.mock import MagicMock
import numpy as np
import pytest

from tsutawaru.config import VadCfg
from tsutawaru.audio.vad import SileroVADGate, WIN
from tsutawaru.pipeline.queues import utt_q, reset_all


@pytest.fixture(autouse=True)
def clean_queues():
    reset_all()
    yield
    reset_all()


def test_vad_prob_accepts_numpy():
    """FIX 3: Verify that _prob wraps numpy in torch tensor and does not raise AttributeError."""
    cfg = VadCfg()
    gate = SileroVADGate(cfg)
    window = np.zeros(WIN, dtype=np.float32)
    prob = gate._prob(window)
    assert isinstance(prob, float)
    assert 0.0 <= prob <= 1.0


def test_vad_reset_states_called_on_flush():
    """FIX 4: reset_states() must be called on every flush to prevent RNN state drift."""
    cfg = VadCfg(hangover_ms=64, min_utterance_ms=100)
    gate = SileroVADGate(cfg)
    gate.model.reset_states = MagicMock(wraps=gate.model.reset_states)

    # Mock _prob to simulate speech then silence
    calls = 0
    def mock_prob(w):
        nonlocal calls
        calls += 1
        return 0.9 if calls <= 10 else 0.0

    gate._prob = mock_prob

    # Feed 15 frames of 512 samples
    for _ in range(15):
        gate.feed(np.ones(WIN, dtype=np.float32) * 0.1)

    assert gate.model.reset_states.called
    assert utt_q.qsize() == 1


def test_vad_forced_flush_on_long_audio():
    """FIX 1 & FIX 2: Forced flush on > max_utterance_ms must not raise ValueError and must preserve carry."""
    cfg = VadCfg(min_utterance_ms=350, max_utterance_ms=2000)  # 2s max utterance
    gate = SileroVADGate(cfg)

    # Simulate continuous speech
    gate._prob = lambda w: 0.95

    # Feed 3 seconds of audio (3 * 16000 / 512 = ~94 windows)
    for i in range(100):
        # Insert a quiet DIP near the end of the 2-second block so argmin has a clear target
        amp = 0.01 if 55 <= i <= 60 else 0.5
        chunk = np.ones(WIN, dtype=np.float32) * amp
        gate.feed(chunk)

    # Should have triggered at least one forced flush into utt_q
    assert utt_q.qsize() >= 1
    utt = utt_q.get_nowait()
    assert utt.forced is True
    assert utt.duration <= 2.1  # Cut within bounds


def test_vad_preroll_recovers_onset():
    """Pre-roll deque should capture frames before speech threshold was crossed."""
    cfg = VadCfg(preroll_ms=320, hangover_ms=64, min_utterance_ms=100)
    gate = SileroVADGate(cfg)

    frame_idx = 0
    def mock_prob(w):
        nonlocal frame_idx
        frame_idx += 1
        return 0.0 if frame_idx <= 5 else 0.9

    gate._prob = mock_prob

    # Feed 5 silence windows then 10 speech windows
    for i in range(15):
        # Tag each window with its index
        gate.feed(np.ones(WIN, dtype=np.float32) * i)

    # Close the utterance with silence
    gate._prob = lambda w: 0.0
    for _ in range(5):
        gate.feed(np.zeros(WIN, dtype=np.float32))

    assert utt_q.qsize() == 1
    utt = utt_q.get_nowait()
    # The audio must include the pre-roll frames
    assert len(utt.audio) > 10 * WIN


def test_vad_filters_short_noise():
    """Short click/tap (< min_utterance_ms) should be discarded without error."""
    cfg = VadCfg(min_utterance_ms=350, hangover_ms=64)
    gate = SileroVADGate(cfg)

    # 2 windows of speech = 64 ms (< 350 ms)
    frame_idx = 0
    def mock_prob(w):
        nonlocal frame_idx
        frame_idx += 1
        return 0.9 if frame_idx <= 2 else 0.0

    gate._prob = mock_prob

    for _ in range(10):
        gate.feed(np.ones(WIN, dtype=np.float32) * 0.1)

    assert utt_q.qsize() == 0
