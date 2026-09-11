"""Tests for the silent-stream detector (§6.6)."""
from __future__ import annotations

import logging
import sys
from tsutawaru.audio.devices import silent_stream_help
from tsutawaru.logbus import RateLimitedWarner


def test_silent_stream_help_mentions_both_causes():
    msg = silent_stream_help("BlackHole 2ch", seconds=5.0)
    assert 'No signal from "BlackHole 2ch"' in msg
    if sys.platform == "darwin":
        assert "System output" in msg or "Output routing" in msg
        assert "Microphone permission" in msg
    elif sys.platform == "win32":
        assert "CABLE Input" in msg
        assert "48000 Hz" in msg


def test_rate_limited_warner_suppresses_repetition(caplog):
    logger = logging.getLogger("test.silent")
    warner = RateLimitedWarner(logger)

    with caplog.at_level(logging.WARNING):
        # First warning fires
        assert warner.warn("Stream silent") is True
        # Immediate subsequent warnings suppressed
        assert warner.warn("Stream silent") is False
        assert warner.warn("Stream silent") is False

        # Signal returns: rearm
        warner.rearm()

        # Next silence warning fires again
        assert warner.warn("Stream silent") is True

    assert caplog.text.count("Stream silent") == 2
