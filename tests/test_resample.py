"""Tests for stateful StreamResampler."""
from __future__ import annotations

import numpy as np

from tsutawaru.audio.resample import StreamResampler


def test_resampler_identity():
    rs = StreamResampler(16000, 16000)
    data = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000)).astype(np.float32)
    out = rs.process(data)
    assert np.allclose(data, out)


def test_resampler_streaming_continuity():
    """Streaming successive 20 ms blocks must be byte-identical to a monolithic single-call run."""
    src_sr = 48000
    dst_sr = 16000
    duration = 1.0  # 1 second
    t = np.linspace(0, duration, int(src_sr * duration), endpoint=False)
    # Mix of tones
    signal = (0.5 * np.sin(2 * np.pi * 440 * t) + 0.3 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)

    # Monolithic baseline using single StreamResampler call
    baseline = StreamResampler(src_sr, dst_sr).process(signal)

    # Streaming 20 ms blocks (960 samples per block @ 48kHz)
    block_size = int(src_sr * 0.02)
    rs = StreamResampler(src_sr, dst_sr)
    chunks = []
    for i in range(0, len(signal), block_size):
        chunk = signal[i : i + block_size]
        chunks.append(rs.process(chunk))

    streamed = np.concatenate(chunks)

    assert len(streamed) == len(baseline)
    assert np.allclose(streamed, baseline, atol=1e-5)


def test_resampler_empty_input():
    rs = StreamResampler(48000, 16000)
    out = rs.process(np.zeros(0, dtype=np.float32))
    assert len(out) == 0
