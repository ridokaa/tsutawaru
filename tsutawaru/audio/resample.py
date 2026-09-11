"""Stateful polyphase rational resampler.

Carries filter delay state and sub-sample phase across consecutive audio blocks
to eliminate block-edge boundary discontinuities and periodic 50 Hz artifacts.
"""
from __future__ import annotations

from math import gcd
import numpy as np
from scipy.signal import firwin, lfilter, lfilter_zi


class StreamResampler:
    """Stateful polyphase-style rational resampler. Continuous across calls."""

    def __init__(self, src_sr: int, dst_sr: int, taps: int = 129):
        self.src_sr = src_sr
        self.dst_sr = dst_sr
        g = gcd(src_sr, dst_sr)
        self.up, self.down = dst_sr // g, src_sr // g
        self.phase = 0

        if self.up == 1 and self.down == 1:
            self.h = None
            self.zi = None
        else:
            # anti-alias at the lower of the two Nyquist limits, in upsampled domain
            cutoff = min(0.95 / max(self.up, self.down), 0.99)
            self.h = (firwin(taps, cutoff) * self.up).astype(np.float64)
            self.zi = lfilter_zi(self.h, 1.0) * 0.0

    def process(self, x: np.ndarray) -> np.ndarray:
        if len(x) == 0:
            return np.zeros(0, dtype=np.float32)
        if self.up == 1 and self.down == 1:
            return x.astype(np.float32, copy=False)

        # upsample by zero-stuffing
        up = np.zeros(len(x) * self.up, dtype=np.float64)
        up[::self.up] = x.astype(np.float64, copy=False)
        y, self.zi = lfilter(self.h, 1.0, up, zi=self.zi)

        # decimate, carrying the sub-sample phase across the call boundary
        idx = np.arange(self.phase, len(y), self.down)
        self.phase = int(idx[-1] + self.down - len(y)) if len(idx) else self.phase
        return y[idx].astype(np.float32)
