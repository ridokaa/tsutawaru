"""Per-process audio capture via ProcTap (Core Audio process taps).

Active when `[audio] backend` resolves to `"process"` — the default on macOS 14.4+
via `"auto"`. Requires macOS 14.4+ and Screen Recording granted to "ProcTap
Helper"; see the capture section of the README.

`ProcessCapture` is a drop-in for `capture.Capture`: same `start()`/`stop()`, same
`native_sr`/`channels` attributes, same real-time callback discipline (the
callback does nothing but hand bytes to `raw_q`). The orchestrator's VAD worker
therefore needs no knowledge of which backend is running.

The format is *not* the same, which is the one thing that cannot be papered over.
ProcTap always emits 48 kHz / 2ch / **float32**, while `capture.Capture` opens
devices as int16. `capture.FrameDecoder` starts with

    np.frombuffer(raw, dtype=np.int16)

so feeding it float32 bytes produces loud noise, not quiet audio — it is not a
degradation, it is garbage. Hence `Float32FrameDecoder` below, selected by the
orchestrator alongside the backend.

Note on resampling cost: the tap's native rate is whatever the app renders at
(44.1 kHz for Chrome), ProcTap resamples that to a fixed 48 kHz, and we then
resample 48 -> 16 kHz. That is one conversion more than the device path needs.
`pip install proc-tap[hq-resample]` puts libsamplerate behind ProcTap's half and
makes it markedly cheaper than the scipy fallback.
"""
from __future__ import annotations

import numpy as np

from tsutawaru.audio.resample import StreamResampler
from tsutawaru.audio.sources import Resolved, Source, resolve
from tsutawaru.logbus import get_logger
from tsutawaru.pipeline.queues import raw_q

log = get_logger(__name__)

TARGET_SR = 16000


class Float32FrameDecoder:
    """float32 interleaved -> mono float32 @ 16 kHz.

    Byte-aligned defensively: a chunk boundary that splits a frame would make
    `np.frombuffer` raise, and one raised exception per chunk in the VAD worker
    is a dead pipeline. The partial frame is carried to the next call instead.
    """

    def __init__(self, native_sr: int, channels: int):
        self.channels = max(1, int(channels))
        self.rs = StreamResampler(native_sr, TARGET_SR)  # ONE instance reused
        self._tail = b""

    def decode(self, raw: bytes) -> np.ndarray:
        if not raw:
            return np.zeros(0, dtype=np.float32)
        buf = self._tail + raw if self._tail else raw
        stride = 4 * self.channels
        n = (len(buf) // stride) * stride
        self._tail = buf[n:]
        if n == 0:
            return np.zeros(0, dtype=np.float32)
        a = np.frombuffer(buf[:n], dtype=np.float32)
        if self.channels > 1:
            a = a.reshape(-1, self.channels).mean(axis=1)
        return self.rs.process(a)


class ProcessCapture:
    """Core Audio process tap on one application's audio, via ProcTap."""

    def __init__(self, source: Source, blocksize_ms: int = 20):
        from proctap import STANDARD_CHANNELS, STANDARD_SAMPLE_RATE

        self.source = source
        self.blocksize_ms = blocksize_ms
        # Fixed by ProcTap's API contract, not negotiable per-source.
        self.native_sr = int(STANDARD_SAMPLE_RATE)
        self.channels = int(STANDARD_CHANNELS)
        self.resolved: Resolved | None = None
        self._tap = None
        # Escalation counter into the source's candidate list. A tap on the wrong
        # process opens cleanly and then delivers nothing, so "which pid" cannot be
        # settled at build time — only by trying.
        self.candidate = 0

    # `Capture` exposes the device it is reading; the orchestrator logs this and
    # the UI shows it. For a process tap the equivalent is app + pid.
    @property
    def device(self) -> str:
        return self.resolved.label if self.resolved else self.source.label

    def start(self) -> "ProcessCapture":
        """Resolve the source's audio pid and open a tap on it.

        Raises SourceUnavailable when the app is not playing — the caller is
        expected to retry, not to treat that as fatal.
        """
        from proctap import ProcessAudioCapture

        # Resolve on every start: pids are not stable, and `candidate` may have
        # been bumped since the last attempt.
        self.resolved = resolve(self.source, self.candidate)
        tap = ProcessAudioCapture(self.resolved.pid, on_data=self._callback)
        tap.start()
        self._tap = tap
        log.info("tapping %s [%s]", self.resolved.label, self.resolved.kind)
        return self

    def advance_candidate(self) -> str:
        """Give up on the current process and prefer the next one.

        Called when a tap has been open and silent for a while. This is the only
        way to recover from tapping the wrong process, because that failure is
        indistinguishable from "the app is quiet" at the Core Audio level: measured
        on this machine, Discord's audio.mojom.AudioService accepts a tap and
        returns 0 bytes while its renderer returns real audio.
        """
        self.candidate += 1
        try:
            nxt = resolve(self.source, self.candidate)
        except Exception:
            return "no other candidate"
        return f"{nxt.kind} (pid {nxt.pid})"

    def _tap_alive(self) -> bool:
        """Whether the tap is still clocking, per ProcTap rather than per our hopes.

        `self._tap is not None` only says we opened one; the helper is a separate
        signed .app launched via `open`, so it can die without us noticing.
        """
        if self._tap is None:
            return False
        probe = getattr(self._tap, "is_running", None)
        if not callable(probe):
            return True
        try:
            return bool(probe())
        except Exception:
            return False

    def needs_retap(self) -> bool:
        """True only when re-opening would actually change something.

        Silence alone is *not* a reason. The first version of this reattached on a
        timer whenever no audio had arrived, which tore down a perfectly good tap
        every few seconds and relaunched the helper app each time — audible as
        nothing, but it would have punched a hole in live capture and it never
        converged while a source sat idle. Reopening only helps in two cases: we
        hold no live tap, or the pid we hold is no longer the one that owns the
        audio (Chromium respawns its audio service with a new pid).
        """
        if not self._tap_alive():
            return True
        from tsutawaru.audio.sources import SourceUnavailable

        try:
            want = resolve(self.source, self.candidate)
        except SourceUnavailable:
            # Nothing better on offer; keep what we have rather than closing the
            # tap and being unable to reopen it.
            return False
        return self.resolved is None or want.pid != self.resolved.pid

    def _callback(self, pcm: bytes, frames: int) -> None:
        # Real-time path: no DSP, no locks, no logging, no allocation beyond the
        # bytes we were handed. Same contract as the sounddevice callback.
        raw_q.put_latest(pcm)

    def stop(self) -> None:
        tap, self._tap = self._tap, None
        if tap is None:
            return
        try:
            tap.close()
        except Exception:
            log.debug("process tap close failed", exc_info=True)
