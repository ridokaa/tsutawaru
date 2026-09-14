"""Voice Activity Detection (VAD) gating engine.

Turns continuous audio stream into utterance boundaries without cutting words.
Silero VAD (ONNX/Torch) is the primary engine, with WebRTC VAD as a fallback.
"""
from __future__ import annotations

import collections
import itertools
import sys
import time
import numpy as np
import torch
from silero_vad import load_silero_vad

from tsutawaru.config import VadCfg
from tsutawaru.models import Utterance
from tsutawaru.pipeline.queues import utt_q

#: Groups a provisional utterance with the final one that replaces it. Its own
#: counter rather than models._ids, which Utterance and Segment already share —
#: a line id colliding with a segment id would be a confusing coincidence to
#: debug.
_line_ids = itertools.count(1)

WIN = 512  # Silero requires exactly 512 samples @ 16 kHz (32 ms)
TAIL_WINDOWS = 15  # 15 * 512 = 7680 samples (480 ms)
TAIL = TAIL_WINDOWS * WIN


class SileroVADGate:
    """Silero VAD state machine."""

    def __init__(self, cfg: VadCfg):
        self.cfg = cfg
        self.model = load_silero_vad(onnx=True)
        self.thr = cfg.threshold
        self.win = WIN
        self.preroll = collections.deque(maxlen=max(1, round(cfg.preroll_ms / 32)))
        self.hangover_frames = max(1, round(cfg.hangover_ms / 32))
        self.min_len = cfg.min_utterance_ms * 16
        self.max_len = cfg.max_utterance_ms * 16
        # 0 disables provisional output entirely, which is the old behaviour.
        self.prov_len = cfg.provisional_after_ms * 16
        self.buf: list[np.ndarray] = []
        self.silence_run = 0
        self.in_speech = False
        self._line_id = 0
        self._prov_sent = 0  # samples already shown provisionally, 0 = none
        self._resid = np.zeros(0, dtype=np.float32)

    def reset(self) -> None:
        """Drop all in-flight state without emitting the partial utterance.

        Called when the audio source changes underneath us. Everything buffered
        belongs to the *previous* source, so flushing it would emit one utterance
        containing two different apps — and `_resid` would additionally splice
        the two waveforms mid-window. Discarding is the only correct answer; the
        RNN state has to go too, or the first window of the new source is scored
        with the old one's context.
        """
        self.buf = []
        self.preroll.clear()
        self.silence_run = 0
        self.in_speech = False
        self._prov_sent = 0
        self._resid = np.zeros(0, dtype=np.float32)
        try:
            self.model.reset_states()
        except Exception:  # pragma: no cover - defensive
            pass

    def feed(self, chunk: np.ndarray) -> None:
        """Process arbitrary-length float32 @ 16 kHz audio chunk."""
        if len(chunk) == 0:
            return
        data = np.concatenate([self._resid, chunk])
        n = (len(data) // self.win) * self.win
        self._resid = data[n:]
        for i in range(0, n, self.win):
            self._window(data[i : i + self.win])

    def _prob(self, w: np.ndarray) -> float:
        # Silero ONNX wrapper requires a torch Tensor (validates via x.dim())
        return float(self.model(torch.from_numpy(w), 16000))

    def _window(self, w: np.ndarray) -> None:
        speech = self._prob(w) >= self.thr
        if not self.in_speech:
            self.preroll.append(w)
            if speech:
                self.in_speech = True
                self.buf = list(self.preroll)  # recovers the clipped onset
                self.preroll.clear()
                self.silence_run = 0
                self._line_id = next(_line_ids)
                self._prov_sent = 0
        else:
            self.buf.append(w)
            self.silence_run = 0 if speech else self.silence_run + 1
            total = sum(len(b) for b in self.buf)
            if self.silence_run >= self.hangover_frames:
                self._flush()
            elif total >= self.max_len:
                self._flush(forced=True)
            elif self.prov_len and total - self._prov_sent >= self.prov_len:
                self._provisional(total)

    def _provisional(self, total: int) -> None:
        """Emit what has been said so far, without ending the utterance.

        A prefix, not a chunk: the buffer is left intact and the final flush
        still transcribes the whole run, so nothing is cut mid-phrase. Measured
        on the recorded corpora, transcribing a long utterance whole scores
        within a few percent of transcribing it in halves, so re-doing the work
        costs accuracy nothing — what it buys is the first words appearing while
        the speaker is still talking.

        Re-armed by `_prov_sent` rather than a flag, so a long run of speech
        keeps refreshing instead of freezing on its first 2.5 seconds.

        Skipped whenever anything is already waiting for STT. `utt_q` is
        drop-oldest, so an unconditional provisional could evict a *final*
        utterance — trading speech that exists nowhere else for a preview of
        speech that is about to arrive anyway. It is the luxury in the queue and
        so it is the first thing to go.
        """
        if utt_q.qsize():
            self._prov_sent = total  # re-arm anyway; do not retry every 32 ms
            return
        self._prov_sent = total
        utt_q.put_latest(
            Utterance.new(
                audio=np.concatenate(self.buf),
                t_start=time.monotonic() - total / 16000,
                t_end=time.monotonic(),
                line_id=self._line_id,
                provisional=True,
            )
        )

    def _flush(self, forced: bool = False) -> None:
        audio = np.concatenate(self.buf) if self.buf else np.zeros(0, np.float32)
        carry = np.zeros(0, np.float32)

        if forced and len(audio) > TAIL:
            # Search for the lowest-energy window in the trailing 480 ms
            tail = audio[-TAIL:]
            frames = tail.reshape(TAIL_WINDOWS, WIN)
            q = int(np.argmin(np.abs(frames).mean(axis=1)))
            cut = len(audio) - TAIL + (q + 1) * WIN
            audio, carry = audio[:cut], audio[cut:]

        # Reset RNN state on every flush to prevent LSTM context leakage
        self.model.reset_states()

        # Restore carry before min_len check so trailing audio is not lost
        line_id, self._prov_sent = self._line_id, 0
        if len(carry):
            self.buf, self.in_speech, self.silence_run = [carry], True, 0
            # The remainder is a different line: the one just flushed has been
            # shown and must not be overwritten by what comes after it.
            self._line_id = next(_line_ids)
        else:
            self.buf, self.in_speech, self.silence_run = [], False, 0

        if len(audio) < self.min_len:
            return  # cough, click, keyboard tap

        now = time.monotonic()
        utt_q.put_latest(
            Utterance.new(
                audio=audio,
                t_start=now - len(audio) / 16000,
                t_end=now,
                forced=forced,
                line_id=line_id,
            )
        )


class WebRTCVADGate:
    """WebRTC VAD fallback."""

    def __init__(self, cfg: VadCfg):
        if sys.version_info >= (3, 14):
            raise RuntimeError(
                "WebRTC VAD is unavailable on Python 3.14 (see plan §3.1). "
                "Use backend='silero' or run on Python 3.11."
            )
        try:
            import webrtcvad

            self.vad = webrtcvad.Vad(cfg.aggressiveness)
        except ImportError:
            raise RuntimeError("webrtcvad is not installed")
        self.cfg = cfg
        self.win = 320  # 20 ms @ 16 kHz
        self.preroll = collections.deque(maxlen=max(1, round(cfg.preroll_ms / 20)))
        self.hangover_frames = max(1, round(cfg.hangover_ms / 20))
        self.min_len = cfg.min_utterance_ms * 16
        self.max_len = cfg.max_utterance_ms * 16
        self.buf: list[np.ndarray] = []
        self.silence_run = 0
        self.in_speech = False
        self._resid = np.zeros(0, dtype=np.float32)

    def reset(self) -> None:
        """Drop in-flight state on a source change. See SileroVADGate.reset."""
        self.buf = []
        self.preroll.clear()
        self.silence_run = 0
        self.in_speech = False
        self._resid = np.zeros(0, dtype=np.float32)

    def feed(self, chunk: np.ndarray) -> None:
        if len(chunk) == 0:
            return
        data = np.concatenate([self._resid, chunk])
        n = (len(data) // self.win) * self.win
        self._resid = data[n:]
        for i in range(0, n, self.win):
            w = data[i : i + self.win]
            pcm16 = (np.clip(w, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
            speech = self.vad.is_speech(pcm16, 16000)
            if not self.in_speech:
                self.preroll.append(w)
                if speech:
                    self.in_speech = True
                    self.buf = list(self.preroll)
                    self.preroll.clear()
                    self.silence_run = 0
            else:
                self.buf.append(w)
                self.silence_run = 0 if speech else self.silence_run + 1
                total = sum(len(b) for b in self.buf)
                if self.silence_run >= self.hangover_frames:
                    self._flush()
                elif total >= self.max_len:
                    self._flush(forced=True)

    def _flush(self, forced: bool = False) -> None:
        audio = np.concatenate(self.buf) if self.buf else np.zeros(0, np.float32)
        self.buf, self.in_speech, self.silence_run = [], False, 0
        if len(audio) < self.min_len:
            return
        now = time.monotonic()
        utt_q.put_latest(
            Utterance.new(
                audio=audio,
                t_start=now - len(audio) / 16000,
                t_end=now,
                forced=forced,
            )
        )


def VADGate(cfg: VadCfg):
    if cfg.backend == "webrtc":
        return WebRTCVADGate(cfg)
    return SileroVADGate(cfg)
