"""Audio ingestion: non-blocking driver callback and downstream frame decoding.

Callback rules: zero DSP, zero locks, zero logging, zero allocations beyond bytes.
Decoding and stateful resampling are executed in the consumer VAD worker thread.
"""
from __future__ import annotations

import sys
import numpy as np
import sounddevice as sd

from tsutawaru.audio.resample import StreamResampler
from tsutawaru.pipeline.queues import raw_q

TARGET_SR = 16000


class Capture:
    """Real-time safe audio capture stream."""

    def __init__(self, device: int | str, blocksize_ms: int = 20, loopback: bool = False):
        self.device = device
        self.blocksize_ms = blocksize_ms
        self.loopback = loopback and sys.platform == "win32"
        self.pa = None
        self.stream = None

        if self.loopback:
            from tsutawaru.audio.capture_win import open_loopback

            self._open_loopback = open_loopback
            self.native_sr = TARGET_SR
            self.channels = 2
            self.block = int(self.native_sr * blocksize_ms / 1000)
        else:
            info = sd.query_devices(device)
            self.native_sr = int(info["default_samplerate"])
            self.channels = 1 if info["max_input_channels"] == 1 else 2
            self.block = int(self.native_sr * blocksize_ms / 1000)

    def _callback(self, indata, frames, time_info, status):
        # Over/underflow flags are ignored here to avoid blocking real-time audio thread
        raw_q.put_latest(bytes(indata))

    def _win_callback(self, in_data, frame_count, time_info, status):
        raw_q.put_latest(in_data)
        return (None, 0)  # paContinue

    def start(self) -> "Capture":
        if self.loopback:
            self.pa, self.stream, loop_info = self._open_loopback(
                self.block, self._win_callback
            )
            self.native_sr = int(loop_info["defaultSampleRate"])
            self.channels = int(loop_info["maxInputChannels"])
            self.stream.start_stream()
        else:
            self.stream = sd.RawInputStream(
                device=self.device,
                samplerate=self.native_sr,
                channels=self.channels,
                dtype="int16",
                blocksize=self.block,
                latency="low",
                callback=self._callback,
            )
            self.stream.start()
        return self

    def stop(self) -> None:
        if self.stream is not None:
            try:
                if self.loopback:
                    self.stream.stop_stream()
                    self.stream.close()
                    if self.pa is not None:
                        self.pa.terminate()
                else:
                    self.stream.stop()
                    self.stream.close()
            except Exception:
                pass
            finally:
                self.stream = None
                self.pa = None


class FrameDecoder:
    """Decodes raw int16 PCM bytes to float32 mono and resamples to 16 kHz."""

    def __init__(self, native_sr: int, channels: int):
        self.channels = channels
        self.rs = StreamResampler(native_sr, TARGET_SR)  # ONE instance reused

    def decode(self, raw: bytes) -> np.ndarray:
        if not raw:
            return np.zeros(0, dtype=np.float32)
        a = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if self.channels > 1:
            # Multi-channel downmix to mono
            n_samples = len(a) // self.channels
            if n_samples > 0:
                a = a[:n_samples * self.channels].reshape(-1, self.channels).mean(axis=1)
        return self.rs.process(a)
