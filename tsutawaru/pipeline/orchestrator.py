"""Pipeline lifecycle: builds the stages, wires the threads, owns shutdown.

Stage construction is lazy and each stage is optional at import time. Stages
that have not been written yet raise `StageNotImplemented`, which `run.py`
renders as "build this next, see plan §X" rather than an ImportError traceback.
That makes a partially-built tree runnable and self-describing.

Threading model (plan §5.2):
    audio callback -> raw_q          (OS-owned, real-time)
    vad thread     -> reads raw_q,   writes utt_q
    stt thread     -> reads utt_q,   writes seg_q     (exactly one)
    nlp thread     -> reads seg_q,   writes trans_q
    translate pool -> 1 sentence worker + N gloss workers
    main thread    -> UI
"""
from __future__ import annotations

import collections
import queue
import statistics
import threading
import time
from dataclasses import dataclass, replace as dc_replace

import numpy as np

from tsutawaru.config import Config
from tsutawaru.logbus import RateLimitedWarner, get_logger, metrics
from tsutawaru.models import Utterance
from tsutawaru.pipeline import queues
from tsutawaru.pipeline.queues import (compare_q, raw_q, seg_q, trans_q,
                                       ui_q, utt_q)

log = get_logger(__name__)

SILENT_AFTER_S = 5.0  # §6.6 silent-stream detector: warn
RESTART_AFTER_S = 25.0  # …and if still dead, rebuild the stream
RESTART_BACKOFF_S = 60.0  # minimum gap between restart attempts
JOIN_TIMEOUT_S = 3.0
DRAIN_TIMEOUT_S = 2.0
# Extra time at shutdown for in-flight sentence translations to land in the
# session log. Only spent when --record/--export-md asked for one.
RECORD_GRACE_S = 2.0

# Per-process capture (backend="process") needs its own timings, because for that
# backend "no audio" is the normal idle state rather than a fault. A Chromium app
# only owns a Core Audio process object while it is actually playing, so the tap
# legitimately has nothing to deliver — and the pid to tap changes each time the
# audio service is respawned. Re-resolving often is therefore the mechanism that
# picks up "the user just started a video", not an error path.
PROC_GRACE_S = 1.5  # don't even look until this much quiet has passed
PROC_RETAP_EVERY_S = 3.0  # how often to re-resolve the source's audio pid
PROC_SILENT_WARN_S = 20.0 # only then is silence worth complaining about
# A tap on the wrong process opens cleanly and returns nothing, which is
# indistinguishable from an idle app. After this long we stop believing the current
# guess and try the source's next candidate process.
#
# 12.0 was not generous enough, and the failure was measured rather than guessed:
# in a 50-minute six-person Discord call this fired on four ordinary conversational
# pauses, abandoning the correct renderer each time and cycling every Discord
# process before wrapping back to it — 108/114/117/119 s of audio lost per event,
# ~16% of the session. Two changes came out of that:
#   * escalate only while the current candidate is *unconfirmed* (see
#     _proc_confirmed) — a process that has produced real audio is never wrong,
#     so a pause of any length is just a pause;
#   * 45 s rather than 12 s, so the startup search still converges quickly but no
#     plausible gap in speech reaches it before the confirmation check does.
PROC_ESCALATE_S = 45.0


class StageNotImplemented(RuntimeError):
    """A pipeline stage is unavailable: not written yet, or its deps are missing."""

    def __init__(self, module: str, section: str, missing_dep: str | None = None):
        self.module, self.section, self.missing_dep = module, section, missing_dep
        if missing_dep:
            super().__init__(
                f"{module} needs the {missing_dep!r} package "
                f"(pip install {missing_dep}) — see plan {section}"
            )
        else:
            super().__init__(f"{module} is not implemented yet (see plan {section})")


# pip name != import name for a few of our dependencies.
_PIP_NAME = {
    "faster_whisper": "faster-whisper",
    "mlx_whisper": "mlx-whisper",
    "silero_vad": "silero-vad[onnx-cpu]",
    "deep_translator": "deep-translator",
    "pyaudiowpatch": "PyAudioWPatch",
    "sounddevice": "sounddevice",
    "janome": "janome",
    "pykakasi": "pykakasi",
    "websockets": "websockets",
    "PyQt6": "PyQt6",
}


def _require(module_path: str, attr: str, section: str):
    """Import a stage, or raise StageNotImplemented naming the plan section.

    Two distinct failures land here and the message must tell them apart:
      * our module does not exist yet          -> "not implemented, see §X"
      * our module exists but a dep is absent  -> "pip install faster-whisper"
    Rev 1 of this helper re-raised the second case, which surfaced as a raw
    ModuleNotFoundError traceback instead of an actionable message.
    """
    import importlib

    try:
        mod = importlib.import_module(module_path)
    except ModuleNotFoundError as e:
        name = str(e.name or "")
        # Derived, not spelled out: this decides "one of our stages is not built
        # yet" versus "a third-party package is missing", and a hardcoded project
        # name silently stops matching the moment the package is renamed — after
        # which every unbuilt stage is misreported as a missing pip install.
        root = __name__.split(".")[0]
        ours = (name == module_path or module_path.startswith(name + ".")
                or name.startswith(root))
        if ours:
            raise StageNotImplemented(module_path, section) from None
        raise StageNotImplemented(
            module_path, section, _PIP_NAME.get(name, name)
        ) from None
    except ImportError as e:
        # e.g. a native library present but unloadable (wrong arch, missing dylib)
        raise StageNotImplemented(module_path, section, str(e)[:80]) from None
    obj = getattr(mod, attr, None)
    if obj is None:
        raise StageNotImplemented(f"{module_path}.{attr}", section)
    return obj


@dataclass
class Stages:
    """What actually got built. Missing stages are None."""

    capture: object | None = None
    decoder: object | None = None
    vad: object | None = None
    stt: object | None = None
    stt_compare: object | None = None  # optional A/B model, None when off
    tokenizer: object | None = None
    translator: object | None = None
    pool: object | None = None
    sink: object | None = None
    missing: list[StageNotImplemented] = None  # type: ignore[assignment]


class Orchestrator:
    def __init__(self, cfg: Config, *, dump_utterances: str | None = None,
                 record: str | None = None, export_md: str | None = None):
        self.cfg = cfg
        self.stop_evt = threading.Event()
        self._alt_tokenizer = None  # comparison lane's own, built on its thread
        self._alt_romanize = None   # resolved alongside it
        self.threads: list[threading.Thread] = []
        self.stages = Stages(missing=[])
        self.dump_utterances = dump_utterances
        from tsutawaru.pipeline.recorder import SessionRecorder

        self.recorder = SessionRecorder(
            jsonl=record,
            markdown=export_md,
            meta={
                "source": cfg.audio.source if cfg.audio.effective_backend == "process"
                else cfg.audio.device,
                "model": cfg.stt.model,
                "provider": cfg.translate.provider,
            },
        )
        self._silent_warner = RateLimitedWarner(log)
        self._last_nonzero = time.monotonic()
        self._device_name = "<unset>"
        self._last_restart = 0.0
        self._restarts = 0
        # Resolved once, here, and read from `self._backend` everywhere below.
        # cfg.audio.backend may be "auto", which matches neither branch of any
        # `== "process"` test — an unresolved value silently behaves as "device"
        # at every one of them.
        self._backend = cfg.audio.effective_backend
        if cfg.audio.backend == "auto":
            # Announced rather than left implicit. The two backends fail in
            # completely different ways — a silent device path means check your
            # loopback routing, a silent tap means check the app's output device
            # or the Screen Recording grant — so "which one am I on" is the first
            # question any silence raises, and the log has to answer it.
            log.info("[audio] backend 'auto' -> %r", self._backend)
        self._last_retap = 0.0
        self._retap_busy = threading.Event()
        self._switch_lock = threading.Lock()
        # Packets-since-tap, not amplitude. "no packets" (app idle) and "packets of
        # zeros" (no Screen Recording grant) are different faults with different
        # fixes, and this counter is the only way to tell them apart — Core Audio
        # reports no error for either.
        self._frames_since_tap = 0
        # Set once the *current* candidate has delivered real (nonzero) audio.
        # Escalation exists only to find the right process; once one has proven
        # itself there is never a reason to walk away from it, and doing so cost
        # ~2 minutes of a live session four times over (see PROC_ESCALATE_S).
        # Cleared wherever the tap is reopened or the source changes.
        self._proc_confirmed = False
        # Recent utterance peaks, for the relative amplitude test in §6.1. A
        # window rather than the whole session so it tracks a source switch or a
        # volume change instead of averaging across both.
        self._peaks: collections.deque[float] = collections.deque(maxlen=100)

    # ---------------------------------------------------------------- build

    def build(self) -> Stages:
        """Construct every stage that exists. Never raises for missing stages."""
        s = self.stages
        for label, fn in (
            ("capture", self._build_capture),
            ("vad", self._build_vad),
            ("stt", self._build_stt),
            ("nlp", self._build_nlp),
            ("translate", self._build_translate),
            ("ui", self._build_ui),
        ):
            try:
                fn()
            except StageNotImplemented as e:
                log.debug("stage %s unavailable: %s", label, e)
                s.missing.append(e)
        return s

    def _build_capture(self):
        if self._backend == "process":
            return self._build_capture_process()
        return self._build_capture_device()

    def _build_capture_process(self):
        """Per-process capture via a Core Audio process tap (ProcTap).

        Deliberately does not fail when the source is not playing: the tap is
        opened lazily by the retap loop instead. Refusing to build here would
        mean "start tsutawaru before opening Discord" could never work, and
        that is the normal order of operations.
        """
        from tsutawaru.audio import sources

        ProcessCapture = _require(
            "tsutawaru.audio.capture_proc", "ProcessCapture", "§2"
        )
        Float32FrameDecoder = _require(
            "tsutawaru.audio.capture_proc", "Float32FrameDecoder", "§2"
        )

        src = sources.get(self.cfg.audio.source)
        cap = ProcessCapture(src, blocksize_ms=self.cfg.audio.blocksize_ms)
        self.stages.capture = cap
        self.stages.decoder = Float32FrameDecoder(cap.native_sr, cap.channels)
        self._device_name = src.label

    def _build_capture_device(self):
        from tsutawaru.audio.devices import resolve

        idx = resolve(self.cfg.audio.device)
        Capture = _require("tsutawaru.audio.capture", "Capture", "§4 Phase 1c")
        FrameDecoder = _require("tsutawaru.audio.capture", "FrameDecoder", "§4 Phase 1c")

        import sounddevice as sd

        info = sd.query_devices(idx)
        self._device_name = info["name"]
        native_sr = int(info["default_samplerate"])
        channels = min(2, max(1, int(info["max_input_channels"])))

        self.stages.capture = Capture(idx, blocksize_ms=self.cfg.audio.blocksize_ms)
        self.stages.decoder = FrameDecoder(native_sr, channels)

    def _build_vad(self):
        VADGate = _require("tsutawaru.audio.vad", "VADGate", "§4 Phase 1d")
        self.stages.vad = VADGate(self.cfg.vad)

    def _build_stt(self):
        build_engine = _require("tsutawaru.stt.factory", "build_engine", "§4 Phase 2d")
        self.stages.stt = build_engine(self.cfg.stt)
        if self.cfg.stt.compare_model:
            # Safe to construct here even for MLX: no engine loads weights in
            # its constructor, so each lane's load lands on its own worker
            # thread. See qwen_mlx_engine for why that is load-bearing.
            alt = dc_replace(self.cfg.stt, model=self.cfg.stt.compare_model)
            self.stages.stt_compare = build_engine(alt)
            log.info("comparing %s against %s",
                     self.cfg.stt.model, self.cfg.stt.compare_model)

    def _build_nlp(self):
        build_tokenizer = _require("tsutawaru.nlp.tokenizer", "build_tokenizer", "§4 Phase 3a")
        self.stages.tokenizer = build_tokenizer(self.cfg.nlp)

    def _build_translate(self):
        if self.cfg.translate.provider == "none":
            return
        GlossCache = _require("tsutawaru.translate.cache", "GlossCache", "§4 Phase 4a")
        TranslationPool = _require("tsutawaru.translate.pool", "TranslationPool", "§4 Phase 4d")
        if self.cfg.translate.provider == "local":
            backend_cls = _require("tsutawaru.translate.local_mlx", "LocalTranslator", "§4 Phase 4c")
        else:
            backend_cls = _require("tsutawaru.translate.online", "OnlineTranslator", "§4 Phase 4c")
        backend = backend_cls(self.cfg.translate)
        cache = GlossCache(self.cfg.translate.lru_size, self.cfg.translate.persist_cache)
        self.stages.translator = backend
        self.stages.pool = TranslationPool(backend, cache, self.cfg.translate)

    def _build_ui(self):
        ws_sink = None
        if self.cfg.ui.ws_port:
            try:
                from tsutawaru.ui.ws_server import WSSink

                ws_sink = WSSink(self.cfg.ui.ws_port)
                ws_sink.start()
            except Exception as e:
                log.warning("WebSocket server could not be started: %s", e)

        # `pool` is built before the UI (see build() ordering), so we can tell
        # the sink whether an ("english", …) event will ever arrive.
        kw = dict(ws_sink=ws_sink, expect_translation=self.stages.pool is not None)

        # Window-only extras, kept out of `kw`: the console sink receives the same
        # dict and would reject them.
        win_kw: dict = {}
        # The source switcher only appears when it can actually do something —
        # on the device backend there is nothing to switch between.
        if self._backend == "process":
            from tsutawaru.audio.sources import SOURCES

            win_kw["sources"] = [(s.key, s.label) for s in SOURCES.values()]
            win_kw["current_source"] = self.cfg.audio.source
            win_kw["on_source"] = self.switch_source

        # Lets the window fetch the breakdown for one line on demand, for lines
        # that came through with the tier switched off. Window-only, so it stays
        # out of `kw`, which the console sink also receives.
        if self.stages.pool is not None:
            win_kw["on_glosses"] = self.stages.pool.submit_glosses

        want = self.cfg.ui.sink
        if want == "window":
            try:
                WindowSink = _require("tsutawaru.ui.window_qt", "WindowSink", "§4 Phase 6a")
                self.stages.sink = WindowSink(
                    self.cfg.ui, device_name=self._device_name, **kw, **win_kw
                )
                return
            except (StageNotImplemented, ImportError) as e:
                # A missing GUI toolkit must not cost you the transcript.
                log.warning("window UI unavailable (%s) — falling back to console", e)

        ConsoleSink = _require("tsutawaru.ui.console", "ConsoleSink", "§4 Phase 5b")
        self.stages.sink = ConsoleSink(self.cfg.ui, **kw)

    # --------------------------------------------------------------- workers

    def _vad_worker(self):
        """raw_q -> decode + resample -> VAD -> utt_q. Owns both stateful objects.

        `decoder` is re-read from `self.stages` each iteration rather than bound
        once: a capture restart replaces it (the device may return at a
        different sample rate), and a stale local reference would keep feeding
        the old resampler forever.
        """
        gate = self.stages.vad
        while not self.stop_evt.is_set():
            decoder = self.stages.decoder
            try:
                raw = raw_q.get(timeout=0.2)
            except queue.Empty:
                self._check_silence()
                continue
            self._frames_since_tap += 1
            t0 = time.perf_counter()
            try:
                chunk = decoder.decode(raw)
            except Exception:
                log.exception("frame decode failed")
                continue
            if chunk.size and float(np.abs(chunk).max()) > 0.0:
                self._last_nonzero = time.monotonic()
                self._silent_warner.rearm()
                # Real audio from this candidate proves the guess was right.
                self._proc_confirmed = True
            else:
                self._check_silence()
            try:
                gate.feed(chunk)
            except Exception:
                log.exception("vad failed")
            metrics.record("vad", (time.perf_counter() - t0) * 1000)

    def _check_silence(self):
        """§6.6: warn once when the stream is open but delivering only zeros.

        This runs on every silent frame (~50/s), so it must be cheap when it is
        not going to emit. The message is passed as a callable: building it
        shells out to `system_profiler` (~200 ms) and doing that eagerly here
        starved this thread badly enough to overflow raw_q.
        """
        silent_for = time.monotonic() - self._last_nonzero
        if self._backend == "process":
            self._check_silence_process(silent_for)
            return
        if silent_for <= SILENT_AFTER_S:
            return

        # A CoreAudio stream can stop delivering frames without raising and
        # without closing — the callback simply stops firing. §6.6 only covered
        # the noisy case (device unplugged -> PortAudioError); this is the quiet
        # one, and it is unrecoverable without intervention. Observed in the
        # wild: a stream left open on BlackHole went permanently silent while a
        # second process on the same device captured normally.
        if silent_for > RESTART_AFTER_S:
            self._try_restart_capture()

        if not self._silent_warner.arm_once():
            return

        # Even the single legitimate emission costs ~200 ms of subprocess time,
        # which is a visible stall in a 20 ms real-time loop. Emit off-thread so
        # no `system_profiler` call ever runs on the audio path.
        def _emit():
            from tsutawaru.audio.devices import silent_stream_help

            log.warning("\n" + silent_stream_help(self._device_name, SILENT_AFTER_S))

        threading.Thread(target=_emit, name="silent-warn", daemon=True).start()

    def _check_silence_process(self, silent_for: float) -> None:
        """Silence handling for the per-process backend.

        Distinct from the device path on purpose. For a device, an all-zero stream
        means something is broken. For a process tap it usually means the app is
        idle — Chromium only holds a Core Audio process object while it plays, so
        there is frequently nothing to tap at all. Treating that as a fault would
        fire the §6.6 diagnostic through every quiet moment of a call and restart
        a perfectly healthy tap on a 60 s loop.

        So the two behaviours are inverted relative to the device path: re-attach
        *eagerly* (that is how a newly-spawned audio service is discovered) and
        complain *late*.
        """
        if silent_for <= PROC_GRACE_S:
            return

        now = time.monotonic()
        if now - self._last_retap >= PROC_RETAP_EVERY_S:
            self._last_retap = now
            # A tap that has been open and silent this long is probably on the
            # wrong process, not on a quiet one. Escalating *before* the retap
            # means the reopen lands on the next candidate.
            cap = self.stages.capture
            if (silent_for > PROC_ESCALATE_S and not self._proc_confirmed
                    and cap is not None
                    and hasattr(cap, "advance_candidate")):
                nxt = cap.advance_candidate()
                log.info(
                    "no audio from %s after %.0fs — trying %s",
                    cap.device, silent_for, nxt,
                )
                self._retap(force=True)
            else:
                self._retap()

        if silent_for > PROC_SILENT_WARN_S and self._silent_warner.arm_once():
            src = getattr(self.stages.capture, "source", None)
            if src is None:
                return

            saw_frames = self._frames_since_tap > 0

            def _emit():
                from tsutawaru.audio.sources import source_help

                log.warning("\n" + source_help(src, silent_for, got_frames=saw_frames))

            threading.Thread(target=_emit, name="source-warn", daemon=True).start()

    def _retap(self, force: bool = False) -> None:
        """Re-resolve the source pid and reopen the tap, off the audio thread.

        Resolution walks every process with its exe and cmdline, which costs tens
        of milliseconds — the same reason `silent_stream_help` was moved off this
        thread. The busy flag matters because retaps are attempted every few
        seconds indefinitely while a source is idle, and overlapping ones would
        race to assign `self._tap`.
        """
        if self._retap_busy.is_set() or self.stages.capture is None:
            return
        self._retap_busy.set()

        def _work():
            try:
                from tsutawaru.audio.sources import SourceUnavailable

                # A switch in progress is already doing this, for a different
                # source. Yielding rather than waiting avoids reopening the tap the
                # user just navigated away from.
                if not self._switch_lock.acquire(blocking=False):
                    return
                try:
                    cap = self.stages.capture
                    if cap is None:
                        return
                    # Silence is not itself a reason to reopen. Asking the capture
                    # first is what stops this loop from cycling a healthy tap
                    # every few seconds while a source is merely idle. `force`
                    # bypasses it after an escalation, where the whole point is to
                    # abandon a tap the capture still considers healthy.
                    if not force and not cap.needs_retap():
                        return
                    try:
                        cap.stop()  # a stale tap on a dead pid never recovers
                    except Exception:
                        log.debug("stop before retap failed", exc_info=True)
                    try:
                        cap.start()
                    except SourceUnavailable as e:
                        log.debug("retap: %s", e)
                        return
                    except Exception as e:
                        log.debug("retap failed: %s", e)
                        return
                    self._last_nonzero = time.monotonic()
                    self._frames_since_tap = 0
                    self._proc_confirmed = False
                    log.info("re-attached to %s", cap.device)
                finally:
                    self._switch_lock.release()
            finally:
                self._retap_busy.clear()

        threading.Thread(target=_work, name="retap", daemon=True).start()

    def _stt_worker(self):
        engine = self.stages.stt
        try:
            engine.warmup()
        except Exception:
            # Stop, don't carry on. An engine that cannot transcribe one second
            # of silence fails identically on every utterance, and a session
            # that captures audio and emits nothing looks like a quiet room —
            # the 2026-09-04 Qwen run cost seven minutes that way.
            log.exception(
                "stt warmup failed (%s) — stopping, nothing this run captures "
                "could be transcribed. Traceback above is the real fault; "
                "--model kotoba to keep going without it.",
                type(engine).__name__,
            )
            self.stop_evt.set()
            return
        while not self.stop_evt.is_set():
            try:
                utt: Utterance = utt_q.get(timeout=0.2)
            except queue.Empty:
                continue
            if self.dump_utterances:
                self._dump(utt)
            t0 = time.perf_counter()
            try:
                res = engine.transcribe(utt.audio)
            except Exception:
                log.exception("stt failed")
                continue
            metrics.record("stt", (time.perf_counter() - t0) * 1000)
            self._emit_segment(utt, res)

    def _stt_compare_worker(self):
        """Second ASR model, same audio, beside the first.

        Not symmetric with `_stt_worker`: a failure here disables the comparison
        and lets the session run on, because taking down a working transcript to
        report a broken diagnostic is the wrong trade. Never touches
        `seg.original`, `seg.english` or the tokens — the primary transcript
        must be exactly what it would have been with comparison off.
        """
        engine = self.stages.stt_compare
        try:
            engine.warmup()
        except Exception:
            log.exception(
                "comparison model %r failed to warm up — continuing with %r "
                "alone; the transcript is unaffected",
                self.cfg.stt.compare_model, self.cfg.stt.model,
            )
            self.stages.stt_compare = None
            return
        while not self.stop_evt.is_set():
            try:
                utt, seg = compare_q.get(timeout=0.2)
            except queue.Empty:
                continue
            t0 = time.perf_counter()
            try:
                res = engine.transcribe(utt.audio)
            except Exception:
                log.exception("comparison stt failed")
                continue
            metrics.record("stt_compare", (time.perf_counter() - t0) * 1000)
            seg.alt_model = self.cfg.stt.compare_model
            seg.alt_original = (getattr(res, "text", "") or "").strip()
            seg.alt_romaji = self._alt_reading(seg.alt_original)
            ui_q.put(("alt", seg))
            if self.stages.pool is not None and seg.alt_original:
                self.stages.pool.submit_alt(seg)

    def _alt_reading(self, text: str) -> str:
        """Romanize the comparison transcript, using this lane's own tokenizer.

        A second instance rather than sharing `stages.tokenizer`: the NLP worker
        is using that one concurrently and MeCab taggers are not thread-safe, so
        sharing would corrupt the *primary* reading to save a few hundred ms of
        startup on a diagnostic. Built lazily so the cost lands on this worker's
        thread, and only when a comparison is actually running.
        """
        if not text:
            return ""
        try:
            if self._alt_romanize is None:
                self._alt_romanize = _require("tsutawaru.nlp.pipeline",
                                              "reading", "§4 Phase 3e")
            if self._alt_tokenizer is None:
                build = _require("tsutawaru.nlp.tokenizer", "build_tokenizer",
                                 "§4 Phase 3a")
                self._alt_tokenizer = build(self.cfg.nlp)
            return self._alt_romanize(text, self._alt_tokenizer, self.cfg.nlp)
        except Exception:
            # A missing reading is a cosmetic loss; losing the transcript it
            # belongs to would not be.
            log.debug("comparison romaji failed", exc_info=True)
            return ""

    def _emit_segment(self, utt: Utterance, res):
        from tsutawaru.models import Segment

        text = (getattr(res, "text", "") or "").strip()
        if not text:
            return
        try:
            is_hallucination = _require(
                "tsutawaru.stt.filters", "is_hallucination", "§6.1"
            )
            min_samples = _require(
                "tsutawaru.stt.filters", "PEAK_REF_MIN_SAMPLES", "§6.1"
            )
            # Peak amplitude of the utterance the text came from, against the
            # running level of this source. Whisper's own confidence cannot tell
            # a muttered word from an invention over laughter; relative loudness
            # can — but only relative, since sources differ by ~4x in level.
            # See filters.PEAK_FLOOR_RATIO.
            peak = float(np.abs(utt.audio).max()) if utt.audio.size else 0.0
            self._peaks.append(peak)
            ref = (
                statistics.median(self._peaks)
                if len(self._peaks) >= min_samples else None
            )
            if is_hallucination(text, res, self.cfg.stt, peak=peak, peak_ref=ref):
                return
        except StageNotImplemented:
            pass  # filter not built yet — pass everything through

        seg = Segment.new(
            stream=utt.stream,
            original=text,
            lang=getattr(res, "language", "ja"),
            confidence=getattr(res, "avg_logprob", 0.0),
            no_speech=getattr(res, "no_speech_prob", -1.0),
            continued=utt.forced,
            t_audio_end=utt.t_end,
            t_stt_done=time.monotonic(),
        )
        if self.stages.stt_compare is not None:
            # Only while comparing: an ordinary session's recorded lines stay
            # exactly the shape they were before the A/B lane existed.
            seg.model = self.cfg.stt.model
        # Registered at birth, not on completion. The recorder holds the
        # reference and renders at shutdown, by which point the later stages
        # have filled it in — see recorder.py for why no completion event is
        # trustworthy enough to write on.
        self.recorder.add(seg)
        ui_q.put(("new", seg))  # paint the JP line before translation
        seg_q.put_latest(seg)
        if self.stages.stt_compare is not None:
            # The *segment*, not just the audio, so the second transcript hangs
            # on the line it belongs beside. Consequence: only utterances the
            # primary model kept are ever compared, so "one model heard speech,
            # the other heard nothing" is invisible here by construction.
            compare_q.put_latest((utt, seg))

    def _nlp_worker(self):
        run_nlp = _require("tsutawaru.nlp.pipeline", "annotate", "§4 Phase 3e")
        while not self.stop_evt.is_set():
            try:
                seg = seg_q.get(timeout=0.2)
            except queue.Empty:
                continue
            t0 = time.perf_counter()
            try:
                run_nlp(seg, self.stages.tokenizer, self.cfg.nlp)
            except Exception:
                log.exception("nlp failed")
                continue
            metrics.record("nlp", (time.perf_counter() - t0) * 1000)
            ui_q.put(("romaji", seg))
            trans_q.put_latest(seg)

    def _translate_worker(self):
        pool = self.stages.pool
        while not self.stop_evt.is_set():
            try:
                seg = trans_q.get(timeout=0.2)
            except queue.Empty:
                continue
            if pool is None:
                # provider="none" disables *translation*, not the bundled
                # lexicon. Static glosses are a zero-latency dictionary lookup
                # with no network, and cover ~35% of conversational tokens
                # (plan §4 Phase 4b) — losing particles like は/を/と in offline
                # mode would gut the breakdown tier for no reason.
                self._apply_static_glosses(seg)
                seg.partial = False
                ui_q.put(("breakdown", seg))
                continue
            try:
                # Read per segment, not captured once: the window mutates this
                # flag live when the reader toggles the breakdown tier.
                pool.submit(seg, glosses=self.cfg.ui.show_breakdown)
            except Exception:
                log.exception("translate submit failed")

    # ------------------------------------------------------------ source switch

    def switch_source(self, key: str) -> str:
        """Point per-process capture at a different application.

        Blocking (stops a tap, walks the process table, launches ProcTap's helper),
        so callers on a UI thread must hand this to a worker. Returns a short
        status string for display.

        Order matters. The old tap is closed first, then `raw_q` is drained and the
        VAD is reset, and only then is the new tap opened. Skipping the middle step
        would let bytes captured from the *previous* app decode into the new
        source's first utterance — one transcript line built from two applications,
        which is exactly the kind of defect that is invisible in testing and
        obvious to a user reading the transcript.
        """
        if self._backend != "process":
            log.warning(
                "source switching applies to [audio] backend='process'; "
                "this run uses %r", self._backend,
            )
            return "not available on the device backend"

        from tsutawaru.audio import sources

        src = sources.get(key)
        with self._switch_lock:
            cap = self.stages.capture
            if cap is None:
                return "capture is unavailable"

            try:
                cap.stop()
            except Exception:
                log.debug("stop during switch failed", exc_info=True)

            # Everything still queued belongs to the old application.
            drained = 0
            while True:
                try:
                    raw_q.get_nowait()
                    drained += 1
                except queue.Empty:
                    break
            gate = self.stages.vad
            if gate is not None and hasattr(gate, "reset"):
                gate.reset()

            self.cfg.audio.source = src.key
            cap.source = src
            # Start the new app from its own best guess rather than inheriting an
            # escalation that only made sense for the previous one.
            if hasattr(cap, "candidate"):
                cap.candidate = 0
            # Fresh decoder: the resampler carries filter state and a partial-frame
            # tail from the old stream, and both would smear across the boundary.
            self.stages.decoder = type(self.stages.decoder)(cap.native_sr, cap.channels)
            self._device_name = src.label
            self._silent_warner.rearm()
            self._last_nonzero = time.monotonic()
            self._frames_since_tap = 0
            self._proc_confirmed = False  # new source, new search
            self._last_retap = 0.0  # let the retap loop try again immediately

            log.info("switching source to %s (dropped %d queued frame(s))",
                     src.label, drained)
            try:
                cap.start()
            except Exception as e:
                # Not playing yet is the common case; the retap loop takes over.
                log.info("%s — waiting for it to play audio", e)
                return f"{src.label}: waiting for audio"
        return f"capturing {cap.device}"

    def source_label(self) -> str:
        cap = self.stages.capture
        if cap is None:
            return self._device_name
        return getattr(cap, "device", self._device_name)

    def _try_restart_capture(self) -> None:
        """Rebuild a silently-dead capture stream. Rate-limited, never fatal."""
        if self._backend == "process":
            return  # the retap loop owns recovery for process taps
        now = time.monotonic()
        if now - self._last_restart < RESTART_BACKOFF_S:
            return
        self._last_restart = now
        cap = self.stages.capture
        if cap is None:
            return

        self._restarts += 1
        log.warning(
            "no audio for %.0fs — restarting the capture stream (attempt %d)",
            now - self._last_nonzero, self._restarts,
        )
        try:
            cap.stop()
        except Exception:
            log.debug("stop during restart failed", exc_info=True)
        try:
            # Re-resolve by NAME: indices renumber when devices come and go.
            from tsutawaru.audio.devices import resolve

            cap.device = resolve(self._device_name or self.cfg.audio.device)
            cap.start()
            # The device may have come back at a different rate.
            self.stages.decoder = type(self.stages.decoder)(cap.native_sr, cap.channels)
            self._last_nonzero = time.monotonic()  # give the new stream a fair window
            log.info("capture stream restarted on %s", self._device_name)
        except Exception as e:
            log.warning("capture restart failed: %s — will retry in %.0fs",
                        e, RESTART_BACKOFF_S)

    @staticmethod
    def _apply_static_glosses(seg) -> None:
        """Fill glosses that need no provider. Safe if the table is unavailable."""
        try:
            from tsutawaru.translate.static_gloss import static_gloss
        except ImportError:
            return
        for tok in seg.tokens:
            if tok.gloss is None:
                tok.gloss = static_gloss(tok) or ""

    def _dump(self, utt: Utterance):
        import wave
        from pathlib import Path

        d = Path(self.dump_utterances)
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"utt_{utt.id:05d}{'_forced' if utt.forced else ''}.wav"
        pcm = np.clip(utt.audio, -1.0, 1.0)
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes((pcm * 32767).astype(np.int16).tobytes())

    # ----------------------------------------------------------- lifecycle

    def start(self) -> None:
        specs = [
            ("vad", self._vad_worker, self.stages.vad and self.stages.decoder),
            ("stt", self._stt_worker, self.stages.stt),
            ("stt-compare", self._stt_compare_worker, self.stages.stt_compare),
            ("nlp", self._nlp_worker, self.stages.tokenizer),
            ("translate", self._translate_worker, True),
        ]
        for name, fn, enabled in specs:
            if not enabled:
                log.debug("worker %s not started (stage missing)", name)
                continue
            t = threading.Thread(target=self._guard(fn), name=name, daemon=True)
            t.start()
            self.threads.append(t)

        if self.stages.capture is not None:
            self._start_capture()

    def _start_capture(self) -> None:
        """Open the capture stream. Tolerates a not-yet-playing process source."""
        cap = self.stages.capture
        if self._backend == "process":
            from tsutawaru.audio.sources import SourceUnavailable

            try:
                cap.start()
            except SourceUnavailable as e:
                # Expected, not exceptional: the app is closed or silent. The
                # retap loop will pick it up the moment it starts playing.
                log.info("%s — waiting for it to play audio", e)
                return
            except Exception as e:
                log.warning("could not open the process tap: %s", e)
                return
            self._last_nonzero = time.monotonic()  # fair window for a fresh tap
            self._frames_since_tap = 0
            self._proc_confirmed = False
        else:
            cap.start()
        log.info("capturing from %s", cap.device if self._backend == "process"
                 else self._device_name)

    def _guard(self, fn):
        def wrapped():
            try:
                fn()
            except StageNotImplemented as e:
                log.warning("%s worker stopped: %s", threading.current_thread().name, e)
            except Exception:
                log.exception("%s worker crashed", threading.current_thread().name)
        return wrapped

    def stop(self) -> None:
        """§6.8 shutdown: stop input, drain, flush caches, join, then give up."""
        if self.stop_evt.is_set():
            return
        log.info("shutting down...")
        self.stop_evt.set()

        if self.stages.capture is not None:
            try:
                self.stages.capture.stop()
            except Exception:
                log.debug("capture stop failed", exc_info=True)

        deadline = time.monotonic() + DRAIN_TIMEOUT_S
        while time.monotonic() < deadline and utt_q.qsize():
            time.sleep(0.05)

        if self.stages.pool is not None:
            try:
                self.stages.pool.shutdown()  # flushes the gloss cache
            except Exception:
                log.debug("pool shutdown failed", exc_info=True)

        for t in self.threads:
            t.join(timeout=JOIN_TIMEOUT_S)
        stuck = [t.name for t in self.threads if t.is_alive()]
        if stuck:
            log.warning("threads still running after %.0fs: %s", JOIN_TIMEOUT_S, ", ".join(stuck))

        if self.recorder.enabled:
            # The worker joins above do not cover the translation pool, whose
            # executors are shut down without waiting so exit stays quick. The
            # last line or two can therefore still be mid-translation, and a
            # study sheet whose final entries have no English is exactly the
            # part a reader looks at first. Bounded, and skipped entirely when
            # no provider is running — with translation off, every segment is
            # permanently "awaiting" and the wait would buy nothing.
            if self.stages.pool is not None:
                deadline = time.monotonic() + RECORD_GRACE_S
                while time.monotonic() < deadline and self.recorder.awaiting_english():
                    time.sleep(0.05)
            self.recorder.close()

        dropped = queues.total_dropped()
        if dropped:
            log.info("dropped %d item(s) to stay real-time", dropped)

    def wait(self) -> None:
        """Block until stopped. The UI sink owns the main thread when present."""
        if self.stages.sink is not None:
            self.stages.sink.run(self.stop_evt)
        else:
            while not self.stop_evt.is_set():
                time.sleep(0.2)
