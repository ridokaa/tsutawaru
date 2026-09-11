"""Structured logging + latency metrics.

`metrics.record(stage, ms)` is called from every worker. You cannot optimise
what you do not measure, so this is wired in Phase 0 rather than bolted on at
the end.

Logging goes to stderr so it never corrupts a piped transcript on stdout.
"""
from __future__ import annotations

import logging
import math
import sys
import threading
import time
from collections import defaultdict, deque
from typing import Callable

# Rolling window per stage. 512 samples is a few minutes of conversation and
# keeps p95 responsive to recent conditions rather than the whole session.
_WINDOW = 512


class Metrics:
    """Thread-safe rolling latency recorder."""

    def __init__(self, window: int = _WINDOW):
        self._window = window
        self._d: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=window))
        self._counts: dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()
        self._t0 = time.monotonic()

    def record(self, stage: str, ms: float) -> None:
        with self._lock:
            self._d[stage].append(ms)
            self._counts[stage] += 1

    def timer(self, stage: str) -> "_Timer":
        """`with metrics.timer("stt"): ...`"""
        return _Timer(self, stage)

    def snapshot(self) -> dict[str, dict]:
        with self._lock:
            out = {}
            for stage, vals in self._d.items():
                if not vals:
                    continue
                s = sorted(vals)
                out[stage] = {
                    "n": self._counts[stage],
                    "p50": _pct(s, 0.50),
                    "p95": _pct(s, 0.95),
                    "max": s[-1],
                }
            return out

    def reset(self, *stages: str) -> None:
        """Clear rolling samples for `stages` (or all of them if none given).

        Used when swapping a pipeline component at runtime: the percentiles are
        a rolling window with no notion of which component produced them, so
        without this a switch silently averages the old and new implementations.
        """
        with self._lock:
            targets = stages or tuple(self._d.keys())
            for s in targets:
                self._d.pop(s, None)
                self._counts.pop(s, None)

    def uptime(self) -> float:
        return time.monotonic() - self._t0

    def format_table(self, extra: dict[str, str] | None = None) -> str:
        snap = self.snapshot()
        if not snap and not extra:
            return "no metrics recorded yet"
        w = max([len(s) for s in snap] + [5])
        lines = [
            f"{'stage'.ljust(w)}  {'n':>6}  {'p50 ms':>9}  {'p95 ms':>9}  {'max ms':>9}"
        ]
        lines.append("-" * len(lines[0]))
        for stage in sorted(snap):
            m = snap[stage]
            lines.append(
                f"{stage.ljust(w)}  {m['n']:>6}  "
                f"{m['p50']:>9.1f}  {m['p95']:>9.1f}  {m['max']:>9.1f}"
            )
        if extra:
            lines.append("")
            for k, v in extra.items():
                lines.append(f"{k.ljust(w)}  {v}")
        return "\n".join(lines)


def _pct(sorted_vals: list[float], q: float) -> float:
    """Nearest-rank percentile: always a value some sample actually took.

    Interpolating would report a duration no line ever ran at, which at the
    sample sizes here (a few dozen lines in a session) is the common case.
    """
    if not sorted_vals:
        return 0.0
    return sorted_vals[min(len(sorted_vals) - 1, math.ceil(q * len(sorted_vals)) - 1)]


class _Timer:
    __slots__ = ("_m", "_stage", "_t")

    def __init__(self, m: Metrics, stage: str):
        self._m, self._stage = m, stage

    def __enter__(self):
        self._t = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self._m.record(self._stage, (time.perf_counter() - self._t) * 1000.0)
        return False


metrics = Metrics()


def setup_logging(verbose: bool = False, quiet: bool = False) -> None:
    """Configure root logging to stderr.

    stdout is reserved for the transcript so `--plain > file` stays clean.
    """
    level = logging.DEBUG if verbose else (logging.WARNING if quiet else logging.INFO)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
    # These are chatty and we never act on their output.
    for noisy in ("urllib3", "httpx", "httpcore", "huggingface_hub", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


class RateLimitedWarner:
    """Emit a warning once, then stay quiet until the condition clears.

    Used by the §6.6 silent-stream detector: a message repeating every 5 s
    through a quiet stretch is worse than no message at all.

    `warn()` accepts a **callable** as well as a string, and this matters for
    correctness rather than style. The silent-stream message is expensive to
    build — it shells out to `system_profiler` (~200 ms) to report the current
    output device. Passing an eagerly-formatted string paid that cost on every
    call while suppressed, which starved the VAD thread and overflowed raw_q at
    ~600 dropped frames per 30 s. Suppressing the log is not enough; the message
    must not be built at all unless it will be emitted.
    """

    def __init__(self, log: logging.Logger):
        self._log = log
        self._armed = True

    def warn(self, msg: "str | Callable[[], str]") -> bool:
        """Emit if armed. Returns True if the message was actually printed.

        Prefer passing a zero-arg callable when the message is costly to build.
        """
        if not self._armed:
            return False
        self._armed = False
        self._log.warning(msg() if callable(msg) else msg)
        return True

    def arm_once(self) -> bool:
        """Claim the right to emit, without emitting.

        For callers that want to do the emitting themselves — e.g. off-thread,
        so that neither building nor writing the message runs on a real-time
        path. Returns True at most once per armed cycle.
        """
        if not self._armed:
            return False
        self._armed = False
        return True

    def rearm(self) -> None:
        self._armed = True
