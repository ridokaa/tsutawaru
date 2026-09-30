"""Bounded queues with an explicit drop-oldest policy.

Unbounded queues turn a transient slowdown into permanent, ever-growing lag —
the most common failure mode in real-time pipelines. Every inter-stage queue
here is bounded and drops the *oldest* item when full, so the pipeline stays
near real time instead of accumulating a backlog it can never clear.

The queues are module-level singletons because every stage in the plan imports
them by name (`from tsutawaru.pipeline.queues import raw_q`). `reset_all()` exists
so tests can start from a clean slate.
"""
from __future__ import annotations

import logging
import queue

log = logging.getLogger(__name__)


class DropOldestQueue(queue.Queue):
    """When full, discard the oldest item so we always stay near real time."""

    def __init__(self, maxsize: int = 0, name: str = "queue"):
        super().__init__(maxsize=maxsize)
        self.name = name
        self.dropped = 0

    def put_latest(self, item):
        """Non-blocking put that evicts the oldest entry rather than blocking.

        Returns the evicted item, or None when nothing had to go. Callers that
        own a *visible* object need it: a segment silently dropped here renders
        as "…" forever, which is the defect `_strand` exists to prevent. Callers
        that queue raw frames can keep ignoring the return.
        """
        lost = None
        while True:
            try:
                self.put_nowait(item)
                return lost
            except queue.Full:
                try:
                    lost = self.get_nowait()
                    self.dropped += 1
                    # Log sparsely: under sustained overload this fires constantly.
                    if self.dropped == 1 or self.dropped % 50 == 0:
                        log.warning(
                            "%s full — dropped %d item(s) to stay real-time",
                            self.name,
                            self.dropped,
                        )
                except queue.Empty:  # pragma: no cover - racing consumer
                    pass


raw_q = DropOldestQueue(maxsize=500, name="raw_q")  # ~10 s of 20 ms frames
utt_q = DropOldestQueue(maxsize=8, name="utt_q")  # backpressure onto STT
seg_q = DropOldestQueue(maxsize=32, name="seg_q")
# The sentence lane's only bound. `TranslationPool.submit` runs the translation
# on the thread that drains this queue, so its depth *is* the translator's
# backlog — see the comment on `TranslationPool.submit`.
#
# 4 rather than 32, measured. Service time on the 160-line fixed sample is p50
# 0.73 s / p99 2.68 s against a median 3.0-4.0 s gap between utterances on all
# four corpus clips, so the queue is essentially always empty: simulated over
# 905 real utterance arrivals, depth 4 evicts 0.0% on every clip. What the depth
# buys is the ceiling when something does go wrong — 32 admitted a 93 s backlog,
# and a 59 s line was observed live on 2026-09-30.
trans_q = DropOldestQueue(maxsize=4, name="trans_q")
# (utterance, segment) pairs for the optional second ASR model. Small and
# drop-oldest: a slow comparison model must lose its own lines rather than
# apply backpressure to the pipeline it is measuring.
compare_q = DropOldestQueue(maxsize=4, name="compare_q")
ui_q: queue.Queue = queue.Queue()  # unbounded: rendering must never drop

ALL = (raw_q, utt_q, seg_q, trans_q, compare_q)


def depths() -> dict[str, int]:
    d = {q.name: q.qsize() for q in ALL}
    d["ui_q"] = ui_q.qsize()
    return d


def total_dropped() -> int:
    return sum(q.dropped for q in ALL)


def reset_all() -> None:
    """Empty every queue and zero the drop counters. Tests and restarts."""
    for q in (*ALL, ui_q):
        while True:
            try:
                q.get_nowait()
            except queue.Empty:
                break
    for q in ALL:
        q.dropped = 0
