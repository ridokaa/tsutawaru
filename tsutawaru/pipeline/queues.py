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

    def put_latest(self, item) -> None:
        """Non-blocking put that evicts the oldest entry rather than blocking."""
        while True:
            try:
                self.put_nowait(item)
                return
            except queue.Full:
                try:
                    self.get_nowait()
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
trans_q = DropOldestQueue(maxsize=32, name="trans_q")
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
