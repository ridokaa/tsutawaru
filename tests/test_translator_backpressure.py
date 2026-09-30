"""The sentence lane must not grow a backlog behind a bounded queue.

The regression these guard: `TranslationPool.submit` used to hand the
translation to a `ThreadPoolExecutor(max_workers=1)`, whose work queue is
unbounded. `trans_q` is bounded drop-oldest, but submit emptied it instantly, so
the slowest stage in the pipeline was the only one with no backpressure — and a
live session recorded a line 59 s behind the speech it came from.
"""
from __future__ import annotations

import threading
import time

import pytest

from tsutawaru.config import TranslateCfg
from tsutawaru.models import Segment, Token
from tsutawaru.pipeline.queues import DropOldestQueue, trans_q, ui_q
from tsutawaru.translate.cache import GlossCache
from tsutawaru.translate.pool import TranslationPool


class _Slow:
    name = "slow"

    def __init__(self, delay: float = 0.05):
        self.delay = delay
        self.calls = 0

    def sentence(self, text: str, remember: bool = True) -> str:
        self.calls += 1
        time.sleep(self.delay)
        return f"<{text}>"

    def batch(self, texts: list[str]) -> list[str]:
        return ["" for _ in texts]


def _seg(text: str = "ねこ") -> Segment:
    seg = Segment.new(stream="test", original=text)
    seg.tokens = [Token(surface=text, base_form=text, pos="名詞")]
    return seg


@pytest.fixture
def pool():
    cfg = TranslateCfg(gloss_workers=1)
    p = TranslationPool(_Slow(), GlossCache(8, persist=False), cfg)
    yield p
    p.gloss_ex.shutdown(wait=False)


def test_submit_translates_on_the_calling_thread(pool):
    """Synchronous by design: the caller's thread IS the lane's only worker."""
    seg = _seg()
    where = []
    orig = pool.backend.sentence

    def spy(text, remember=True):
        where.append(threading.current_thread().name)
        return orig(text, remember=remember)

    pool.backend.sentence = spy
    pool.submit(seg, glosses=False)

    assert seg.english == "<ねこ>", "submit returned before translating"
    assert where == [threading.current_thread().name], (
        "the translation ran on another thread, so nothing bounds its backlog"
    )


def test_submit_blocks_so_the_queue_can_apply_backpressure(pool):
    """A slow backend must slow the caller down, not queue up behind it."""
    pool.backend.delay = 0.2
    t0 = time.perf_counter()
    pool.submit(_seg(), glosses=False)
    assert time.perf_counter() - t0 >= 0.15, (
        "submit returned early — the work went somewhere unbounded"
    )


def test_the_pool_owns_no_unbounded_sentence_executor(pool):
    """The shape of the bug, asserted directly so it cannot come back."""
    assert not hasattr(pool, "sentence_ex"), (
        "a sentence ThreadPoolExecutor is an unbounded queue in front of the "
        "slowest stage in the pipeline"
    )


def test_put_latest_hands_back_what_it_evicted():
    """Callers that own a visible object need to know it was dropped."""
    q = DropOldestQueue(maxsize=2, name="probe")
    assert q.put_latest("a") is None
    assert q.put_latest("b") is None
    assert q.put_latest("c") == "a", "the evicted item was discarded silently"
    assert q.dropped == 1


def test_an_evicted_line_settles_instead_of_reading_as_pending():
    """`trans_q` overflow is now the translator's only loss. It must be visible."""
    from tsutawaru.pipeline.orchestrator import Orchestrator
    from tsutawaru.stt.filters import DROPS

    orch = Orchestrator.__new__(Orchestrator)  # no audio devices wanted here
    while not ui_q.empty():
        ui_q.get_nowait()
    before = DROPS["mt-backlog"]

    lost = _seg()
    orch._strand(lost, "mt-backlog")

    assert lost.dropped and not lost.partial, "the card would wait forever"
    assert DROPS["mt-backlog"] == before + 1, "not counted for --stats"
    assert ui_q.get_nowait()[0] == "dropped", "the window is never told"

    orch._strand(lost, "mt-backlog")
    assert DROPS["mt-backlog"] == before + 1, "double-counted on a second pass"


def test_trans_q_is_shallow_enough_to_bound_the_translator():
    """Depth is the translator's backlog now, so it is load-bearing.

    At the measured p99 service time of 2.7 s, 32 admitted a 93 s backlog. The
    simulation over 905 real utterance arrivals evicts 0.0% at depth 4.
    """
    assert trans_q.maxsize <= 8, f"trans_q depth {trans_q.maxsize} is a backlog"
