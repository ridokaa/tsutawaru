"""Two-executor priority translation pool (1 sentence lane + N gloss workers)."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import time

from tsutawaru.config import TranslateCfg
from tsutawaru.logbus import metrics
from tsutawaru.models import Segment
from tsutawaru.pipeline.queues import ui_q
from tsutawaru.translate.base import Translator
from tsutawaru.translate.cache import GlossCache
from tsutawaru.translate.jmdict import gloss as jmdict_gloss
from tsutawaru.translate.static_gloss import static_gloss


def _name(backend: Translator) -> str:
    """Backend identifier, tolerant of one that never declared it.

    Both call sites run per segment, so raising here would take out the whole
    lane over a cosmetic label. "unknown" keeps the cache honest — it is its own
    namespace, so unnamed results can never be mistaken for a named backend's.
    """
    return getattr(backend, "name", None) or "unknown"


class TranslationPool:
    def __init__(self, backend: Translator, cache: GlossCache, cfg: TranslateCfg):
        self.backend = backend
        self.cache = cache
        self.cfg = cfg
        # Dedicated lane: Sentence jobs never queue behind gloss batches
        self.sentence_ex = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="xlate-sent"
        )
        self.gloss_ex = ThreadPoolExecutor(
            max_workers=cfg.gloss_workers, thread_name_prefix="xlate-gloss"
        )

    def submit(self, seg: Segment, glosses: bool = True) -> None:
        """`glosses=False` runs the sentence lane alone (breakdown tier hidden).

        The gloss lane is a per-token round-trip whose only consumer is a tier
        the reader has turned off, so skipping it is the whole saving. It also
        makes the sentence lane the last one to report, which is why it has to
        take over marking the line complete — see `_sentence`.
        """
        self.sentence_ex.submit(self._sentence, seg, not glosses)  # priority lane
        if glosses:
            self.gloss_ex.submit(self._glosses, seg)  # bulk lane

    def submit_glosses(self, seg: Segment) -> None:
        """Fill in a line's breakdown after the fact.

        For a segment that went through with `glosses=False` and whose breakdown
        the reader has now opened. `_glosses` skips tokens that already carry a
        gloss and checks the static table and the cache before the network, so
        re-running it on a partly-filled segment costs only the misses.
        """
        self.gloss_ex.submit(self._glosses, seg)

    def submit_alt(self, seg: Segment) -> None:
        """Comparison transcript, on the bulk lane.

        Not the sentence lane: that one is single-threaded and a diagnostic must
        never queue in front of a real line's translation.
        """
        self.gloss_ex.submit(self._alt_sentence, seg)

    def shutdown(self) -> None:
        self.sentence_ex.shutdown(wait=False)
        self.gloss_ex.shutdown(wait=False)
        self.cache.flush()

    def _sentence(self, seg: Segment, completes: bool = False) -> None:
        """`completes` when no gloss job was submitted for this segment.

        Marking the line finished is normally the gloss lane's last act, and it
        is load-bearing in three places: the block stops rendering as pending,
        `line_ms` gets its end point, and the `--log-file` sink waits on the
        breakdown event before writing a line. With the gloss lane skipped
        nobody else would ever do it, so the line would hang pending forever.
        """
        t0 = time.perf_counter()
        backend = self.backend  # read once, so one job uses one backend
        try:
            seg.english = backend.sentence(seg.original)
        except Exception as e:
            seg.english = f"[translation unavailable: {type(e).__name__}]"
        metrics.record("xlate_sentence", (time.perf_counter() - t0) * 1000)
        ui_q.put(("english", seg))
        if completes:
            # Glosses stay None rather than "": the breakdown renders None as
            # "…" (not looked up) and "" as "?" (looked up, nothing found), and
            # a later backfill needs that difference to know what to fetch.
            seg.partial = False
            seg.t_complete = time.monotonic()
            ui_q.put(("breakdown", seg))

    def _alt_sentence(self, seg: Segment) -> None:
        """`alt_original` -> `alt_english`. Never touches the primary tiers.

        No `t_complete`, no `partial` change: a slow comparison model must not
        hold lines open, or the study sheet's latency figures end up measuring
        the diagnostic instead of the pipeline.
        """
        t0 = time.perf_counter()
        backend = self.backend
        try:
            seg.alt_english = backend.sentence(seg.alt_original, remember=False)
        except Exception as e:
            seg.alt_english = f"[translation unavailable: {type(e).__name__}]"
        metrics.record("xlate_alt", (time.perf_counter() - t0) * 1000)
        ui_q.put(("alt", seg))

    def _glosses(self, seg: Segment) -> None:
        t0 = time.perf_counter()
        backend = self.backend  # read once, so one job uses one backend
        pending, keys = [], []
        for tok in seg.tokens:
            s = static_gloss(tok)  # merged surface first, then base
            if s:
                tok.gloss = s
                continue
            # JMdict before the network, for every provider: it answers the
            # question the breakdown actually asks (what does this dictionary
            # form mean, out of context) in ~0.02 ms with no round-trip, where
            # word-by-word MT invents a sentence. Measured on the 20260822
            # corpus it answers 96.7% of tokens together with static_gloss, so
            # the backend below sees only proper nouns and ASR garbage.
            s = jmdict_gloss(tok)
            if s:
                tok.gloss = s
                continue
            # Keyed on the backend that will answer, not on cfg.provider: the
            # two diverge whenever config asks for something it cannot have
            # (deepl without a key falls back to Google). Rows live in the
            # SQLite L2 cache across restarts, so a mislabelled one is forever.
            k = f"{_name(backend)}:{tok.base_form}:{tok.pos}"
            hit = self.cache.get(k)
            if hit is not None:
                tok.gloss = hit
                continue
            pending.append(tok)
            keys.append(k)

        if pending:
            try:
                res = backend.batch([t.base_form for t in pending])
                for tok, k, v in zip(pending, keys, res):
                    v = (v or "").strip().lower()
                    tok.gloss = v
                    if v:
                        self.cache.put(k, v)
            except Exception:
                for tok in pending:
                    tok.gloss = ""  # renders as "?"

        metrics.record("xlate_gloss", (time.perf_counter() - t0) * 1000)
        seg.partial = False
        seg.t_complete = time.monotonic()
        ui_q.put(("breakdown", seg))


if __name__ == "__main__":  # self-check: python -m tsutawaru.translate.pool
    # The regression this guards: with the gloss lane skipped, nobody stamps the
    # line complete, so every line renders pending forever, `line_ms` is never
    # set and --log-file never writes a row.
    import queue as _queue

    from tsutawaru.models import Segment, Token

    class _Stub:
        name = "stub"

        def sentence(self, text: str) -> str:
            return f"<{text}>"

        def batch(self, texts: list[str]) -> list[str]:
            return [f"<{t}>" for t in texts]

    def _drain() -> dict:
        seen = {}
        while True:
            try:
                kind, seg = ui_q.get_nowait()
            except _queue.Empty:
                return seen
            seen[kind] = seg

    def _run(glosses: bool) -> tuple[Segment, dict]:
        pool = TranslationPool(_Stub(), GlossCache(8, persist=False), TranslateCfg())
        seg = Segment.new(stream="test", original="ねこ")
        # A word the bundled lexicon cannot know, so the backend is the only
        # thing that can fill it — a real one like ねこ is answered by
        # static_gloss before the backend is ever consulted.
        seg.tokens = [Token(surface="ぬるぽぽぽ", base_form="ぬるぽぽぽ", pos="名詞",
                            romaji="nurupopopo")]
        pool.submit(seg, glosses=glosses)
        pool.sentence_ex.shutdown(wait=True)
        pool.gloss_ex.shutdown(wait=True)
        return seg, _drain()

    seg, seen = _run(glosses=False)
    assert seg.english == "<ねこ>", seg.english
    assert seg.partial is False, "line left pending with the gloss lane skipped"
    assert seg.t_complete > 0, "line_ms would never get an end point"
    assert "breakdown" in seen, "--log-file waits on this event before writing"
    assert seg.tokens[0].gloss is None, "skipped glosses must stay None, not ''"

    seg, seen = _run(glosses=True)
    assert seg.partial is False and seg.t_complete > 0
    assert seg.tokens[0].gloss == "<ぬるぽぽぽ>", seg.tokens[0].gloss

    print("pool self-check ok")
