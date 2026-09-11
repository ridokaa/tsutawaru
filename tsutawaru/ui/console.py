"""Console output sink: rich.Live scrolling transcript or --plain logging."""
from __future__ import annotations

import queue
import threading
from typing import Optional
from rich.console import Console, Group
from rich.live import Live
from rich.text import Text

from tsutawaru.config import UiCfg
from tsutawaru.models import Segment
from tsutawaru.pipeline.queues import ui_q
from tsutawaru.ui.formatter import render_block


class ConsoleSink:
    def __init__(self, cfg: UiCfg, ws_sink: Optional[object] = None,
                 expect_translation: bool = True):
        self.cfg = cfg
        self.ws_sink = ws_sink
        # --plain prints a segment once, fully complete. "Complete" means every
        # lane that will report has reported. The sentence and gloss lanes run
        # on independent executors (plan §4 Phase 4d), so neither event is
        # reliably last — with a remote provider the gloss batch usually lands
        # ~250 ms BEFORE the sentence translation.
        self._expect = {"breakdown"} | ({"english"} if expect_translation else set())
        self.segments: dict[int, Segment] = {}
        self.order: list[int] = []
        self.console = Console()
        # --plain emits each segment exactly once. Segments are mutated in place
        # and traverse ui_q several times ("new" -> "romaji" -> "breakdown"), so
        # a completed segment can still have earlier events queued behind it.
        self._emitted: set[int] = set()
        self._emitted_order: list[int] = []
        self._seen: dict[int, set[str]] = {}

    def run(self, stop_evt: threading.Event) -> None:
        if self.cfg.plain:
            self._run_plain(stop_evt)
        else:
            self._run_live(stop_evt)

    def _run_plain(self, stop_evt: threading.Event) -> None:
        while not stop_evt.is_set():
            try:
                kind, seg = ui_q.get(timeout=0.2)
            except queue.Empty:
                continue

            if self.ws_sink is not None:
                try:
                    self.ws_sink.push(seg)
                except Exception:
                    pass

            # BUG FIX 1: the old condition was `kind == "breakdown" or not
            # seg.partial`. Once the translate stage cleared `partial`, every
            # still-queued earlier event for that same segment also satisfied it,
            # printing the block two or three times.
            # BUG FIX 2: gating on "breakdown" alone printed before the English
            # line arrived, because the gloss lane finishes first.
            if seg.id in self._emitted:
                continue
            seen = self._seen.setdefault(seg.id, set())
            seen.add(kind)
            if not self._expect.issubset(seen):
                continue

            self._seen.pop(seg.id, None)
            self._emitted.add(seg.id)
            self._emitted_order.append(seg.id)
            if len(self._emitted_order) > 4096:  # bound the memory over a long session
                self._emitted.discard(self._emitted_order.pop(0))

            text = render_block(seg, self.cfg)
            print(text, flush=True)
            if self.cfg.log_file:
                with open(self.cfg.log_file, "a", encoding="utf-8") as fh:
                    fh.write(text + "\n")

    def _run_live(self, stop_evt: threading.Event) -> None:
        with Live(
            console=self.console, refresh_per_second=8, vertical_overflow="crop"
        ) as live:
            while not stop_evt.is_set():
                try:
                    kind, seg = ui_q.get(timeout=0.2)
                except queue.Empty:
                    continue

                if self.ws_sink is not None:
                    try:
                        self.ws_sink.push(seg)
                    except Exception:
                        pass

                if seg.id not in self.segments:
                    self.order.append(seg.id)
                    if len(self.order) > self.cfg.max_lines:
                        self.segments.pop(self.order.pop(0), None)
                self.segments[seg.id] = seg

                # Use Text(...) NOT Text.from_markup(...) so '[' and ']' in Japanese are not parsed as tags
                blocks = [
                    Text(render_block(self.segments[i], self.cfg))
                    for i in self.order[-self.cfg.max_lines :]
                ]
                live.update(Group(*blocks))

                if (kind == "breakdown" or not seg.partial) and self.cfg.log_file:
                    with open(self.cfg.log_file, "a", encoding="utf-8") as fh:
                        fh.write(render_block(seg, self.cfg) + "\n")
