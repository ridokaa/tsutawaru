"""Session capture: a JSONL log and a Markdown study sheet.

Segments are registered once, at birth, and the recorder keeps the *reference*.
Workers mutate them in place — NLP fills romaji and tokens, the translation pool
fills English and glosses — so reading them at write time yields the finished
four tiers without the recorder ever having to decide when a line is done.

That decision is the whole reason this is not event-driven. There is no single
event meaning "complete": the gloss lane sets `partial = False` while the
sentence lane is still fetching English, so a line written the moment it looked
finished would routinely be logged with an empty translation. Holding references
and rendering late sidesteps the race instead of racing it.

Because every write renders the full held state, the JSONL is rewritten whole
rather than appended to. That is O(n) per flush, which is affordable at these
sizes and is the only way a file on disk can stay consistent with segments that
are still changing behind it.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

from tsutawaru.logbus import _pct
from tsutawaru.models import Segment

log = logging.getLogger(__name__)

# Segments added between JSONL rewrites. Small enough that a kill -9 costs at
# most a few lines, large enough that a busy session is not rewriting the file
# on every utterance.
FLUSH_EVERY = 25

# Word classes worth studying. Particles and auxiliaries carry grammar rather
# than vocabulary, and they would otherwise dominate a frequency-sorted table
# purely by being unavoidable — the top ten rows would be は, を, の, and た.
CONTENT_POS = ("名詞", "動詞", "形容詞", "副詞")


class SessionRecorder:
    """Accumulates segments and writes them out on close.

    Safe to construct with neither path: `enabled` is then False and `add` is a
    cheap no-op, so callers need no conditional around it.
    """

    def __init__(
        self,
        jsonl: str | None = None,
        markdown: str | None = None,
        meta: dict | None = None,
    ):
        self.jsonl = Path(jsonl) if jsonl else None
        self.markdown = Path(markdown) if markdown else None
        self.meta = meta or {}
        self._segs: list[Segment] = []
        self._lock = threading.Lock()
        self._since_flush = 0
        self._started = time.monotonic()
        self._started_wall = datetime.now()

    @property
    def enabled(self) -> bool:
        return bool(self.jsonl or self.markdown)

    def add(self, seg: Segment) -> None:
        """Register a segment. Called once per line, from the STT worker."""
        if not self.enabled:
            return
        with self._lock:
            self._segs.append(seg)
            self._since_flush += 1
            due = self._since_flush >= FLUSH_EVERY
        if due:
            self.flush()

    def awaiting_english(self) -> int:
        """Registered segments with no translation yet.

        Read during shutdown to decide whether waiting a moment longer would
        actually save anything. Counting is cheaper than guessing.
        """
        with self._lock:
            return sum(1 for s in self._segs if not s.english)

    def flush(self) -> None:
        """Rewrite the JSONL from current state. Never raises."""
        if self.jsonl is None:
            return
        with self._lock:
            rows = [s.to_dict() for s in self._segs]
            self._since_flush = 0
        try:
            self.jsonl.parent.mkdir(parents=True, exist_ok=True)
            with self.jsonl.open("w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        except OSError:
            # A recorder that takes down the pipeline it is observing is worse
            # than one that loses a session log.
            log.warning("could not write %s", self.jsonl, exc_info=True)

    def close(self) -> None:
        """Final flush plus the study sheet. Never raises."""
        if not self.enabled:
            return
        self.flush()
        if self.jsonl is not None and self._segs:
            log.info("recorded %d line(s) to %s", len(self._segs), self.jsonl)
        if self.markdown is not None:
            try:
                self.markdown.parent.mkdir(parents=True, exist_ok=True)
                self.markdown.write_text(self._render_markdown(), encoding="utf-8")
                log.info("study sheet written to %s", self.markdown)
            except OSError:
                log.warning("could not write %s", self.markdown, exc_info=True)

    # ------------------------------------------------------------- rendering

    def _vocabulary(self) -> list[dict]:
        """Content words seen this session, most frequent first.

        Grouped on (base_form, pos) so that 潜って and 潜る count as one word,
        while the surfaces actually heard are kept alongside — recognising the
        inflected form in speech is the skill being practised, and a table that
        only ever shows the dictionary form does not train it.
        """
        counts: Counter[tuple[str, str]] = Counter()
        info: dict[tuple[str, str], dict] = {}
        for seg in self._segs:
            for t in seg.tokens:
                if t.pos not in CONTENT_POS or not t.base_form:
                    continue
                key = (t.base_form, t.pos)
                counts[key] += 1
                slot = info.setdefault(
                    key, {"romaji": "", "gloss": "", "surfaces": []}
                )
                if t.surface not in slot["surfaces"]:
                    slot["surfaces"].append(t.surface)
                # Only an uninflected occurrence's romaji belongs next to the
                # headword. Taking it from any token would print 潜る as
                # "moguttemita" — the row would then be teaching a reading for
                # a word that is not the one in the Word column. Rows with no
                # such occurrence are romanized from the base form below.
                if t.romaji and t.surface == t.base_form:
                    slot["romaji"] = t.romaji
                if t.gloss and not slot["gloss"]:
                    slot["gloss"] = t.gloss

        try:
            from tsutawaru.nlp.romaji import to_romaji
        except ImportError:  # pragma: no cover - nlp stack always present
            to_romaji = None

        rows = []
        for (base, pos), n in counts.most_common():
            slot = info[(base, pos)]
            heard = [s for s in slot["surfaces"] if s != base]
            romaji = slot["romaji"]
            if not romaji and to_romaji is not None:
                # Every occurrence was inflected (常に「潜ってみた」, never 潜る).
                # Romanize the dictionary form directly rather than borrowing a
                # reading that belongs to a different surface.
                romaji = to_romaji(base)
            rows.append({
                "word": base,
                "pos": pos,
                "romaji": romaji,
                "gloss": slot["gloss"],
                "count": n,
                "heard": heard,
            })
        return rows

    def _latency(self) -> str:
        """p50/p95 end-of-speech -> full line, or "" when nothing was timed.

        The header is the only place this figure can be read without extra
        tooling, which is what turns a recording into the measurement itself.
        """
        vals = sorted(
            v for v in (s.timing()["line_ms"] for s in self._segs) if v is not None
        )
        if not vals:
            return ""
        return f"p50 {_pct(vals, 0.50):.0f} ms · p95 {_pct(vals, 0.95):.0f} ms"

    def _render_markdown(self) -> str:
        started = self._started_wall.strftime("%Y-%m-%d %H:%M")
        mins = (time.monotonic() - self._started) / 60.0
        vocab = self._vocabulary()

        bits = [f"{len(self._segs)} lines", f"{len(vocab)} words", f"{mins:.0f} min"]
        lat = self._latency()
        if lat:
            bits.append(lat)
        for k in ("source", "model", "provider"):
            if self.meta.get(k):
                bits.insert(0, f"{k}: {self.meta[k]}")

        out = [f"# tsutawaru — {started}", "", " · ".join(bits), ""]

        if not self._segs:
            out += ["Nothing was transcribed this session.", ""]
            return "\n".join(out)

        out += [
            "## Vocabulary",
            "",
            "Nouns, verbs, adjectives and adverbs, most frequent first. Meanings are "
            "the per-word glosses from the breakdown tier — dictionary senses looked "
            "up on the base form, so they can miss the sense actually used in the "
            "line. Read them against the transcript below, not on their own.",
            "",
            "| Word | Romaji | Meaning | Seen | As heard |",
            "| --- | --- | --- | ---: | --- |",
        ]
        for r in vocab:
            heard = "、".join(r["heard"]) if r["heard"] else ""
            out.append(
                f"| {r['word']} | {r['romaji'] or ''} | {r['gloss'] or '—'} "
                f"| {r['count']} | {heard} |"
            )

        out += ["", "## Transcript", ""]
        for i, seg in enumerate(self._segs, 1):
            out.append(f"**{i}.** {seg.original}")
            if seg.romaji:
                out.append(f"*{seg.romaji}*")
            out.append(seg.english or "—")
            out.append("")
        return "\n".join(out)
