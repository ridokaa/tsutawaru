"""A turn the VAD cut at max length must read as one utterance, not two lines.

`continued` is set from `utt.forced` on the utterance the flush *truncated*, so
it marks the FRONT half and the line after it carries the rest. Confirmed
against experiments/sessions/: every flagged line ends mid-word (…まあゆ,
…予約の, …行きますけ). Getting the direction backwards would join each pair to
the wrong neighbour, which is why it is asserted here rather than assumed.

The join itself is checked in pixels. It is made by *removing* the gap between
two cards so their backgrounds and accent rails abut, and whether Qt actually
paints two adjacent tables as one continuous block is not something the markup
can tell you — the same lesson test_window_cards.py records about borders on
divs.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6")

from PyQt6 import QtGui  # noqa: E402

from tsutawaru.config import UiCfg  # noqa: E402
from tsutawaru.models import Segment  # noqa: E402
from tsutawaru.pipeline.queues import ui_q  # noqa: E402
from tsutawaru.ui.window_qt import (  # noqa: E402
    RAIL_FINAL, TranscriptWindow, WindowSink, _CSS, _fmt_block,
)

WIDTH = 640
RAIL_X = 5      # inside the 4px rail, past the document's own left margin
FRONT = "あ、受付は二階って書いてるね。一度しか入れないから予約の時間は結構気をつけなくちゃいけなかったのね。なので明日はまあゆ"
BACK = "っくり行くことにして余裕を持たせようか。"


@pytest.fixture(scope="module")
def qapp():
    from PyQt6 import QtWidgets

    yield QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _pair(stream_back: str = "s"):
    front = Segment.new(stream="s", original=FRONT, romaji="a uketsuke",
                        english="Ah, it says reception is upstairs.", partial=False,
                        continued=True)
    back = Segment.new(stream=stream_back, original=BACK, romaji="kkuri",
                       english="maybe give ourselves more time.", partial=False)
    return front, back


def _rail_runs(html: str) -> list[tuple[int, int]]:
    """Vertical runs of rail colour down the rail column, in a real layout."""
    doc = QtGui.QTextDocument()
    doc.setDefaultStyleSheet(_CSS)
    doc.setTextWidth(WIDTH)
    doc.setHtml(f"<body>{html}</body>")
    img = QtGui.QImage(WIDTH, max(1, int(doc.size().height())),
                       QtGui.QImage.Format.Format_RGB32)
    img.fill(QtGui.QColor("#0f1115"))
    painter = QtGui.QPainter(img)
    doc.drawContents(painter)
    painter.end()

    runs, start = [], None
    for y in range(img.height() + 1):
        hit = y < img.height() and img.pixelColor(RAIL_X, y).name() == RAIL_FINAL
        if hit and start is None:
            start = y
        elif not hit and start is not None:
            runs.append((start, y - 1))
            start = None
    return runs


def test_joined_cards_paint_one_continuous_rail(qapp):
    front, back = _pair()
    joined = (_fmt_block(front, UiCfg(), joined_below=True)
              + _fmt_block(back, UiCfg(), continues_above=True))
    apart = _fmt_block(front, UiCfg()) + _fmt_block(back, UiCfg())

    assert len(_rail_runs(joined)) == 1, "the joined pair must paint one rail"
    assert len(_rail_runs(apart)) == 2, "unrelated lines must stay two cards"


def test_marker_is_on_the_back_half_only():
    front, back = _pair()
    assert "continued" not in _fmt_block(front, UiCfg(), joined_below=True)
    assert "↳ continued" in _fmt_block(back, UiCfg(), continues_above=True)


# --- which half the flag marks, and who gets joined to whom ----------------

@pytest.fixture
def sink(qapp):
    while not ui_q.empty():          # the queue is shared across test modules
        ui_q.get_nowait()
    s = WindowSink(UiCfg(), device_name="dev")
    win = TranscriptWindow(s.cfg, on_clear=s.clear)
    win.resize(700, 760)
    win.show()
    qapp.processEvents()
    yield s, win, qapp
    win.close()


def _feed(sink, *segs):
    s, win, app = sink
    for seg in segs:
        ui_q.put(("new", seg))
    s.tick(win)
    app.processEvents()


def test_the_flagged_line_joins_down_not_up(sink):
    """The front half closes flush; the line before it is untouched."""
    s, win, app = sink
    before = Segment.new(stream="s", original="ポイントが。", english="The point is",
                         partial=False)
    front, back = _pair()
    _feed(sink, before, front, back,
          Segment.new(stream="s", original="そうね。", english="Right.", partial=False))

    # positions 0..3; the pair is 1-2, so only card 1 closes joined and only
    # card 2 is marked.
    assert s._card(0).endswith('<div class="gap">&nbsp;</div>')
    assert s._card(1).endswith("</table>")
    assert "↳ continued" not in s._card(1)
    assert "↳ continued" in s._card(2)
    assert s._card(2).endswith('<div class="gap">&nbsp;</div>')


def test_a_pair_never_spans_a_source_switch(sink):
    """The divider would land inside the joined card, and the halves are not one
    utterance across two apps anyway."""
    s, win, app = sink
    front, back = _pair(stream_back="other")
    _feed(sink, front, back,
          Segment.new(stream="other", original="次。", english="Next.", partial=False))

    assert s._card(0).endswith('<div class="gap">&nbsp;</div>')
    assert "↳ continued" not in s._card(1)


def test_marker_survives_the_pane_boundary(sink):
    """The back half is usually the newest line, so it lands in the live lane —
    a separate widget, where the flush join cannot reach it."""
    s, win, app = sink
    front, back = _pair()
    _feed(sink, front, back)

    assert "↳ continued" in win.live.toPlainText()
    assert FRONT[:8] in win.view.toPlainText()      # front half stayed in history
    assert FRONT[:8] not in win.live.toPlainText()


def test_no_horizontal_scrollbar_when_the_history_shrinks(sink):
    """Regression: a vertical scrollbar disappearing left the document 16px wider
    than the viewport, so a transcript that wraps grew a horizontal scrollbar."""
    s, win, app = sink
    long_ones = [
        Segment.new(stream="s", original=FRONT, romaji="a uketsuke",
                    english="a long english line " * 3, partial=False)
        for _ in range(8)
    ]
    _feed(sink, *long_ones)
    assert win.view.verticalScrollBar().maximum() > 0, "test needs a scrollbar first"

    s.clear()
    _feed(sink, Segment.new(stream="s", original="短い。", english="Short.",
                            partial=False))
    assert not win.view.horizontalScrollBar().isVisible()
    assert win.view.document().textWidth() <= win.view.viewport().width()
