"""The three reading fixes: copyable transcript, a still live lane, visible doubt.

Each assertion targets the way the old behaviour actually failed:

  * `setHtml()` at a 100ms poll destroyed the selection, so the transcript could
    not be copied from while audio was flowing — which is exactly when there is
    something worth pasting into a dictionary.
  * the newest segment sat at the bottom edge of a growing document and moved
    twice per line, as it filled in and again as the next one arrived.
  * `confidence` was carried on every Segment and rendered nowhere, so a line the
    ASR half-guessed looked identical to a clean one — under a fluent English
    translation that makes the noise read as signal.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6")

from tsutawaru.config import UiCfg  # noqa: E402
from tsutawaru.models import Segment, Token  # noqa: E402
from tsutawaru.pipeline.queues import ui_q  # noqa: E402
from tsutawaru.ui.window_qt import (  # noqa: E402
    LIVE_MAX_PX, LIVE_MIN_PX, TOK_SCHEME, TranscriptWindow, WindowSink,
    _fmt_block, _untrusted,
)

JP = "昨日の夜に友達と映画を見に行ったんだけど、思っていたよりずっと面白かった"


@pytest.fixture(scope="module")
def qapp():
    from PyQt6 import QtWidgets

    yield QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def sink(qapp):
    while not ui_q.empty():          # other tests share the module-level queue
        ui_q.get_nowait()
    s = WindowSink(UiCfg(), device_name="Test Device")
    win = TranscriptWindow(s.cfg, on_clear=s.clear, on_toggle=s.toggle)
    win.resize(700, 760)
    win.show()
    qapp.processEvents()
    yield s, win, qapp
    win.close()


def _push(n: int = 1, **kw) -> list[Segment]:
    segs = []
    for i in range(n):
        seg = Segment.new(stream="main", original=f"{JP}{i}",
                          romaji=f"line {i}", english=f"this is line {i}",
                          partial=False, **kw)
        segs.append(seg)
        ui_q.put(("new", seg))
    return segs


# --- 1. the transcript is copyable ----------------------------------------

def test_selection_survives_incoming_speech(sink):
    """The defect: a new line landing wiped the reader's selection."""
    s, win, app = sink
    _push(4)
    s.tick(win)
    app.processEvents()

    cur = win.view.textCursor()
    cur.setPosition(0)
    cur.movePosition(cur.MoveOperation.NextWord,
                     cur.MoveMode.KeepAnchor, 3)
    win.view.setTextCursor(cur)
    held = win.view.textCursor().selectedText()
    assert held, "test set up no selection"
    assert win.has_selection()

    _push(3)                      # speech keeps arriving while they hold it
    s.tick(win)
    app.processEvents()
    assert win.view.textCursor().selectedText() == held
    assert s._dirty, "deferred, not dropped"

    # Deselecting catches up on everything that arrived meanwhile.
    cur.clearSelection()
    win.view.setTextCursor(cur)
    s.tick(win)
    app.processEvents()
    assert not s._dirty
    assert "this is line 2" in win.view.toPlainText()  # the lines held back


def test_no_selection_still_repaints(sink):
    s, win, app = sink
    _push(2)
    s.tick(win)
    app.processEvents()
    assert "this is line 0" in win.view.toPlainText()
    assert not s._dirty


def _click_toggle(view, qapp) -> bool:
    """Click the card's breakdown toggle the way a reader does.

    Synthetic, but through the real event path: the bug is in what Qt does to
    the selection on a link click, which no direct call to the handler shows.
    """
    from PyQt6 import QtCore
    from PyQt6.QtTest import QTest

    lay = view.document().documentLayout()
    off = view.verticalScrollBar().value()
    for y in range(0, view.viewport().height(), 2):
        for x in range(10, 400, 4):
            if lay.anchorAt(QtCore.QPointF(x, y + off)).startswith(TOK_SCHEME):
                QTest.mouseClick(view.viewport(),
                                 QtCore.Qt.MouseButton.LeftButton,
                                 QtCore.Qt.KeyboardModifier.NoModifier,
                                 QtCore.QPoint(x, y))
                qapp.processEvents()
                return True
    return False


@pytest.mark.parametrize("pane", ["view", "live"])
def test_opening_a_breakdown_is_not_a_selection(sink, pane):
    """Qt selects a link's own text when it is clicked — one click on the toggle
    left "▸ 2 words" selected. The freeze read that as the reader holding a
    selection and stopped rebuilding, so the breakdown stayed shut, with
    "paused — text selected" under it, until they clicked somewhere else."""
    s, win, app = sink
    toks = [Token(surface="水", base_form="水", romaji="mizu", gloss="water"),
            Token(surface="飲む", base_form="飲む", romaji="nomu", gloss="to drink")]
    for jp in ("水を飲む。", "そうだね。"):
        ui_q.put(("new", Segment.new(stream="main", original=jp, romaji="r",
                                     english="e", partial=False,
                                     tokens=list(toks))))
    s.tick(win)
    app.processEvents()

    view = getattr(win, pane)
    assert _click_toggle(view, app), f"no toggle anchor found in {pane}"
    s.tick(win)
    app.processEvents()

    assert not win.has_selection()
    assert "paused" not in win.status.currentMessage()
    assert "water" in view.toPlainText(), "breakdown did not open on the first click"


def test_clear_does_not_come_back(sink):
    """view.clear() alone left `order` intact and the next line restored it all."""
    s, win, app = sink
    _push(3)
    s.tick(win)
    app.processEvents()

    win._clear()
    s.tick(win)
    app.processEvents()
    assert "this is line 0" not in win.view.toPlainText()
    assert "Listening" in win.view.toPlainText()   # back to the empty state

    _push(1)                      # and it stays gone once speech resumes
    s.tick(win)
    app.processEvents()
    assert "this is line 0" not in win.view.toPlainText()


# --- 2. the live lane -----------------------------------------------------

def test_live_lane_holds_the_newest_and_history_does_not(sink):
    s, win, app = sink
    _push(3)
    s.tick(win)
    app.processEvents()

    assert win.live.isVisible()
    assert "this is line 2" in win.live.toPlainText()
    assert "this is line 2" not in win.view.toPlainText()
    assert "this is line 1" in win.view.toPlainText()

    _push(1)                      # the old newest hands off to the history
    s.tick(win)
    app.processEvents()
    assert "this is line 2" in win.view.toPlainText()
    assert "this is line 2" not in win.live.toPlainText()


def test_live_lane_height_is_stable_and_clamped(sink):
    """It must not resize under the reader on an ordinary line, and an opened
    breakdown must not be allowed to eat the window."""
    s, win, app = sink
    short, = _push(1)
    s.tick(win)
    app.processEvents()
    first = win.live.height()
    assert LIVE_MIN_PX <= first <= LIVE_MAX_PX

    ui_q.put(("english", short))   # same line, filled in further
    s.tick(win)
    app.processEvents()
    assert win.live.height() == first

    huge = Segment.new(stream="main", original=JP * 6, english="x " * 400,
                       partial=False)
    ui_q.put(("new", huge))
    s.tick(win)
    app.processEvents()
    assert win.live.height() == LIVE_MAX_PX
    # Clamped, so the overflow has to remain reachable.
    assert win.live.verticalScrollBar().maximum() > 0


def test_live_lane_hidden_until_the_first_line(sink):
    s, win, app = sink
    assert not win.live.isVisible()
    _push(1)
    s.tick(win)
    app.processEvents()
    assert win.live.isVisible()


# --- 3. low confidence is visible ----------------------------------------

def test_untrusted_only_fires_where_the_score_means_something():
    cfg = UiCfg()                          # floor -0.6
    assert _untrusted(Segment.new(stream="m", original=JP, confidence=-0.9), cfg)
    assert not _untrusted(Segment.new(stream="m", original=JP, confidence=-0.2), cfg)
    # 0.0 is "the engine reported nothing" — qwen3 never reports a logprob, and
    # tagging every one of its lines would make the tag meaningless.
    assert not _untrusted(Segment.new(stream="m", original=JP, confidence=0.0), cfg)
    # A prefix of speech still in progress scores worse as a class.
    assert not _untrusted(
        Segment.new(stream="m", original=JP, confidence=-0.9, provisional=True), cfg)


def test_untrusted_line_renders_dimmed_and_tagged(qapp):
    """Checked in the laid-out document, not the markup: the dimming rides on a
    multi-class selector (`.jp.shaky`) and Qt's CSS subset is the thing that has
    to honour it."""
    from PyQt6 import QtGui

    from tsutawaru.ui.window_qt import _CSS

    cfg = UiCfg()

    def colour_of(text: str, html: str) -> str:
        doc = QtGui.QTextDocument()
        doc.setDefaultStyleSheet(_CSS)
        doc.setTextWidth(640)
        doc.setHtml(f"<body>{html}</body>")
        block = doc.begin()
        while block.isValid():
            if text in block.text():
                it = block.begin()
                frag = it.fragment()
                return frag.charFormat().foreground().color().name()
            block = block.next()
        raise AssertionError(f"{text!r} not in the document")

    clean = _fmt_block(Segment.new(stream="m", original=JP, english="a movie",
                                   confidence=-0.2), cfg)
    shaky = _fmt_block(Segment.new(stream="m", original=JP, english="a movie",
                                   confidence=-0.9), cfg)

    assert "unsure" in shaky and "unsure" not in clean
    assert colour_of(JP, shaky) != colour_of(JP, clean)
    assert colour_of("a movie", shaky) != colour_of("a movie", clean)
    # Still the 19px JP face — the override must not cost the tier its typography.
    doc = QtGui.QTextDocument()
    doc.setDefaultStyleSheet(_CSS)
    doc.setTextWidth(640)
    doc.setHtml(f"<body>{shaky}</body>")
    block = doc.begin()
    while block.isValid():
        if JP in block.text():
            assert block.begin().fragment().charFormat().font().pixelSize() == 19
            break
        block = block.next()


def _live_rule(prop: str, selector: str = ".cbody") -> str:
    """Read a declaration straight out of _LIVE_CSS.

    The expected values come from the sheet rather than from literals here, so
    retuning the lane does not mean editing the test in two places. What is
    being asserted is that Qt *applies* them, which no amount of reading the
    sheet can show.
    """
    from tsutawaru.ui.window_qt import _LIVE_CSS

    body = _LIVE_CSS.split(selector, 1)[1].split("}", 1)[0]
    return body.split(prop + ":", 1)[1].split(";", 1)[0].strip()


def test_live_card_is_a_different_dark_than_the_history_cards(sink):
    """The live lane is styled by a second sheet, not by a flag on the renderer.

    Both panes are handed the same card HTML, so the only thing that can make
    the newest line read as the active one is `_LIVE_CSS` winning the cascade in
    that one document. Asserted on pixels because that cascade is Qt's, not
    ours: an equally specific earlier rule, or a leftover bgcolor attribute,
    would leave both panes identical and every string assertion would pass.
    """
    from PyQt6 import QtGui

    s, win, app = sink
    _push(3)
    s.tick(win)
    app.processEvents()

    def colours(widget):
        img = widget.grab().toImage()
        return {img.pixelColor(x, y).name()
                for y in range(0, img.height(), 3)
                for x in range(0, img.width(), 7)}

    card = _live_rule("background-color")
    history, live = colours(win.view), colours(win.live)
    assert "#171a21" in history and "#171a21" not in live
    assert card in live and card not in history
    # Brighter than the history card, and still dark — a lift, not a light card.
    lit, base = QtGui.QColor(card), QtGui.QColor("#171a21")
    assert lit.lightness() > base.lightness()
    assert max(lit.getRgb()[:3]) < 70


def test_live_card_sets_its_type_larger_without_losing_the_faces(sink):
    """Bigger is the second axis, and it is the one that can quietly break.

    The base sheet restates font-family on .jp precisely because inheritance
    stops resolving it once the Latin rule exists — so an override that touched
    the family would drop the Japanese to a generic sans at the larger size.
    This pins both: every live tier is larger than its history counterpart, and
    every face is unchanged.
    """
    from PyQt6 import QtGui

    from tsutawaru.ui.window_qt import _CSS, _LIVE_CSS

    seg = Segment.new(stream="m", original=JP, romaji="kinou no yoru",
                      english="a movie", confidence=-0.2)
    card = _fmt_block(seg, UiCfg())

    def tiers(sheet):
        doc = QtGui.QTextDocument()
        doc.setDefaultStyleSheet(sheet)
        doc.setTextWidth(640)
        doc.setHtml(f"<body>{card}</body>")
        out, block = {}, doc.begin()
        while block.isValid():
            if block.text().strip():
                f = block.begin().fragment().charFormat().font()
                out[block.text()[:12]] = (f.pixelSize(), f.family())
            block = block.next()
        return out

    base, live = tiers(_CSS), tiers(_CSS + _LIVE_CSS)
    assert base and base.keys() == live.keys()
    for key, (size, face) in base.items():
        assert live[key][0] > size, f"{key!r} did not grow"
        assert live[key][1] == face, f"{key!r} changed face"
    # The Japanese keeps the CJK family specifically — the bug the sheet records.
    assert live[JP[:12]][1] == "Hiragino Sans"
    assert "font-family" not in _LIVE_CSS


def _rail_colour(widget) -> set[str]:
    """Every colour in the strip where the 4px accent rail is drawn.

    A column scan rather than a fixed x: the rail sits behind the view's own
    12px padding and the document's left margin, and pinning the exact pixel
    would make this test a measurement of Qt's box model instead of the rail.
    """
    img = widget.grab().toImage()
    return {img.pixelColor(x, y).name()
            for x in range(0, 30)
            for y in range(0, img.height())}


def test_the_rail_says_which_pane_as_well_as_which_state(sink):
    """Settled-and-live and settled-in-the-scrollback are different colours.

    Both were RAIL_FINAL blue, so the one axis the rail carried was partial vs
    settled. The pane is now a hue and the state is still the value — and the
    pending grey is deliberately NOT overridden, because "not finished yet"
    means the same thing wherever it appears.
    """
    from tsutawaru.ui.window_qt import RAIL_FINAL, RAIL_LIVE, RAIL_PENDING

    s, win, app = sink
    _push(3)
    s.tick(win)
    app.processEvents()

    history, live = _rail_colour(win.view), _rail_colour(win.live)
    assert RAIL_FINAL in history and RAIL_FINAL not in live
    assert RAIL_LIVE in live and RAIL_LIVE not in history

    # A line still being spoken keeps the neutral rail in the live lane.
    seg = Segment.new(stream="main", original=JP, romaji="r", english="",
                      partial=True)
    ui_q.put(("new", seg))
    s.tick(win)
    app.processEvents()
    pending = _rail_colour(win.live)
    assert RAIL_PENDING in pending
    assert RAIL_LIVE not in pending and RAIL_FINAL not in pending
