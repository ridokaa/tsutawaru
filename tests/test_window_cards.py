"""The transcript card, checked where it actually failed: in pixels.

The markup this replaces asked for a card — a div with a background, a rounded
corner and a coloured left border — and QTextBrowser drew none of it, because it
paints a block element's background per *paragraph* and drops borders on divs.
Reading the HTML would have said the card was there; only rendering it says
otherwise. So these assertions run a real QTextDocument and sample the result:
the longest unbroken run of card background down a column was 11px of a 124px
block before this change, and a header assertion could not have caught that.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6")

from PyQt6 import QtGui  # noqa: E402
from PyQt6.QtWidgets import QApplication, QTextBrowser  # noqa: E402

from tsutawaru.config import UiCfg  # noqa: E402
from tsutawaru.models import Segment  # noqa: E402
from tsutawaru.ui.window_qt import (  # noqa: E402
    _CSS, MAX_TEXT_PX, RAIL_FINAL, RAIL_PENDING, WindowSink, _CappedView,
    _divider, _fmt_block,
)

CARD_BG = "#171a21"
PAGE_BG = "#0f1115"
WIDTH = 700


@pytest.fixture(scope="module")
def qapp():
    yield QApplication.instance() or QApplication([])


def _render(html: str) -> QtGui.QImage:
    doc = QtGui.QTextDocument()
    doc.setDefaultStyleSheet(_CSS)
    doc.setTextWidth(WIDTH)
    doc.setHtml(html)
    img = QtGui.QImage(WIDTH, max(1, int(doc.size().height())),
                       QtGui.QImage.Format.Format_RGB32)
    img.fill(QtGui.QColor("#0f1115"))
    painter = QtGui.QPainter(img)
    doc.drawContents(painter)
    painter.end()
    return img


def _column(img: QtGui.QImage, x: int) -> list[str]:
    return [img.pixelColor(x, y).name() for y in range(img.height())]


def _longest_run(colours: list[str], want: str) -> int:
    best = run = 0
    for c in colours:
        run = run + 1 if c == want else 0
        best = max(best, run)
    return best


def _seg(**kw) -> Segment:
    kw.setdefault("stream", "YouTube (Chrome)")
    kw.setdefault("partial", False)  # Segment defaults to True — that is the pending rail
    kw.setdefault("original", "やっぱり明日にしておこうかな。")
    kw.setdefault("romaji", "yappari ashita ni shite okou kana")
    kw.setdefault("english", "I think I'll leave it until tomorrow after all.")
    return Segment.new(**kw)


def test_card_paints_as_one_box_not_a_stack_of_stripes(qapp):
    """The whole point: no page showing through between the tiers.

    Measured as "is the page background visible inside the card", not as a run
    of card colour: a striped card leaks #0f1115 into the margins between tiers,
    a real box never does. Sampled in the cell's right-hand padding, which no
    glyph can reach — a column through the text reads antialiasing, and one of
    those intermediate greys lands on #0f1115 by coincidence.
    """
    img = _render(_fmt_block(_seg(), UiCfg()))
    rail = _column(img, 5)
    rows = [y for y, c in enumerate(rail) if c == RAIL_FINAL]
    assert rows, "no rail drawn at all — the card table did not render"

    padding = _column(img, WIDTH - 8)
    leaked = [y for y in range(rows[0], rows[-1] + 1) if padding[y] == PAGE_BG]
    assert not leaked, (
        f"page background visible inside the card on {len(leaked)} row(s) "
        f"({leaked[:5]}…) — the tiers are separate stripes, not one box"
    )


def test_rail_colour_says_whether_the_line_is_settled(qapp):
    """A provisional line is greyed at the rail, a final one accented."""
    for partial, want in ((False, RAIL_FINAL), (True, RAIL_PENDING)):
        img = _render(_fmt_block(_seg(partial=partial), UiCfg()))
        rail = _column(img, 5)  # the 4px rail cell, past the table's left inset
        assert _longest_run(rail, want) > 40, (
            f"partial={partial}: no {want} rail; column was {set(rail)}"
        )


def test_divider_prints_only_where_the_source_changes(qapp):
    """One label per boundary, not one per line — including the very first."""
    sink = WindowSink.__new__(WindowSink)  # no Qt, no queues: only the join logic
    sink.cfg = UiCfg()
    sink._expanded = set()
    sink.segments, sink.order = {}, []
    for stream in ("YouTube (Chrome)", "YouTube (Chrome)", "Discord", "Discord"):
        seg = _seg(stream=stream)
        sink.segments[seg.id] = seg
        sink.order.append(seg.id)

    html = sink._transcript_html()
    assert html.count(_divider("YouTube (Chrome)")) == 1
    assert html.count(_divider("Discord")) == 1
    # And the one that changed is the one that prints second.
    assert html.index(_divider("Discord")) > html.index(_divider("YouTube (Chrome)"))


def test_divider_label_does_not_render_one_letter_per_line(qapp):
    """A two-cell divider squeezed the label to a column of single letters.

    Caught by eye, kept as a check: the label must be wider than it is tall.
    """
    img = _render(_divider("YouTube (Chrome)"))
    assert img.width() / max(1, img.height()) > 4, (
        f"divider is {img.width()}x{img.height()} — the label is stacked vertically"
    )


# ------------------------------------------------------- line length and face

# Long enough to wrap at any sane column width. The measurements in the plan were
# taken on this sentence.
LONG_EN = (
    "Ah, by the way, I looked up the opening times earlier, and it turns out "
    "they close at six, so I think we should probably set off a good while "
    "before lunch tomorrow."
)
LONG_JP = "あ、そういえばさ、さっき調べたんだけど六時で閉まるらしいから、明日は昼前に出た方がいいんだろうけど"
# Where comfortable reading stops is about 90; this allows a little past it,
# because MAX_TEXT_PX was widened from 620 to 660 on request and that costs
# three characters. The guard is still worth having: it catches the cap being
# lost altogether, which is 117 characters at the default window and 161
# maximised — not a drift of three.
MAX_CHARS = 95


def _laid_out(text_width: int):
    """A laid-out document of one card, and a way to ask Qt about its tiers."""
    doc = QtGui.QTextDocument()
    doc.setDefaultStyleSheet(_CSS)
    doc.setTextWidth(text_width)
    doc.setHtml(_fmt_block(_seg(original=LONG_JP, english=LONG_EN), UiCfg()))
    doc.documentLayout().documentSize()  # force layout; lineCount is 0 until it runs
    return doc


def _tier(doc, starts_with: str):
    block = doc.begin()
    while block.isValid():
        if block.text().strip().startswith(starts_with):
            return block
        block = block.next()
    raise AssertionError(f"no block starting {starts_with!r}")


def test_english_lines_stay_inside_the_reading_width(qapp):
    """The cap's whole purpose, measured in characters rather than pixels."""
    # What the view will hand the document once the cap is in force: the viewport
    # less the QTextBrowser's own 12px padding on each side.
    doc = _laid_out(MAX_TEXT_PX - 24)
    layout = _tier(doc, "Ah,").layout()
    lines = [layout.lineAt(i).textLength() for i in range(layout.lineCount())]
    assert lines, "the English tier did not lay out at all"
    assert max(lines) <= MAX_CHARS, f"English wraps at {max(lines)} characters: {lines}"


def test_the_cap_holds_a_wide_window_and_yields_to_a_narrow_one(qapp):
    """Wider than the cap: margin grows. Narrower: behave as before.

    Also the recursion guard — setViewportMargins raises resizeEvent, and the
    unguarded version blew the stack.
    """
    view, plain = _CappedView(), QTextBrowser()
    for w in (view, plain):
        w.show()  # an unshown top-level ignores resize() on the offscreen platform
    for window_w in (500, 1400, 860, 500):  # last repeat: the cap must release again
        for w in (view, plain):
            w.resize(window_w, 600)
        qapp.processEvents()
        natural = plain.viewport().width()  # what an uncapped view would have given
        assert view.viewport().width() == min(natural, MAX_TEXT_PX), (
            f"window {window_w}: viewport {view.viewport().width()}, "
            f"uncapped {natural}, cap {MAX_TEXT_PX}"
        )


def test_latin_tiers_are_not_set_in_the_japanese_face(qapp):
    """The English was inheriting Hiragino Sans from body — a CJK face.

    Asserted via QFontInfo, which reports what Qt *resolved*, not what the CSS
    asked for: the family name in the stylesheet was never the problem.
    """
    doc = _laid_out(MAX_TEXT_PX - 24)

    def family(starts_with: str) -> str:
        fmt = _tier(doc, starts_with).begin().fragment().charFormat()
        return QtGui.QFontInfo(fmt.font()).family()

    assert family("Ah,") != "Hiragino Sans", "English is still in the Japanese face"
    assert family("あ、") == "Hiragino Sans", "Japanese tier lost its face"


# --------------------------------------------------- the stranded provisional

def test_a_stranded_provisional_says_so_instead_of_waiting_forever(qapp):
    """`utt_q` is drop-oldest, so a closing utterance can be evicted and the
    preview left on screen for good. It used to render an eternal "…", which is
    the same thing a line still in flight renders — and it was never written to
    --log-file, because the window waits for lanes that will never report.
    """
    from tsutawaru.pipeline.orchestrator import Orchestrator
    from tsutawaru.pipeline.queues import ui_q
    from tsutawaru.stt.filters import DROPS

    cfg = UiCfg()
    live = Segment.new(stream="t", original="こんにちは", romaji="konnichiwa",
                       provisional=True)
    assert 'class="waiting"' in _fmt_block(live, cfg), "a live preview waits"

    # Strand it exactly as the stale sweep in _show_provisional does.
    orch = Orchestrator.__new__(Orchestrator)  # no audio devices wanted here
    before = DROPS["stranded"]
    while not ui_q.empty():
        ui_q.get_nowait()
    orch._strand(live)

    assert live.dropped and not live.provisional and not live.partial
    assert DROPS["stranded"] == before + 1, "not counted for --stats"
    assert ui_q.get_nowait()[0] == "dropped", "the window is never told"

    html = _fmt_block(live, cfg)
    assert 'class="lost"' in html and "dropped" in html
    assert 'class="waiting"' not in html.split('class="en"')[1], \
        "the English tier still reads as pending"

    # Idempotent: a second sweep must not double-count a line already stranded.
    orch._strand(live)
    assert DROPS["stranded"] == before + 1


def test_a_stranded_line_still_reaches_the_log_file(qapp, tmp_path):
    """The second half of the same bug: `_expect` waits for `english`, which the
    sentence lane never sends for a provisional, so the line vanished."""
    from tsutawaru.pipeline.queues import ui_q

    log = tmp_path / "t.log"
    sink = WindowSink.__new__(WindowSink)
    sink.cfg = UiCfg(log_file=str(log))
    sink.ws_sink = None
    sink.segments, sink.order = {}, []
    sink._expanded, sink._backfilled, sink._logged = set(), set(), set()
    sink._dirty, sink._seen_total = False, 0
    sink._expect, sink._seen = {"breakdown", "english"}, {}

    seg = Segment.new(stream="t", original="ねこ", romaji="neko", dropped=True,
                      partial=False)
    while not ui_q.empty():
        ui_q.get_nowait()
    ui_q.put(("dropped", seg))
    sink._drain()

    assert seg.id in sink._logged, "a dropped line never got written"
    assert "ねこ" in log.read_text(encoding="utf-8")
