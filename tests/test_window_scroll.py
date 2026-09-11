"""Transcript view must hold the reader's place across updates.

Regression test for the scroll drift reported after the 2026-08-22 session:
the view crept upward on every new translation and would not stay at the bottom
when scrolled back down. Cause was reading verticalScrollBar().maximum()
immediately after setHtml(), before rich-text layout had run — the error
compounded (13, 15, 26, 47, 81px over five lines) until it exceeded
STICKY_BOTTOM_PX and the view stopped following new output entirely.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6")

from tsutawaru.config import UiCfg  # noqa: E402
from tsutawaru.models import Segment  # noqa: E402
from tsutawaru.ui.window_qt import TranscriptWindow, _fmt_block  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    from PyQt6 import QtWidgets

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


@pytest.fixture
def win(qapp):
    w = TranscriptWindow(UiCfg())
    w.resize(500, 300)
    w.show()
    qapp.processEvents()
    yield w
    w.close()


def _body(n: int) -> str:
    return "".join(
        _fmt_block(
            Segment.new(stream="main", original=f"これはテスト行{i}です、長めの文章にしておきます",
                        english=f"this is line {i}"),
            UiCfg(), False,
        )
        for i in range(n)
    )


def test_stays_exactly_at_bottom_while_streaming(win, qapp):
    win.set_html(_body(20), True)
    qapp.processEvents()
    sb = win.view.verticalScrollBar()

    worst = 0
    for n in range(21, 40):
        win.set_html(_body(n), True)
        qapp.processEvents()
        worst = max(worst, sb.maximum() - sb.value())

    assert worst == 0, f"drifted {worst}px off the bottom"
    assert win.at_bottom()


def test_holds_position_when_scrolled_up(win, qapp):
    win.set_html(_body(30), True)
    qapp.processEvents()
    sb = win.view.verticalScrollBar()

    sb.setValue(sb.maximum() // 2)
    qapp.processEvents()
    held = sb.value()
    assert win._pin_bottom is False, "manual scroll must disarm the pin"

    for n in range(31, 37):
        win.set_html(_body(n), False)
        qapp.processEvents()

    assert sb.value() == held, "reader's position moved under them"


def test_rearms_when_scrolled_back_to_bottom(win, qapp):
    win.set_html(_body(30), True)
    qapp.processEvents()
    sb = win.view.verticalScrollBar()

    sb.setValue(sb.maximum() // 2)
    qapp.processEvents()
    assert win._pin_bottom is False

    sb.setValue(sb.maximum())  # reader scrolls back down
    qapp.processEvents()
    assert win._pin_bottom is True, "returning to the bottom must re-arm sticking"

    for n in range(31, 41):
        win.set_html(_body(n), win.at_bottom())
        qapp.processEvents()

    assert sb.value() == sb.maximum()


# --------------------------------------------------------------------------
# Regression: the document rebuild must not be able to rewrite _pin_bottom.
#
# Reported 2026-08-24 — following the tail, the view would sometimes hold,
# sometimes fall a line behind, sometimes several. The three tests above did not
# catch it and cannot: they assert on final scroll positions, which under
# QT_QPA_PLATFORM=offscreen settle synchronously. On a real display layout is
# incremental, and setHtml() emits valueChanged throughout the teardown and
# rebuild — measured on a single update: value 0 against maximum 0, then 4039,
# then 4, then the real position. Each reached _on_scrolled, which recomputed
# `_pin_bottom` from at_bottom() on a document mid-rebuild. 0/0 reads as "at the
# bottom" and armed the pin under a reader who had scrolled up; 4/4039 reads as
# the opposite and disarmed it under one following the tail. Which landed last
# raced with layout, hence the intermittency.
#
# So these assert on the *invariant* — no unguarded event reaches _on_scrolled
# during a rebuild — rather than on a settled position, and therefore do not
# depend on the platform's layout timing to have any teeth.
# --------------------------------------------------------------------------


def test_rebuild_emits_no_unguarded_scroll_events(win, qapp):
    win.set_html(_body(30), True)
    qapp.processEvents()
    sb = win.view.verticalScrollBar()

    sb.setValue(sb.maximum() // 2)  # reader scrolls up to read back
    qapp.processEvents()
    assert win._pin_bottom is False

    leaked: list[tuple[int, int]] = []
    sb.valueChanged.connect(
        lambda v: None if win._programmatic_scroll else leaked.append((v, sb.maximum()))
    )

    win.set_html(_body(31), win.at_bottom())
    qapp.processEvents()

    assert leaked == [], (
        "document rebuild leaked scroll events to _on_scrolled "
        f"(value, maximum): {leaked} — each one rewrites _pin_bottom from a "
        "document that is still being laid out"
    )
    assert win._pin_bottom is False, "reader's scroll-up was silently discarded"


def test_set_scroll_does_not_clear_an_outer_guard(win, qapp):
    """_set_scroll must save and restore the flag, not clear it.

    setHtml() runs under the guard, and the layout pass it triggers calls
    _on_doc_resized -> _set_scroll. Clearing unconditionally would release the
    outer guard partway through the rebuild and let the remaining transients
    through — the same defect, reachable by a different route.
    """
    win.set_html(_body(20), True)
    qapp.processEvents()

    win._programmatic_scroll = True  # stand in for the setHtml guard
    try:
        win._set_scroll(10)
        assert win._programmatic_scroll is True, (
            "_set_scroll cleared a guard it did not set"
        )
    finally:
        win._programmatic_scroll = False


# --------------------------------------------------------------------------
# Regression: past `max_lines` the transcript becomes a sliding window — one
# block dropped off the top for every one appended at the bottom. The document
# height therefore does not change, so restoring the previous scrollbar *value*
# looked flawless: the bar never moved. The text moved instead, one line per
# utterance, sliding out from under anyone reading back. Confirmed live at ~220
# lines on 2026-08-24, at exactly the predicted one line per new translation.
#
# Nothing above catches it, because every assertion up there is on a scrollbar
# number and the scrollbar number is the one thing that stayed correct. These
# assert on the reader's *content* position instead.
# --------------------------------------------------------------------------


def _stream(win, qapp, segs, cfg, n):
    """Append n more lines, rendering the trailing max_lines window each time."""
    for _ in range(n):
        segs.append(
            Segment.new(stream="main", original="これはテスト行です、長めの文章にしておきます",
                        english="another line of transcript")
        )
        body = "".join(_fmt_block(s, cfg, False) for s in segs[-cfg.max_lines:])
        win.set_html(body, win.at_bottom())
        qapp.processEvents()


def test_reader_holds_their_line_while_the_buffer_trims(win, qapp):
    cfg = UiCfg()
    segs: list[Segment] = []
    _stream(win, qapp, segs, cfg, cfg.max_lines)  # fill to the cap, no trimming yet

    sb = win.view.verticalScrollBar()
    sb.setValue(sb.maximum() // 2)
    qapp.processEvents()
    assert win._pin_bottom is False

    watched, _ = win._anchor_at_top()
    assert watched, "no segment anchor found at the viewport top"
    before = sb.value() - win._anchor_offset(watched)

    _stream(win, qapp, segs, cfg, 60)  # every one of these trims a line off the top

    offset = win._anchor_offset(watched)
    assert offset is not None, "watched line was trimmed too early to judge"
    after = sb.value() - offset
    # Whole-pixel rounding in the anchor arithmetic can settle a few px off; a
    # regression here moves by a line height (~138px) *per update*, not 6px total.
    assert abs(after - before) <= 12, (
        f"reader's line drifted {after - before}px across 60 trims "
        f"(was {before}px above the viewport top, now {after}px)"
    )


def test_tail_following_survives_the_trim_boundary(win, qapp):
    cfg = UiCfg()
    segs: list[Segment] = []
    _stream(win, qapp, segs, cfg, cfg.max_lines + 60)

    sb = win.view.verticalScrollBar()
    assert win._pin_bottom is True, "following the tail must survive trimming"
    assert sb.value() == sb.maximum(), "fell behind the tail across the cap"
