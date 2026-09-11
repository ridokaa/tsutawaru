"""tsutawaru — dedicated transcript window (PyQt6).

A normal, titled, resizable window rather than the frameless streaming overlay
in `overlay_qt.py`. Use this when you want to *read* the transcript; use the
overlay when you want to composite it over a game or a stream.

Threading (plan §6a): Qt must own the main thread. The pipeline runs in worker
threads and never touches a widget. Instead of marshalling every update through
a signal, the Qt thread polls `ui_q` on a QTimer — the queue is already
thread-safe, so this is both simpler and impossible to get wrong.

Rendering is throttled and batched: a burst of patch events coalesces into one
repaint, and the document is only rebuilt when something actually changed.
"""
from __future__ import annotations

import html
import queue
import threading
from typing import Optional

from tsutawaru.config import UiCfg
from tsutawaru.logbus import get_logger, metrics
from tsutawaru.models import Segment
from tsutawaru.pipeline.queues import ui_q

log = get_logger(__name__)

WINDOW_TITLE = "tsutawaru"
POLL_MS = 100          # how often the Qt thread drains ui_q
STICKY_BOTTOM_PX = 40  # treat "within 40px of the end" as pinned to the bottom
# href prefix for the per-word breakdown toggle. Not a real scheme: it never
# leaves the widget, because setOpenLinks(False) routes every click to us.
TOK_SCHEME = "tok:"
# Name prefix for the per-segment scroll anchor. A reader who has scrolled up is
# holding a *line*, not a pixel offset, and the two stop meaning the same thing
# as soon as the transcript is trimmed — see set_html().
ANCHOR_PREFIX = "seg"

_CSS = """
body { background:#0f1115; color:#e8e8ea;
       font-family:'Hiragino Sans','Yu Gothic UI','Noto Sans JP',sans-serif; }
.blk    { margin:0 0 14px 0; padding:10px 12px; background:#171a21;
          border-left:3px solid #3d7dff; border-radius:6px; }
.blk.pending { border-left-color:#5a6172; }
.stream { color:#7d8595; font-size:11px; text-transform:uppercase;
          letter-spacing:.08em; margin-bottom:6px; }
.jp     { font-size:19px; line-height:1.5; margin-bottom:4px; }
.romaji { color:#9aa4b8; font-size:13px; font-style:italic; margin-bottom:6px; }
.en     { color:#8fe3a0; font-size:15px; line-height:1.45; margin-bottom:8px; }
.tokhdr { margin-top:2px; }
.toktoggle { color:#7d8595; font-size:11.5px; text-decoration:none;
             letter-spacing:.04em; }
.tok    { color:#c8cddb; font-size:12.5px; line-height:1.7; margin-top:6px; }
.tsurf  { color:#e8e8ea; }
.trom   { color:#7d8595; }
.tgloss { color:#e0b978; }
.tinfl  { color:#7f9bd0; font-size:11.5px; }
.waiting{ color:#5a6172; }
/* The A/B lane, two columns of equal weight. A <table> because QTextBrowser
   renders a subset of HTML 4 with no flex layout, but solid table support.
   Equal weight is deliberate: dimming the comparison biases reading toward the
   primary on the lines where they disagree — the only lines that matter. */
.abname { color:#7d8595; font-size:10.5px; text-transform:uppercase;
          letter-spacing:.08em; padding-bottom:4px; }
.abname.b { color:#c58f6a; }
.abcell { padding:0 10px 0 0; }
.abcell.b { padding:0 0 0 12px; border-left:2px solid #2b303b; }
.empty   { padding:22px 4px; }
.etitle  { color:#8b93a5; font-size:15px; font-weight:600; margin-bottom:6px; }
.esub    { color:#5a6172; font-size:13px; line-height:1.5; margin-bottom:16px; }
.ehint   { color:#4a5060; font-size:12px; line-height:1.5;
           border-left:2px solid #2a2f3a; padding-left:10px; }
.empty b { color:#7d8595; font-weight:600; }
"""

# Control strip (source switcher). Plain Qt stylesheet, not document CSS.
_BAR_QSS = """
QWidget#srcbar { background:#141720; border-top:1px solid #242936; }
QLabel#srclabel { color:#5a6172; font-size:11px; text-transform:uppercase; }
QPushButton#src {
    background:#1b1f2a; color:#9aa4b8; border:1px solid #2a2f3a;
    border-radius:5px; padding:5px 14px; font-size:12px;
}
QPushButton#src:hover { background:#222735; color:#c8cddb; }
QPushButton#src:checked {
    background:#3d7dff; color:#ffffff; border-color:#3d7dff; font-weight:600;
}
QPushButton#src:disabled { color:#4a5060; border-color:#20242e; }
"""


def _empty_state(device: str, source: str = "") -> str:
    """Shown before the first line lands, so an idle window doesn't read as broken.

    The hint has to match the backend actually running, because the two fail in
    opposite ways and the wrong advice costs more than none. `source` is the
    discriminator and needs no plumbing: the orchestrator only passes a source
    to this sink on the process backend, so a non-empty value *is* "we are
    tapping an app", and an empty one means the device path.

      device path  -> the system output moved out from under the loopback,
                      which macOS does on every headphone plug/unplug.
      process path -> that cannot happen, the tap follows the app. What bites
                      instead is an app with its own output picker pointed
                      somewhere other than the system default: still perfectly
                      audible, captured as pure silence.
    """
    dev = html.escape(device or "…")
    if source:
        hint = (
            f"Nothing yet? A tap only captures while {dev} is actually playing, "
            f"so silence here is normal between turns. If {dev} <i>is</i> audible "
            "and nothing appears, its output is set to a different device than "
            "your system output — apps with their own output picker sit outside "
            "the tap and record silence."
        )
    else:
        hint = (
            "Nothing showing up? macOS switches your system output away whenever "
            "you plug or unplug headphones — reselect your Multi-Output Device "
            "from the menu bar sound icon."
        )
    return (
        '<div class="empty">'
        '<div class="etitle">Listening…</div>'
        f'<div class="esub">Capturing from <b>{dev}</b>. '
        "Japanese speech will appear here.</div>"
        f'<div class="ehint">{hint}</div>'
        "</div>"
    )


def _tiers(jp: str, romaji: str, english: str, cfg: UiCfg) -> str:
    """The JP / romaji / English tiers. One column, or one side of a comparison.

    Shared so the two sides of an A/B are identical markup by construction.
    """
    esc = html.escape
    wait = '<span class="waiting">…</span>'
    out = [f'<div class="jp">{esc(jp) if jp else wait}</div>']
    if cfg.show_romaji:
        out.append(f'<div class="romaji">{esc(romaji) if romaji else wait}</div>')
    out.append(f'<div class="en">{esc(english) if english else wait}</div>')
    return "".join(out)


def _ab_columns(seg: Segment, cfg: UiCfg) -> str:
    """The two models side by side, each with its own JP / romaji / English.

    The breakdown tier stays outside and full width: it is built from the
    primary transcript only, and half-width would make the hardest tier to
    read harder still.
    """
    esc = html.escape
    left = _tiers(seg.original, seg.romaji, seg.english, cfg)
    right = _tiers(seg.alt_original, seg.alt_romaji, seg.alt_english, cfg)
    # `model` is only set while comparing, so fall back to the stream name
    # rather than printing an empty header over the live transcript.
    a_name = esc(seg.model or seg.stream)
    b_name = esc(seg.alt_model)
    return (
        '<table width="100%" cellspacing="0" cellpadding="0"><tr>'
        f'<td width="50%" valign="top" class="abcell">'
        f'<div class="abname">{a_name}</div>{left}</td>'
        f'<td width="50%" valign="top" class="abcell b">'
        f'<div class="abname b">{b_name}</div>{right}</td>'
        "</tr></table>"
    )


def _fmt_block(seg: Segment, cfg: UiCfg, expanded: bool = False) -> str:
    """Render one segment. `expanded` controls the per-word breakdown.

    The breakdown is collapsed by default: on a long sentence it pushed the
    English translation off the top of the view, so reading a line meant
    scrolling back up for it. It is now behind a click, and the header states
    the word count so there is a reason to open it.
    """
    esc = html.escape
    pending = " pending" if seg.partial else ""
    out = [f'<div class="blk{pending}">']
    # Scroll anchor. Carries no href, so QTextBrowser does not treat it as a
    # hyperlink and the block's own CSS keeps winning — verified: the character
    # it attaches to stays #e8e8ea with no underline.
    out.append(f'<a name="{ANCHOR_PREFIX}{seg.id}"></a>')
    out.append(f'<div class="stream">{esc(seg.stream)}</div>')

    if seg.alt_model:
        out.append(_ab_columns(seg, cfg))
    else:
        out.append(_tiers(seg.original, seg.romaji, seg.english, cfg))

    if cfg.show_breakdown and seg.tokens:
        # QTextBrowser has no JavaScript and ignores <details>, so the toggle is
        # an anchor the window intercepts via anchorClicked (setOpenLinks(False))
        # and turns into a re-render with this id flipped in `expanded`.
        n = len(seg.tokens) + (seg.truncated or 0)
        caret = "▾" if expanded else "▸"
        out.append(
            f'<div class="tokhdr"><a class="toktoggle" href="{TOK_SCHEME}{seg.id}">'
            f'{caret} {n} word{"" if n == 1 else "s"}</a></div>'
        )
        if not expanded:
            out.append("</div>")
            return "".join(out)

        rows = []
        for t in seg.tokens:
            if t.gloss is None:
                gloss = '<span class="waiting">…</span>'
            elif t.gloss == "":
                gloss = '<span class="waiting">?</span>'
            else:
                gloss = f'<span class="tgloss">{esc(t.gloss)}</span>'
            rom = f'<span class="trom">({esc(t.romaji)})</span>' if t.romaji else ""
            # The gloss is the dictionary form's meaning; without this the
            # breakdown reads "i can go" for 行けなかった ("couldn't go").
            infl = (
                f' <span class="tinfl">[{esc(", ".join(t.infl))}]</span>'
                if t.infl else ""
            )
            rows.append(
                f'<span class="tsurf">{esc(t.surface)}</span> {rom} — {gloss}{infl}'
            )
        if seg.truncated:
            rows.append(f'<span class="waiting">… (+{seg.truncated} more)</span>')
        out.append('<div class="tok">' + "<br>".join(rows) + "</div>")

    out.append("</div>")
    return "".join(out)


def _set_macos_app_name(name: str) -> None:
    """Name the macOS application menu.

    An unbundled Python process reports CFBundleName "Python", and that — not
    QApplication.setApplicationName(), and not argv[0] — is what macOS puts in
    the menu bar. Overriding the main bundle's info dictionary is the only fix
    that works without shipping a real .app bundle. Must run before
    QApplication is constructed.

    Silently does nothing off macOS or without pyobjc; a cosmetic menu label is
    never worth failing a launch over.
    """
    if sys_platform() != "darwin":
        return
    try:
        from Foundation import NSBundle
    except ImportError:
        log.debug("pyobjc not installed — app menu will read 'Python'")
        return
    try:
        bundle = NSBundle.mainBundle()
        info = bundle.localizedInfoDictionary() or bundle.infoDictionary()
        info["CFBundleName"] = name
    except Exception:  # pragma: no cover - defensive, cosmetic only
        log.debug("could not set macOS app name", exc_info=True)


def sys_platform() -> str:
    import sys

    return sys.platform


try:
    from PyQt6 import QtCore, QtGui, QtWidgets

    class TranscriptWindow(QtWidgets.QMainWindow):
        def __init__(self, cfg: UiCfg, on_close=None, sources=None, on_source=None,
                     current_source: str = "", on_toggle=None, on_refresh=None):
            super().__init__()
            self.cfg = cfg
            self._on_close = on_close
            self._on_toggle = on_toggle
            # Marks the transcript for rebuild on the next tick. A tier toggle
            # changes what every stored line renders as, and without this the
            # change would not show until the next line arrived — which during a
            # quiet moment reads as the button not working.
            self._on_refresh = on_refresh
            self._sources = list(sources or [])
            self._on_source = on_source
            self._src_buttons: dict[str, "QtWidgets.QPushButton"] = {}
            self.setWindowTitle(WINDOW_TITLE)
            self.resize(860, 620)
            self.setMinimumSize(420, 280)

            self.view = QtWidgets.QTextBrowser()
            self.view.setOpenExternalLinks(False)
            # Breakdown toggles are anchors; without this QTextBrowser tries to
            # navigate to "tok:12" and blanks the transcript.
            self.view.setOpenLinks(False)
            self.view.anchorClicked.connect(self._on_anchor)
            self.view.document().setDefaultStyleSheet(_CSS)
            # Rich-text layout is incremental: after setHtml() the document keeps
            # growing across several event-loop turns, so scrollBar.maximum() is
            # wrong not just immediately but for a while afterwards. Re-pinning on
            # every layout change is the only thing that actually holds the bottom
            # — a one-shot deferred setValue still lands short on a long document.
            self._pin_bottom = True
            self._programmatic_scroll = False
            # (anchor name, pixels the view sits below it). The reader's place
            # when they are *not* following the tail — see set_html().
            self._anchor_hold: tuple[str, int] = ("", 0)
            self._holding = False
            self.view.document().documentLayout().documentSizeChanged.connect(
                self._on_doc_resized
            )
            # Pinning has to answer to the reader, not only to the update loop.
            # Without this the mode only changed when a new segment arrived, so
            # scrolling up during a quiet moment was undone by the next layout
            # pass, and scrolling back down did not re-arm until something new
            # was said. Every hand-driven scroll re-decides it immediately.
            self.view.verticalScrollBar().valueChanged.connect(self._on_scrolled)
            self.view.setStyleSheet(
                "QTextBrowser{background:#0f1115;border:none;padding:12px;}"
            )

            # Transcript fills the window; the control bar sits under it. The
            # bar is always built now that it carries the breakdown toggle — the
            # source half of it is what stays conditional, since on the device
            # backend there is nothing to switch between.
            central = QtWidgets.QWidget()
            col = QtWidgets.QVBoxLayout(central)
            col.setContentsMargins(0, 0, 0, 0)
            col.setSpacing(0)
            col.addWidget(self.view, 1)
            col.addWidget(self._mk_source_bar(current_source), 0)
            self.setCentralWidget(central)

            self.status = self.statusBar()
            self.status.setStyleSheet("color:#7d8595;background:#0f1115;")
            self.status.showMessage("starting…")

            pal = self.palette()
            pal.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor("#0f1115"))
            self.setPalette(pal)

            self._mk_actions()

        def _mk_source_bar(self, current: str) -> "QtWidgets.QWidget":
            bar = QtWidgets.QWidget()
            bar.setObjectName("srcbar")
            bar.setStyleSheet(_BAR_QSS)
            row = QtWidgets.QHBoxLayout(bar)
            row.setContentsMargins(12, 8, 12, 8)
            row.setSpacing(8)

            if self._sources and self._on_source is not None:
                cap = QtWidgets.QLabel("Source")
                cap.setObjectName("srclabel")
                row.addWidget(cap)

                group = QtWidgets.QButtonGroup(self)
                group.setExclusive(True)  # mutually exclusive: one tap at a time
                for i, (key, label) in enumerate(self._sources, start=1):
                    b = QtWidgets.QPushButton(label)
                    b.setObjectName("src")
                    b.setCheckable(True)
                    b.setChecked(key == current)
                    b.setToolTip(f"Capture audio from {label}  (⌘{i})")
                    b.clicked.connect(lambda _checked, k=key: self._pick_source(k))
                    group.addButton(b)
                    row.addWidget(b)
                    self._src_buttons[key] = b

            row.addStretch(1)
            self.src_note = QtWidgets.QLabel("")
            self.src_note.setObjectName("srclabel")
            row.addWidget(self.src_note)

            # Off does not merely hide the tier: it stops the per-word lane being
            # submitted at all, which is the point of the switch. Lines captured
            # while it is off keep their word count and fill their meanings in
            # when opened.
            self.bd_button = QtWidgets.QPushButton("Breakdown")
            self.bd_button.setObjectName("src")
            self.bd_button.setCheckable(True)
            self.bd_button.setChecked(self.cfg.show_breakdown)
            self.bd_button.setToolTip(
                "Per-word breakdown. Off skips the per-word lookups entirely; "
                "open a line later to fill them in."
            )
            self.bd_button.clicked.connect(self._set_breakdown)
            row.addWidget(self.bd_button)
            return bar

        def _set_breakdown(self, on: bool) -> None:
            """One flag, two controls — the bar button and the View menu item."""
            self.cfg.show_breakdown = on
            self.bd_button.setChecked(on)
            self.act_breakdown.setChecked(on)
            if self._on_refresh is not None:
                self._on_refresh()

        def _pick_source(self, key: str) -> None:
            """Hand the switch to a worker thread and return immediately.

            Switching closes a Core Audio tap, walks the process table and launches
            ProcTap's helper app — hundreds of milliseconds. Doing that inline would
            freeze the transcript mid-conversation, and Qt would repaint the button
            as pressed only after the work finished, making the UI feel broken
            precisely when the user is watching it.
            """
            if self._on_source is None:
                return
            for k, b in self._src_buttons.items():
                b.setChecked(k == key)
                b.setEnabled(False)
            self.src_note.setText("switching…")

            def _work():
                try:
                    msg = self._on_source(key) or ""
                except Exception as e:  # never let a switch kill the UI thread
                    log.warning("source switch failed: %s", e)
                    msg = f"switch failed: {e}"
                # Back to the Qt thread before touching a widget.
                QtCore.QMetaObject.invokeMethod(
                    self, "_switch_done", QtCore.Qt.ConnectionType.QueuedConnection,
                    QtCore.Q_ARG(str, msg),
                )

            threading.Thread(target=_work, name="ui-switch", daemon=True).start()

        @QtCore.pyqtSlot(str)
        def _switch_done(self, msg: str) -> None:
            for b in self._src_buttons.values():
                b.setEnabled(True)
            self.src_note.setText(msg)

        def _mk_actions(self) -> None:
            m = self.menuBar().addMenu("View")

            self.act_top = QtGui.QAction("Always on top", self, checkable=True)
            self.act_top.triggered.connect(self._toggle_on_top)
            m.addAction(self.act_top)

            self.act_romaji = QtGui.QAction("Show romaji", self, checkable=True)
            self.act_romaji.setChecked(self.cfg.show_romaji)
            self.act_romaji.triggered.connect(self._set_romaji)
            m.addAction(self.act_romaji)

            # Shares `_set_breakdown` with the bar button so the two can never
            # disagree about a flag they both own.
            self.act_breakdown = QtGui.QAction("Show breakdown", self, checkable=True)
            self.act_breakdown.setChecked(self.cfg.show_breakdown)
            self.act_breakdown.triggered.connect(self._set_breakdown)
            m.addAction(self.act_breakdown)

            m.addSeparator()
            clear = QtGui.QAction("Clear", self, shortcut="Ctrl+K")
            clear.triggered.connect(self.view.clear)
            m.addAction(clear)

            if not (self._sources and self._on_source is not None):
                return
            # Keyboard access to the same switch. Qt maps Ctrl to Command on
            # macOS, so these register as ⌘1/⌘2 — which is what the tooltips say.
            sm = self.menuBar().addMenu("Source")
            for i, (key, label) in enumerate(self._sources, start=1):
                a = QtGui.QAction(label, self, shortcut=f"Ctrl+{i}")
                a.triggered.connect(lambda _c=False, k=key: self._pick_source(k))
                sm.addAction(a)

        def _set_romaji(self, on: bool) -> None:
            self.cfg.show_romaji = on
            if self._on_refresh is not None:
                self._on_refresh()

        def _toggle_on_top(self, checked: bool) -> None:
            flags = self.windowFlags()
            flag = QtCore.Qt.WindowType.WindowStaysOnTopHint
            self.setWindowFlags(flags | flag if checked else flags & ~flag)
            self.show()  # re-show: changing flags hides the window on macOS

        def _on_anchor(self, url) -> None:
            href = url.toString()
            if href.startswith(TOK_SCHEME) and self._on_toggle is not None:
                try:
                    self._on_toggle(int(href[len(TOK_SCHEME):]))
                except ValueError:  # pragma: no cover - defensive
                    pass

        def at_bottom(self) -> bool:
            sb = self.view.verticalScrollBar()
            return sb.value() >= sb.maximum() - STICKY_BOTTOM_PX

        def _set_scroll(self, value: int) -> None:
            """Move the bar without it counting as the reader having scrolled.

            Saves and restores the flag rather than clearing it, because this is
            re-entrant: setHtml() runs under the same guard and its layout pass
            calls _on_doc_resized, which lands back here. Clearing unconditionally
            would drop the outer guard halfway through the rebuild and let the
            teardown transients through again.
            """
            was_programmatic = self._programmatic_scroll
            self._programmatic_scroll = True
            try:
                self.view.verticalScrollBar().setValue(value)
            finally:
                self._programmatic_scroll = was_programmatic

        def _on_scrolled(self, _value: int) -> None:
            if not self._programmatic_scroll:
                self._pin_bottom = self.at_bottom()

        @staticmethod
        def _block_anchors(block) -> list[str]:
            """Anchor names carried by a block.

            A name-only <a> produces no text of its own, so Qt hangs it on the
            first character of whatever follows — which lands in the block we
            want. Hence fragments rather than the block's own char format.
            """
            names: list[str] = []
            it = block.begin()
            while not it.atEnd():
                frag = it.fragment()
                if frag.isValid():
                    names.extend(frag.charFormat().anchorNames())
                it += 1
            return names

        def _anchor_at_top(self) -> tuple[str, int]:
            """Which segment the reader is looking at, and by how much it is
            scrolled past the viewport's top edge.

            Returns ("", 0) when nothing is anchored, which puts set_html back on
            the pixel path.
            """
            view_y = self.view.verticalScrollBar().value()
            lay = self.view.document().documentLayout()
            block = self.view.cursorForPosition(QtCore.QPoint(0, 0)).block()
            while block.isValid():
                for name in self._block_anchors(block):
                    if name.startswith(ANCHOR_PREFIX):
                        return name, round(view_y - lay.blockBoundingRect(block).top())
                block = block.previous()
            return "", 0

        def _anchor_offset(self, name: str) -> Optional[int]:
            """Where that anchor sits now, or None if its segment is gone."""
            lay = self.view.document().documentLayout()
            block = self.view.document().begin()
            while block.isValid():
                if name in self._block_anchors(block):
                    return round(lay.blockBoundingRect(block).top())
                block = block.next()
            return None

        def _hold_anchor(self) -> bool:
            """Put the held segment back where the reader had it.

            False when the anchor is not in the new document — it aged out of the
            buffer, or this is a different transcript entirely. The caller falls
            back to the pixel offset, which is wrong only by however much the
            document shifted, rather than dumping the reader at the top.
            """
            name, delta = self._anchor_hold
            top = self._anchor_offset(name)
            if top is None:
                return False
            self._set_scroll(top + delta)
            return True

        def _on_doc_resized(self, _size=None) -> None:
            """Layout grew or shrank. Re-assert whatever the reader was holding.

            Both branches have to survive incremental layout, not just fire once:
            the document keeps changing height for several event-loop turns after
            setHtml() returns, and a single correction lands short.
            """
            if self._pin_bottom:
                self._set_scroll(self.view.verticalScrollBar().maximum())
            elif (
                self._anchor_hold[0]
                # Mid-rebuild the document is briefly empty, so the anchor is not
                # findable and every lookup would miss. set_html re-asserts the
                # hold itself once setHtml() has returned; this branch is only for
                # the incremental layout that keeps arriving afterwards.
                and not self._programmatic_scroll
                and not self._holding
            ):
                self._holding = True
                try:
                    self._hold_anchor()
                finally:
                    self._holding = False

        def set_html(self, body: str, keep_bottom: bool) -> None:
            """Replace the transcript, holding the reader's place.

            The original code read verticalScrollBar().maximum() on the line after
            setHtml() and scrolled there. That value is the *previous* document's
            height, because rich-text layout has not run yet — so the view landed
            above the true bottom and the error compounded on every update
            (measured: 13px, 15, 26, 47, 81 over five consecutive lines). Once the
            drift passed STICKY_BOTTOM_PX the sink stopped considering the view
            pinned at all, and it never recovered without a manual scroll.

            Deferring the setValue by one event-loop turn was still not enough:
            layout is incremental and keeps extending the document afterwards.
            So `_pin_bottom` is a mode rather than a one-off action — while it is
            set, every documentSizeChanged drags the bar back to the bottom, for
            as long as the layout keeps moving.

            setHtml() itself runs under the programmatic guard. It tears the old
            document down before building the new one, and the bar emits
            valueChanged throughout — measured on one update: value 0 against
            maximum 0, then 4039, then 4, then the real position. Ungated, each of
            those reached _on_scrolled, which read `at_bottom()` off a document
            mid-rebuild and rewrote `_pin_bottom` from it. value 0 / maximum 0
            says "at the bottom" and armed the pin under a reader who had
            deliberately scrolled up; value 4 against maximum 4039 says the
            opposite and disarmed it under one who was following the tail. Which
            transient landed last was a race with layout, which is why the
            symptom was intermittent and why it never reproduced offscreen, where
            layout is effectively synchronous.

            A reader who is *not* following the tail is held by segment anchor
            rather than by pixel offset, because past `max_lines` the two stop
            meaning the same thing. At the cap every update drops the oldest
            block and appends a new one, so the document height does not change
            and the old `min(prev, maximum())` restore looked perfect — the
            scrollbar never moved. The text moved instead, one line per
            utterance, sliding out from under whoever was reading it. Confirmed
            live at ~220 lines. Holding the anchor makes the restore mean "the
            line they were on" instead of "that many pixels down".
            """
            sb = self.view.verticalScrollBar()
            prev = sb.value()
            self._anchor_hold = ("", 0) if keep_bottom else self._anchor_at_top()
            self._pin_bottom = keep_bottom
            was_programmatic = self._programmatic_scroll
            self._programmatic_scroll = True
            try:
                self.view.setHtml(f"<body>{body}</body>")
            finally:
                self._programmatic_scroll = was_programmatic
            if keep_bottom:
                # _on_doc_resized corrects any shortfall as layout continues.
                self._set_scroll(sb.maximum())
            elif not (self._anchor_hold[0] and self._hold_anchor()):
                self._set_scroll(min(prev, sb.maximum()))

        def closeEvent(self, event):
            if self._on_close is not None:
                self._on_close()
            super().closeEvent(event)

except ImportError:  # pragma: no cover - PyQt6 optional
    TranscriptWindow = None  # type: ignore[misc, assignment]


class WindowSink:
    """UI sink that owns the Qt event loop. Must run on the main thread."""

    def __init__(self, cfg: UiCfg, ws_sink: Optional[object] = None,
                 expect_translation: bool = True, device_name: str = "",
                 sources=None, on_source=None, current_source: str = "",
                 on_glosses=None):
        if TranscriptWindow is None:
            raise ImportError("PyQt6 is not installed (pip install PyQt6)")
        self.cfg = cfg
        self.ws_sink = ws_sink
        self.expect_translation = expect_translation
        self.device_name = device_name
        self.sources = sources
        self.on_source = on_source
        self.current_source = current_source
        self.on_glosses = on_glosses
        self.segments: dict[int, Segment] = {}
        self.order: list[int] = []
        # Segment ids whose per-word breakdown the reader has opened. Collapsed
        # is the default, so this starts empty and only grows on a click.
        self._expanded: set[int] = set()
        # Segment ids already sent for a late gloss fetch. Without this, opening
        # and closing a line while its lookup is still in flight would queue the
        # same round-trip again.
        self._backfilled: set[int] = set()
        self._dirty = False
        self._seen_total = 0
        # --log-file support. A segment is written once, when every lane that
        # will report has reported — same completion rule as the console sink,
        # because the gloss lane normally finishes before the sentence lane.
        self._expect = {"breakdown"} | ({"english"} if expect_translation else set())
        self._seen: dict[int, set[str]] = {}
        self._logged: set[int] = set()

    def run(self, stop_evt: threading.Event) -> None:
        import sys

        from PyQt6 import QtCore, QtWidgets

        _set_macos_app_name(WINDOW_TITLE)
        sys.argv[0] = WINDOW_TITLE

        app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
        app.setApplicationName(WINDOW_TITLE)
        app.setApplicationDisplayName(WINDOW_TITLE)

        def _toggle(seg_id: int) -> None:
            self._expanded.symmetric_difference_update({seg_id})
            if seg_id in self._expanded:
                self._backfill(seg_id)
            self._dirty = True  # picked up by the next tick

        def _refresh() -> None:
            self._dirty = True

        win = TranscriptWindow(
            self.cfg, on_close=stop_evt.set, sources=self.sources,
            on_source=self.on_source, current_source=self.current_source,
            on_toggle=_toggle, on_refresh=_refresh,
        )
        win.set_html(_empty_state(self.device_name, self.current_source), True)
        win.show()
        win.raise_()
        win.activateWindow()

        def tick() -> None:
            if stop_evt.is_set():
                app.quit()
                return
            self._drain()
            if self._dirty:
                self._dirty = False
                keep = win.at_bottom()
                body = "".join(
                    _fmt_block(self.segments[i], self.cfg, i in self._expanded)
                    for i in self.order[-self.cfg.max_lines:]
                )
                win.set_html(body, keep)
            win.status.showMessage(self._status())

        timer = QtCore.QTimer()
        timer.timeout.connect(tick)
        timer.start(POLL_MS)

        # Let SIGINT (Ctrl-C in the launching terminal) reach the Qt loop.
        sigtimer = QtCore.QTimer()
        sigtimer.timeout.connect(lambda: None)
        sigtimer.start(200)

        app.exec()
        stop_evt.set()

    def _drain(self) -> None:
        """Pull every pending event. Batching keeps a burst to one repaint."""
        while True:
            try:
                kind, seg = ui_q.get_nowait()
            except queue.Empty:
                return

            if self.ws_sink is not None:
                try:
                    self.ws_sink.push(seg)
                except Exception:
                    pass

            if seg.id not in self.segments:
                self.order.append(seg.id)
                self._seen_total += 1
                while len(self.order) > self.cfg.max_lines:
                    dropped = self.order.pop(0)
                    self.segments.pop(dropped, None)
                    self._expanded.discard(dropped)
                    self._backfilled.discard(dropped)
            self.segments[seg.id] = seg
            self._dirty = True

            if self.cfg.log_file and seg.id not in self._logged:
                seen = self._seen.setdefault(seg.id, set())
                seen.add(kind)
                if self._expect.issubset(seen):
                    self._seen.pop(seg.id, None)
                    self._logged.add(seg.id)
                    self._append_log(seg)

    def _backfill(self, seg_id: int) -> None:
        """Fetch the meanings for a line that came through with the tier off.

        Only the gloss is deferred: the tokenizer runs regardless, so the line
        already has its surfaces, romaji and inflections and only the English
        column is blank. `gloss is None` is the marker for "never looked up" —
        an empty string means the lookup ran and found nothing, which no amount
        of retrying will change.
        """
        if self.on_glosses is None or seg_id in self._backfilled:
            return
        seg = self.segments.get(seg_id)
        if seg is None or not any(t.gloss is None for t in seg.tokens):
            return
        self._backfilled.add(seg_id)
        try:
            self.on_glosses(seg)
        except Exception:
            log.exception("gloss backfill failed")

    def _append_log(self, seg: Segment) -> None:
        from tsutawaru.ui.formatter import render_block

        try:
            with open(self.cfg.log_file, "a", encoding="utf-8") as fh:
                fh.write(render_block(seg, self.cfg) + "\n")
        except OSError as e:
            log.warning("could not write log file %s: %s", self.cfg.log_file, e)

    def _status(self) -> str:
        snap = metrics.snapshot()
        if self._seen_total == 0:
            dev = self.device_name or "audio device"
            return f"listening on {dev} — no speech yet"
        bits = [f"{self._seen_total} lines"]
        if "stt" in snap:
            bits.append(f"stt {snap['stt']['p50']:.0f}ms")
        if "xlate_sentence" in snap:
            bits.append(f"translate {snap['xlate_sentence']['p50']:.0f}ms")
        from tsutawaru.pipeline import queues

        dropped = queues.total_dropped()
        if dropped:
            bits.append(f"dropped {dropped}")
        return "   ".join(bits)
