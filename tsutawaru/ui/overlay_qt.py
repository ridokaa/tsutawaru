"""PyQt6 always-on-top transparent floating overlay window."""
from __future__ import annotations

from tsutawaru.config import UiCfg

try:
    from PyQt6 import QtCore, QtWidgets

    class Overlay(QtWidgets.QWidget):
        new_block = QtCore.pyqtSignal(str)  # thread-safe bridge from worker threads

        def __init__(self, cfg: UiCfg):
            super().__init__()
            self.cfg = cfg
            self.setWindowFlags(
                QtCore.Qt.WindowType.FramelessWindowHint
                | QtCore.Qt.WindowType.WindowStaysOnTopHint
                | QtCore.Qt.WindowType.Tool
            )
            self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)
            self.view = QtWidgets.QTextBrowser(self)
            self.view.setStyleSheet(
                "background: rgba(12,12,16,190); color:#EDEDED;"
                "font-family:'Hiragino Sans','Yu Gothic UI','Noto Sans JP',monospace;"
                "font-size:15px; border-radius:10px; padding:10px;"
            )
            lay = QtWidgets.QVBoxLayout(self)
            lay.setContentsMargins(0, 0, 0, 0)
            lay.addWidget(self.view)
            self.resize(720, 320)
            self.new_block.connect(self._append)
            self._drag = None

        @QtCore.pyqtSlot(str)
        def _append(self, html: str) -> None:
            self.view.append(html)
            sb = self.view.verticalScrollBar()
            sb.setValue(sb.maximum())

        def mousePressEvent(self, e):
            if e.button() == QtCore.Qt.MouseButton.LeftButton:
                self._drag = e.globalPosition().toPoint() - self.pos()

        def mouseMoveEvent(self, e):
            if self._drag is not None:
                self.move(e.globalPosition().toPoint() - self._drag)

        def mouseReleaseEvent(self, e):
            self._drag = None

except ImportError:  # pragma: no cover - PyQt6 optional
    Overlay = None  # type: ignore[misc, assignment]
