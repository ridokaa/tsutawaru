"""WebSocket broadcast server for OBS Browser Dock and browser overlay clients."""
from __future__ import annotations

import asyncio
import json
import threading
from typing import Set

import websockets

from tsutawaru.logbus import get_logger
from tsutawaru.models import Segment

log = get_logger(__name__)


class WSSink:
    def __init__(self, port: int = 8765):
        self.port = port
        self.clients: Set[websockets.WebSocketServerProtocol] = set()
        self.loop: asyncio.AbstractEventLoop | None = None
        self.error: Exception | None = None

    async def _handler(self, ws):
        self.clients.add(ws)
        try:
            await ws.wait_closed()
        finally:
            self.clients.discard(ws)

    async def _serve(self, ready: threading.Event | None = None):
        async with websockets.serve(self._handler, "127.0.0.1", self.port):
            log.info("websocket server listening on 127.0.0.1:%d", self.port)
            if ready is not None:
                ready.set()   # bind succeeded
            await asyncio.Future()

    def start(self) -> None:
        """Start the broadcast server. Never raises — the transcript matters more.

        The bind happens on a worker thread, so a failure there escapes any
        try/except around start() and dumps a traceback. The common case is
        entirely benign: a second instance finds the port held by the first.
        Report it in one line and carry on with the server disabled.
        """
        ready = threading.Event()

        def run():
            try:
                self.loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self.loop)
                self.loop.run_until_complete(self._serve(ready))
            except OSError as e:
                self.loop = None
                self.error = e
                if e.errno in (48, 98):  # EADDRINUSE (BSD, Linux)
                    log.warning(
                        "port %d is already in use — WebSocket/OBS output disabled "
                        "for this instance (another tsutawaru is probably running). "
                        "Use --ws-port to pick a different port.",
                        self.port,
                    )
                else:
                    log.warning("WebSocket server could not start: %s", e)
            except Exception as e:  # pragma: no cover - defensive
                self.loop = None
                self.error = e
                log.warning("WebSocket server stopped: %s", e)
            finally:
                ready.set()

        threading.Thread(target=run, daemon=True, name="ws-server").start()
        ready.wait(timeout=3.0)  # surface bind errors before the UI starts
        return self.error is None

    def push(self, seg: Segment) -> None:
        """Push a segment update to all connected WebSocket clients."""
        if not self.loop:
            return
        # Segment.to_dict() rather than a hand-built dict: this was a second,
        # independently maintained projection of the same object, and it drifted
        # — `confidence` was added to to_dict() and silently never reached any
        # WebSocket client, because nothing forces the two lists to agree. The
        # dock reads id/stream/jp/romaji/en and t.s/t.r/t.g, all of which
        # to_dict() emits, so this is a strict superset of the old payload.
        payload = json.dumps(seg.to_dict(), ensure_ascii=False)
        asyncio.run_coroutine_threadsafe(self._broadcast(payload), self.loop)

    async def _broadcast(self, payload: str) -> None:
        if not self.clients:
            return
        results = await asyncio.gather(
            *(c.send(payload) for c in list(self.clients)), return_exceptions=True
        )
        for c, r in zip(list(self.clients), results):
            if isinstance(r, Exception):
                self.clients.discard(c)
