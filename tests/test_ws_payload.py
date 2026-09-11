"""The WebSocket sink must project Segment, not re-describe it."""
from __future__ import annotations

from tsutawaru.models import Segment


def seg(**kw) -> Segment:
    kw.setdefault("stream", "youtube")
    kw.setdefault("original", "テスト")
    return Segment.new(**kw)


def test_ws_payload_is_the_segment_projection():
    """Regression: the sink used to hand-build a second, drifting copy.

    `push()` maintained its own key list independent of `to_dict()`, and nothing
    forced the two to agree. They drifted — a field added to `to_dict()` reached
    the window and `--record` but silently never reached a single WebSocket
    client, with no error at any layer. Serialising `to_dict()` itself is the
    fix; a hand-built dict re-arms the same trap for the next field added.
    """
    import inspect

    from tsutawaru.ui import ws_server

    src = inspect.getsource(ws_server.WSSink.push)
    assert "to_dict()" in src
    assert '"jp": seg.original' not in src


def test_payload_keeps_the_keys_the_dock_reads():
    d = seg(romaji="tesuto", english="test").to_dict()
    for k in ("id", "stream", "jp", "romaji", "en", "tokens"):
        assert k in d


def test_stt_scores_reach_the_payload():
    """Kept for offline analysis of --record sessions, not for any live display."""
    d = seg(confidence=-0.77, no_speech=0.12).to_dict()
    assert d["confidence"] == -0.77 and d["no_speech"] == 0.12
