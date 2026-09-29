"""Escalation must not abandon a process that has already produced audio.

Regression test for the dropout measured in the 2026-08-22 live session: four
ordinary conversational pauses each tripped the 12 s escalation timer, walked
the tap off Discord's correct renderer, and cost 108-119 s of audio while it
cycled every other Discord process. ~16% of a 50-minute call was lost this way.

The fix is `_proc_confirmed`: once a candidate delivers a nonzero frame it is
known-good, and silence after that is just silence.
"""
from __future__ import annotations

import types

from tsutawaru.pipeline.orchestrator import Orchestrator, PROC_ESCALATE_S


def _fake_orch(confirmed: bool):
    """Minimal stand-in exposing only what _check_silence_process touches."""
    calls = {"advance": 0, "retap": 0, "forced": 0}

    def advance_candidate():
        calls["advance"] += 1
        return "renderer (pid 999)"

    cap = types.SimpleNamespace(
        device="Discord (pid 123)",
        source=None,
        advance_candidate=advance_candidate,
    )

    def _retap(force=False):
        calls["retap"] += 1
        if force:
            calls["forced"] += 1

    orch = types.SimpleNamespace(
        _last_retap=0.0,
        _proc_confirmed=confirmed,
        stages=types.SimpleNamespace(capture=cap),
        _retap=_retap,
        _silent_warner=types.SimpleNamespace(arm_once=lambda: False),
    )
    return orch, calls


def test_confirmed_tap_never_escalates_through_a_long_pause():
    """A six-person call goes quiet for two minutes. The tap must stay put."""
    orch, calls = _fake_orch(confirmed=True)
    Orchestrator._check_silence_process(orch, silent_for=120.0)

    assert calls["advance"] == 0, "walked away from a known-good process"
    assert calls["forced"] == 0
    assert calls["retap"] == 1, "should still re-resolve, just not escalate"


def test_unconfirmed_tap_still_escalates():
    """Startup search is untouched: a tap that never produced audio moves on."""
    orch, calls = _fake_orch(confirmed=False)
    Orchestrator._check_silence_process(orch, silent_for=PROC_ESCALATE_S + 1.0)

    assert calls["advance"] == 1, "wrong-process recovery must still work"
    assert calls["forced"] == 1


def test_unconfirmed_tap_waits_out_the_grace_period():
    """Below the threshold it re-resolves without giving up on the candidate."""
    orch, calls = _fake_orch(confirmed=False)
    Orchestrator._check_silence_process(orch, silent_for=PROC_ESCALATE_S - 1.0)

    assert calls["advance"] == 0
    assert calls["retap"] == 1


def test_escalation_threshold_survives_a_conversational_pause():
    """12 s was inside the range of a normal group-call pause. 45 s is not."""
    assert PROC_ESCALATE_S >= 30.0


# --- the startup search itself -------------------------------------------------
#
# The complaint these cover: "tsutawaru won't read anything if I run it before
# YouTube." Nothing was wrong with the tap — the blind scan had walked off the
# audio-service while Chrome sat idle, and with 25 tappable Chrome processes it
# could not get back for 18.75 minutes.

from tsutawaru.audio.capture_proc import ProcessCapture
from tsutawaru.audio.sources import SOURCES, Resolved, SourceUnavailable


def _capture(monkeypatch, n_candidates: int) -> ProcessCapture:
    """A capture whose source offers `n_candidates` pids. 0 = app not running."""
    src = SOURCES["youtube"]

    def fake_resolve(source, index=0):
        if not n_candidates:
            raise SourceUnavailable(source, "the application is not running")
        i = index % n_candidates
        return Resolved(1000 + i, "audio-service" if i == 0 else "renderer", source)

    monkeypatch.setattr("tsutawaru.audio.capture_proc.resolve", fake_resolve)
    return ProcessCapture(src)


def test_scan_returns_to_the_best_guess_every_other_step(monkeypatch):
    """Chrome's audio-service is candidate 0 and the only pid that ever plays."""
    cap = _capture(monkeypatch, 25)
    walked = [cap.candidate]
    for _ in range(8):
        cap.advance_candidate()
        walked.append(cap.candidate)

    assert walked == [0, 1, 0, 2, 0, 3, 0, 4, 0], walked


def test_scan_still_reaches_every_candidate(monkeypatch):
    """Discord's audio is on a renderer, so the far end of the list is load-bearing."""
    # % 4 because `resolve` is what maps the counter onto the list, and it wraps;
    # `start()` hands it the raw value the same way.
    cap = _capture(monkeypatch, 4)
    seen = {cap.candidate}
    for _ in range(12):
        cap.advance_candidate()
        seen.add(cap.candidate % 4)

    assert seen == {0, 1, 2, 3}, seen


def test_counter_does_not_run_away_while_the_app_is_closed(monkeypatch):
    """Absent app -> escalation fires every 3 s, not every 45. It must be inert.

    The bumped-then-looked-up version left the index at ~200 after ten minutes,
    so opening Chrome afterwards tapped 200 % 25 — an arbitrary renderer.
    """
    cap = _capture(monkeypatch, 0)
    for _ in range(200):
        assert cap.advance_candidate() == "no other candidate"

    assert cap.candidate == 0, "must still open on the best guess"
    assert cap._probe == 0
