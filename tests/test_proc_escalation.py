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
