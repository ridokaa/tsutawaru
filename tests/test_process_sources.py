"""Per-process capture: backend resolution, source resolution, format handling.

The invariant this file exists to protect is that `"auto"` never reaches a branch
that only understands `"device"` and `"process"`, and that an *explicit* request
for a tap is never silently downgraded. Everything else here guards a specific
defect that was either found while building it or is invisible until a user hits
it live.
"""
from __future__ import annotations

import numpy as np
import pytest

from tsutawaru.audio import sources
from tsutawaru.config import AudioCfg, Config, load, resolve_audio_backend


# ------------------------------------------------------------ backend resolution

def test_auto_is_the_default():
    assert AudioCfg().backend == "auto"
    assert load().audio.backend == "auto"


def test_auto_never_leaks_downstream():
    """Every consumer branches on == "process"; "auto" matches neither arm.

    An unresolved value does not fail loudly, it silently behaves as "device" at
    every branch point — which is why resolution is a property rather than a
    convention that callers are expected to remember.
    """
    assert AudioCfg().effective_backend in {"device", "process"}


def test_auto_prefers_a_tap_when_the_platform_supports_one(monkeypatch):
    import tsutawaru.config as C

    monkeypatch.setattr(C, "proctap_unavailable", lambda: "")
    assert resolve_audio_backend("auto") == "process"


def test_auto_degrades_to_device_when_it_cannot_tap(monkeypatch):
    import tsutawaru.config as C

    monkeypatch.setattr(C, "proctap_unavailable", lambda: "no proc-tap")
    assert resolve_audio_backend("auto") == "device"
    # An explicit choice is passed through untouched either way — degrading it
    # would mean the user asked for one backend and quietly ran another.
    assert resolve_audio_backend("process") == "process"
    assert resolve_audio_backend("device") == "device"


def test_explicit_process_fails_loudly_rather_than_degrading(monkeypatch):
    """The whole point of spelling it out is that it is not a preference."""
    import tsutawaru.config as C

    monkeypatch.setattr(C, "proctap_unavailable", lambda: "missing 'proc-tap'")
    cfg = Config()
    cfg.audio.backend = "process"
    problems = cfg.validate()
    assert any("proc-tap" in p for p in problems)
    # …while "auto" treats the same condition as a normal outcome, not an error.
    cfg.audio.backend = "auto"
    assert not [p for p in cfg.validate() if "proc-tap" in p]
    assert cfg.audio.effective_backend == "device"


def test_unavailability_is_reported_with_the_pip_command(monkeypatch):
    """Otherwise this surfaces as 'audio capture is unavailable — cannot start'."""
    import tsutawaru.config as C

    monkeypatch.setattr(C, "sys", type("s", (), {"platform": "darwin"})())
    monkeypatch.setattr(C, "_installed", lambda mod: False)
    assert "pip install" in C.proctap_unavailable()


def test_non_macos_cannot_tap(monkeypatch):
    import tsutawaru.config as C

    monkeypatch.setattr(C, "sys", type("s", (), {"platform": "win32"})())
    assert "macOS" in C.proctap_unavailable()


def test_process_backend_is_rejected_unless_spelled_correctly():
    assert load(overrides={"audio": {"backend": "proctap"}}).audio.backend == "auto"
    assert load(overrides={"audio": {"backend": "process"}}).audio.backend == "process"


def test_unknown_source_falls_back_instead_of_raising():
    """A typo'd source must not take down capture."""
    assert sources.get("nope").key == sources.DEFAULT_SOURCE
    assert sources.get("youtube").key == "youtube"


# ------------------------------------------------------------ pid resolution

def _proc(pid, exe, cmdline):
    return {"pid": pid, "exe": exe, "cmdline": cmdline}


class _FakeProc:
    def __init__(self, d):
        self.info = {"pid": d["pid"], "name": "x", "exe": d["exe"],
                     "cmdline": d["cmdline"]}


def _patch_procs(monkeypatch, procs):
    monkeypatch.setattr(sources, "_iter_procs", lambda: iter([_FakeProc(p) for p in procs]))


CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
CHROME_HELPER = ("/Applications/Google Chrome.app/Contents/Frameworks/"
                 "Google Chrome Helper.app/Contents/MacOS/Google Chrome Helper")


def test_resolves_the_audio_service_not_the_main_process(monkeypatch):
    """The core finding: a Core Audio tap is bound to ONE pid, and for Chromium
    that pid is a utility child. Tapping the main pid captures silence forever."""
    _patch_procs(monkeypatch, [
        _proc(100, CHROME, [CHROME]),
        _proc(101, CHROME_HELPER, [CHROME_HELPER, "--type=renderer"]),
        _proc(102, CHROME_HELPER,
              [CHROME_HELPER, "--type=utility", sources.AUDIO_SERVICE_FLAG]),
    ])
    r = sources.resolve(sources.get("youtube"))
    assert r.pid == 102
    assert r.kind == "audio-service"


def test_discord_prefers_its_renderer_over_its_audio_service(monkeypatch):
    """Measured, not assumed: Discord's audio.mojom.AudioService accepts a tap and
    returns 0 bytes, while its renderer returns real audio (peak 0.385). Being
    Chromium-based predicts nothing here — Chrome is the other way round."""
    d = "/Applications/Discord.app/Contents/MacOS/Discord"
    dh = "/Applications/Discord.app/Contents/Frameworks/Discord Helper.app/x"
    _patch_procs(monkeypatch, [
        _proc(2022, d, [d]),
        _proc(2036, dh, [dh, "--type=renderer"]),
        _proc(2037, dh, [dh, "--type=utility", sources.AUDIO_SERVICE_FLAG]),
    ])
    order = sources.resolve_all(sources.get("discord"))
    assert [(r.pid, r.kind) for r in order][:2] == [
        (2036, "renderer"), (2037, "audio-service"),
    ]


def test_chrome_and_discord_use_opposite_orders(monkeypatch):
    """The whole reason `roles` is per-source rather than one global rule."""
    assert sources.get("youtube").roles[0] == "audio-service"
    assert sources.get("discord").roles[0] == "renderer"


def test_a_renderer_is_a_valid_target_when_no_audio_service_exists(monkeypatch):
    """Earlier this raised 'not playing audio yet'. That policy was wrong: a
    renderer really can be where the audio lives, so refusing it lost Discord."""
    _patch_procs(monkeypatch, [
        _proc(100, CHROME, [CHROME]),
        _proc(101, CHROME_HELPER, [CHROME_HELPER, "--type=renderer"]),
    ])
    r = sources.resolve(sources.get("youtube"))
    assert (r.pid, r.kind) == (101, "renderer")


def test_escalation_walks_the_candidates_and_wraps(monkeypatch):
    """A wrong first guess must be temporary. Wrapping keeps it from dead-ending."""
    d = "/Applications/Discord.app/Contents/MacOS/Discord"
    dh = "/Applications/Discord.app/Contents/Frameworks/Discord Helper.app/x"
    _patch_procs(monkeypatch, [
        _proc(2022, d, [d]),
        _proc(2036, dh, [dh, "--type=renderer"]),
        _proc(2037, dh, [dh, "--type=utility", sources.AUDIO_SERVICE_FLAG]),
    ])
    src = sources.get("discord")
    seen = [sources.resolve(src, i).pid for i in range(4)]
    assert seen[:3] == [2036, 2037, 2022]
    assert seen[3] == seen[0]  # wraps back to the best guess


def test_single_process_app_uses_its_main_pid(monkeypatch):
    """Non-Chromium apps do own their audio, so requiring a helper would break them."""
    exe = "/Applications/Discord.app/Contents/MacOS/Discord"
    _patch_procs(monkeypatch, [_proc(200, exe, [exe])])
    r = sources.resolve(sources.get("discord"))
    assert (r.pid, r.kind) == (200, "main")


def test_app_not_running_is_reported_distinctly(monkeypatch):
    _patch_procs(monkeypatch, [])
    with pytest.raises(sources.SourceUnavailable) as e:
        sources.resolve(sources.get("discord"))
    assert "not running" in e.value.reason


def test_other_apps_are_never_matched(monkeypatch):
    """A source must not capture an unrelated Electron app's audio service."""
    other = "/Applications/Some Other.app/Contents/MacOS/Some Other"
    _patch_procs(monkeypatch, [
        _proc(300, other, [other, "--type=utility", sources.AUDIO_SERVICE_FLAG]),
    ])
    with pytest.raises(sources.SourceUnavailable):
        sources.resolve(sources.get("discord"))


# ------------------------------------------------------------- diagnostics

def test_help_names_permissions_only_when_the_app_is_actually_playing(monkeypatch):
    """Three causes, three different fixes. Listing all of them every time sends
    the user to the wrong one — the defect silent_stream_help exists to avoid."""
    src = sources.get("youtube")

    _patch_procs(monkeypatch, [])
    assert "not running" in sources.source_help(src, 5.0)

    _patch_procs(monkeypatch, [_proc(100, CHROME, [CHROME]),
                               _proc(101, CHROME_HELPER, [CHROME_HELPER, "--type=renderer"])])
    idle = sources.source_help(src, 5.0)
    assert "no audio process yet" in idle
    assert "Screen Recording" not in idle  # would be the wrong fix here

    _patch_procs(monkeypatch, [
        _proc(102, CHROME_HELPER,
              [CHROME_HELPER, "--type=utility", sources.AUDIO_SERVICE_FLAG]),
    ])
    playing = sources.source_help(src, 5.0, got_frames=True)
    assert "Screen Recording" in playing and "ProcTap Helper" in playing


def test_no_packets_and_silent_packets_are_diagnosed_differently(monkeypatch):
    """The discriminator established by probing: an idle app delivers no packets,
    while a missing Screen Recording grant delivers packets of pure zeros. Both
    look like 'silence' and neither errors, so conflating them would send the user
    to System Settings when the real answer is 'press play'."""
    src = sources.get("youtube")
    _patch_procs(monkeypatch, [
        _proc(102, CHROME_HELPER,
              [CHROME_HELPER, "--type=utility", sources.AUDIO_SERVICE_FLAG]),
    ])

    no_packets = sources.source_help(src, 5.0, got_frames=False)
    assert "not be playing anything" in no_packets
    assert no_packets.index("not be playing anything") < no_packets.index("Screen Recording")

    zeros = sources.source_help(src, 5.0, got_frames=True)
    assert "every" in zeros and "zero" in zeros
    assert "not be playing anything" not in zeros  # ruled out by the evidence


def test_no_packets_mentions_the_output_device_mismatch(monkeypatch):
    """An audible app with an empty tap is usually pointed at a non-default device.

    ProcTap's aggregate is built on the system default output, so an app with its
    own output picker (Discord, Zoom, OBS) can be loud in your headphones and
    contribute nothing to the tap. Found the hard way: Chrome worked and Discord
    did not, on the same machine with permissions already granted.
    """
    monkeypatch.setattr(
        "tsutawaru.audio.devices.default_output_name", lambda: "Multi-Output Device"
    )
    _patch_procs(monkeypatch, [
        _proc(102, CHROME_HELPER,
              [CHROME_HELPER, "--type=utility", sources.AUDIO_SERVICE_FLAG]),
    ])
    msg = sources.source_help(sources.get("youtube"), 5.0, got_frames=False)
    assert "output device does not match" in msg
    assert "Multi-Output Device" in msg  # names the target to set the app to
    # Ranked above permissions: it is the likelier cause when audio is audible.
    assert msg.index("output device does not match") < msg.index("Screen Recording")


# --------------------------------------------------------------- float32 path

def test_float32_decoder_downmixes_and_resamples():
    from tsutawaru.audio.capture_proc import Float32FrameDecoder

    d = Float32FrameDecoder(48000, 2)
    # 4800 stereo frames of a constant value -> 1600 mono samples @ 16 kHz
    stereo = np.column_stack([
        np.full(4800, 0.5, dtype=np.float32),
        np.full(4800, 0.1, dtype=np.float32),
    ]).ravel()
    out = d.decode(stereo.tobytes())
    assert out.dtype == np.float32
    assert 1500 < out.size < 1700  # 3:1 decimation, minus filter warm-up
    assert float(np.abs(out).max()) > 0.0


def test_float32_decoder_survives_a_split_frame():
    """ProcTap chunk boundaries need not land on a frame; np.frombuffer would
    raise on a ragged buffer, and one raise per chunk is a dead pipeline."""
    from tsutawaru.audio.capture_proc import Float32FrameDecoder

    d = Float32FrameDecoder(48000, 2)
    raw = np.full(2048, 0.25, dtype=np.float32).tobytes()
    a = d.decode(raw[:1001])   # deliberately ragged
    b = d.decode(raw[1001:])
    assert a.dtype == np.float32 and b.dtype == np.float32
    assert (a.size + b.size) > 0


def test_int16_decoder_would_misread_float32_bytes():
    """Why a separate decoder exists rather than reusing FrameDecoder.

    Each float32 sample reinterpreted as two int16s yields a fixed pair of
    unrelated values (-0.32, +0.47 for an input of 0.02). The level that comes out
    bears no relation to the level that went in, so this cannot be dismissed as a
    small precision loss — and it would pass any 'is there signal?' check.
    """
    from tsutawaru.audio.capture import FrameDecoder
    from tsutawaru.audio.capture_proc import Float32FrameDecoder

    quiet = np.full(4096, 0.02, dtype=np.float32)  # ~ -34 dBFS, clearly quiet
    right = Float32FrameDecoder(48000, 2).decode(quiet.tobytes())
    wrong = FrameDecoder(48000, 2).decode(quiet.tobytes())

    assert abs(float(np.abs(right).max()) - 0.02) < 0.005  # level preserved
    assert float(np.abs(wrong).max()) > 3 * float(np.abs(right).max())


# -------------------------------------------------------------- retap policy

def test_healthy_tap_is_not_reopened_just_because_it_is_quiet(monkeypatch):
    """Regression: the first retap loop fired on silence rather than detachment.

    Observed live — it closed a correctly-attached tap and relaunched ProcTap's
    helper app every 3 s, forever, while the source sat idle. Reopening cannot fix
    silence; it can only fix being attached to the wrong pid or to nothing.
    """
    from tsutawaru.audio.capture_proc import ProcessCapture

    _patch_procs(monkeypatch, [
        _proc(102, CHROME_HELPER,
              [CHROME_HELPER, "--type=utility", sources.AUDIO_SERVICE_FLAG]),
    ])
    cap = ProcessCapture(sources.get("youtube"))

    class LiveTap:
        def is_running(self):
            return True

    cap._tap = LiveTap()
    cap.resolved = sources.Resolved(102, "audio-service", sources.get("youtube"))
    assert cap.needs_retap() is False  # quiet but correctly attached -> leave alone


def test_retap_when_the_audio_service_moved_to_a_new_pid(monkeypatch):
    """Chromium retires its audio service when idle and respawns it with a new pid."""
    from tsutawaru.audio.capture_proc import ProcessCapture

    _patch_procs(monkeypatch, [
        _proc(999, CHROME_HELPER,
              [CHROME_HELPER, "--type=utility", sources.AUDIO_SERVICE_FLAG]),
    ])
    cap = ProcessCapture(sources.get("youtube"))
    cap._tap = type("T", (), {"is_running": lambda self: True})()
    cap.resolved = sources.Resolved(102, "audio-service", sources.get("youtube"))
    assert cap.needs_retap() is True


def test_retap_when_no_tap_is_held(monkeypatch):
    from tsutawaru.audio.capture_proc import ProcessCapture

    _patch_procs(monkeypatch, [
        _proc(102, CHROME_HELPER,
              [CHROME_HELPER, "--type=utility", sources.AUDIO_SERVICE_FLAG]),
    ])
    cap = ProcessCapture(sources.get("youtube"))
    assert cap._tap is None
    assert cap.needs_retap() is True


def test_no_retap_when_the_source_has_nothing_to_offer(monkeypatch):
    """Don't close a live tap we would then be unable to reopen."""
    from tsutawaru.audio.capture_proc import ProcessCapture

    cap = ProcessCapture(sources.get("youtube"))
    cap._tap = type("T", (), {"is_running": lambda self: True})()
    cap.resolved = sources.Resolved(102, "audio-service", sources.get("youtube"))
    _patch_procs(monkeypatch, [])  # app vanished
    assert cap.needs_retap() is False


def test_dead_helper_is_detected(monkeypatch):
    """The helper is a separate .app and can die without us being told."""
    from tsutawaru.audio.capture_proc import ProcessCapture

    _patch_procs(monkeypatch, [
        _proc(102, CHROME_HELPER,
              [CHROME_HELPER, "--type=utility", sources.AUDIO_SERVICE_FLAG]),
    ])
    cap = ProcessCapture(sources.get("youtube"))
    cap._tap = type("T", (), {"is_running": lambda self: False})()
    cap.resolved = sources.Resolved(102, "audio-service", sources.get("youtube"))
    assert cap.needs_retap() is True


# ------------------------------------------------------------------ vad reset

def test_vad_reset_discards_the_previous_source():
    """Without this, switching mid-utterance emits one line containing two apps."""
    pytest.importorskip("silero_vad")
    from tsutawaru.audio.vad import VADGate
    from tsutawaru.config import VadCfg

    gate = VADGate(VadCfg())
    gate.feed(np.full(1000, 0.3, dtype=np.float32))  # leaves a residual
    gate.buf = [np.zeros(512, dtype=np.float32)]
    gate.in_speech = True
    gate.reset()
    assert gate.buf == []
    assert gate.in_speech is False
    assert gate._resid.size == 0
    assert len(gate.preroll) == 0
