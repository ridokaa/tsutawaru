"""Audio device enumeration, selection, and the --audio-test probe.

Devices are resolved by **name**, never by cached index: indices renumber every
time anything is plugged or unplugged (plan §6.6).
"""
from __future__ import annotations

import os
import platform
import subprocess
import sys

from tsutawaru.logbus import get_logger

log = get_logger(__name__)

TARGET_SR = 16000

# Substrings that identify a virtual loopback device across platforms.
VIRTUAL_HINTS = (
    "blackhole", "cable output", "vb-audio", "soundflower",
    "voicemeeter", "loopback", "multi-output",
)


def list_devices() -> list[dict]:
    import sounddevice as sd

    out = []
    for i, d in enumerate(sd.query_devices()):
        api = sd.query_hostapis(d["hostapi"])["name"]
        out.append({
            "index": i,
            "name": d["name"],
            "api": api,
            "in": d["max_input_channels"],
            "out": d["max_output_channels"],
            "sr": int(d["default_samplerate"]),
        })
    return out


def autodetect() -> int | None:
    """Prefer a known virtual loopback device over the physical mic."""
    for d in list_devices():
        if d["in"] > 0 and any(h in d["name"].lower() for h in VIRTUAL_HINTS):
            return d["index"]
    return None


def resolve(spec: str | int | None) -> int:
    """Resolve "auto" | index | name-substring to a concrete input device index."""
    devs = [d for d in list_devices() if d["in"] > 0]
    if not devs:
        raise RuntimeError("no audio input devices found")

    if spec is None or spec == "auto":
        idx = autodetect()
        if idx is None:
            import sounddevice as sd

            idx = sd.default.device[0]
            log.warning(
                "no virtual loopback device found — falling back to the default "
                "input (%s). System audio will NOT be captured; see plan §2.",
                next((d["name"] for d in devs if d["index"] == idx), idx),
            )
        return int(idx)

    if isinstance(spec, int) or (isinstance(spec, str) and spec.isdigit()):
        idx = int(spec)
        if not any(d["index"] == idx for d in devs):
            raise ValueError(f"device index {idx} is not an input device")
        return idx

    needle = str(spec).lower()
    matches = [d for d in devs if needle in d["name"].lower()]
    if not matches:
        names = ", ".join(repr(d["name"]) for d in devs)
        raise ValueError(f"no input device matching {spec!r}. Available: {names}")
    if len(matches) > 1:
        log.warning(
            "%r matches %d devices; using %r",
            spec, len(matches), matches[0]["name"],
        )
    return matches[0]["index"]


def default_output_name() -> str | None:
    """Current system default output device (macOS only), or None.

    Used by the §6.6 silent-stream diagnostic: on macOS the likeliest cause of a
    silent capture is that system output is no longer routed to the virtual
    device, which happens every time headphones are plugged or unplugged.
    """
    if platform.system() != "Darwin":
        return None
    try:
        out = subprocess.run(
            ["system_profiler", "SPAudioDataType"],
            capture_output=True, text=True, timeout=10,
        ).stdout
    except Exception:  # pragma: no cover - system_profiler missing/slow
        return None
    current = None
    for line in out.splitlines():
        stripped = line.strip()
        # Device names sit at 8-space indent and end with ':'
        if line.startswith(" " * 8) and not line.startswith(" " * 10) and stripped.endswith(":"):
            current = stripped[:-1]
        elif stripped == "Default Output Device: Yes":
            return current
    return None


def owning_app() -> str | None:
    """The GUI application macOS holds the microphone grant against.

    TCC is scoped to the *parent application*, not to the python binary. A
    process launched from an IDE's integrated terminal inherits that IDE's
    grant — so the same script captures audio from iTerm and silently receives
    all-zero samples from an editor that was never granted access.
    """
    if platform.system() != "Darwin":
        return None
    try:
        pid = os.getppid()
        for _ in range(8):  # walk up until we hit a bundled .app
            out = subprocess.run(
                ["ps", "-o", "ppid=,comm=", "-p", str(pid)],
                capture_output=True, text=True, timeout=5,
            ).stdout.strip()
            if not out:
                return None
            ppid_s, _, comm = out.partition(" ")
            comm = comm.strip()
            if ".app/" in comm:
                app = comm.split(".app/")[0].split("/")[-1]
                return app
            pid = int(ppid_s)
            if pid <= 1:
                return None
    except Exception:
        return None
    return None


def silent_stream_help(device_name: str, seconds: float = 5.0) -> str:
    """The §6.6 message: both causes of an all-zero stream, likeliest first.

    Both causes are checked rather than merely listed. If output routing is
    confirmed good, an exactly-zero stream points hard at a denied microphone
    grant — macOS returns silence with no error — so that cause is promoted and
    the *parent application* holding the grant is named. Listing a cause we can
    see is fine would send the user to the wrong fix, which is the defect this
    message exists to avoid.
    """
    lines = [
        f'No signal from "{device_name}" for {seconds:.0f}s — the stream is open but silent.',
        "",
    ]
    if sys.platform == "darwin":
        current = default_output_name()
        routed_ok = bool(current) and any(h in current.lower() for h in VIRTUAL_HINTS)
        cause_routing = [
            "System output is not routed to your virtual device.",
            "     macOS switches output away whenever you plug/unplug headphones.",
            "     -> menu bar sound icon -> select your Multi-Output Device",
        ]
        cause_perm = [
            "Microphone permission may not be granted.",
            "     macOS treats every input device as a microphone, incl. BlackHole.",
            "     -> System Settings -> Privacy & Security -> Microphone",
        ]
        if routed_ok:
            # Routing is confirmed good, so an exactly-zero stream makes a denied
            # microphone grant MORE likely, not less. macOS delivers silence with
            # no error when permission is missing, and the grant belongs to the
            # parent application — commonly an IDE's integrated terminal.
            app = owning_app()
            lines += [f"  Output routing looks OK (system default: {current})."]
            lines += ["", "  1. Microphone permission is the likely cause."]
            if app:
                lines += [
                    f"     This process was launched by {app!r}, and macOS grants",
                    f"     microphone access per app — so {app} needs it, not python.",
                    "     -> System Settings -> Privacy & Security -> Microphone",
                    f"        -> enable {app}, then restart it",
                    "",
                    "     Quick check: run the same command from Terminal or iTerm.",
                    "     If it works there, it is definitely this.",
                ]
            else:
                lines += [
                    "     macOS treats every input device as a microphone, incl. BlackHole,",
                    "     and grants access per parent app rather than to python.",
                    "     -> System Settings -> Privacy & Security -> Microphone",
                ]
            lines += [
                "",
                "  2. Nothing may actually be playing.",
                "     A correctly-routed device is silent when the source is idle.",
            ]
        else:
            here = f" (currently: {current})" if current else ""
            lines += ["  1. " + cause_routing[0].replace(".", here + "."), *cause_routing[1:]]
            lines += ["", "  2. " + cause_perm[0], *cause_perm[1:]]
    elif sys.platform == "win32":
        lines += [
            "  1. The source app may not be playing to your virtual cable.",
            "     -> Discord -> Voice & Video -> Output Device -> CABLE Input",
            "",
            "  2. Sample rates may not match.",
            "     -> Set CABLE Input AND CABLE Output to 48000 Hz (plan §2.2)",
        ]
    else:
        lines += ["  Check that your source application is routed to this device."]
    lines += ["", f'  Re-test with:  python run.py --audio-test --device "{device_name}"']
    return "\n".join(lines)


def audio_test(spec: str | int | None, seconds: float = 5.0) -> dict:
    """Open the device, measure levels, and report. Plan §2.3."""
    import numpy as np
    import sounddevice as sd

    idx = resolve(spec)
    info = sd.query_devices(idx)
    sr = int(info["default_samplerate"])
    ch = min(2, max(1, int(info["max_input_channels"])))

    print(f"Device : {info['name']} @ {sr} Hz -> resampled {TARGET_SR} Hz")
    if sys.platform == "darwin":
        out_name = default_output_name()
        if out_name:
            print(f"Output : system default is {out_name!r}")

    frames: list = []
    def cb(indata, n, t, status):
        frames.append(indata.copy())

    print(f"Listening for {seconds:.0f}s — play some audio now...")
    with sd.InputStream(device=idx, samplerate=sr, channels=ch,
                        dtype="float32", callback=cb):
        sd.sleep(int(seconds * 1000))

    if not frames:
        raise RuntimeError("no audio frames were delivered by the driver")

    a = np.concatenate(frames)
    mono = a.mean(axis=1) if a.ndim > 1 else a
    rms = float(np.sqrt(np.mean(mono ** 2)))
    peak = float(np.abs(mono).max())
    clip = float(np.mean(np.abs(mono) > 0.99) * 100.0)

    print(f"RMS    : {rms:.4f}   Peak: {peak:.4f}   Clipping: {clip:.2f}%")

    ok = peak > 0.0
    if not ok:
        print()
        print(silent_stream_help(info["name"], seconds))
        print()
        print("Result : FAIL - stream opened but delivered only zeros")
    elif clip > 1.0:
        print("Result : WARN - clipping detected, lower the source volume to ~70%")
    elif rms < 0.001:
        print("Result : WARN - signal present but very quiet; raise the source volume")
    else:
        print("Result : OK - signal present, levels healthy")

    return {"device": info["name"], "samplerate": sr, "channels": ch,
            "rms": rms, "peak": peak, "clipping_pct": clip, "ok": ok}
