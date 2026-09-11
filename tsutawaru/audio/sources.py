"""Named per-process audio sources for the ProcTap capture backend.

Used when `[audio] backend` resolves to `"process"`, which is what `"auto"` picks
on macOS 14.4+. The device backend in `devices.py` is the fallback for every other
platform and is untouched by this module.

Why this module exists at all: `devices.py` resolves an *input device* by name,
and a process is not a device. Nothing in that file applies here, so source
identity gets its own resolver rather than a special case bolted onto `resolve()`.

The one fact that shapes everything below: **a Core Audio process tap is bound to
exactly one pid.** ProcTap's helper builds it with

    CATapDescription(stereoMixdownOfProcesses: [processObjectID])

which has no process-tree option. Chromium-based apps — Discord and Chrome both
— do not play audio from their main process; they spawn a dedicated utility
process for it. So tapping the pid of "Discord" or "Google Chrome" captures
silence forever, and the pid that actually matters is a child that:

  * does not exist until the app plays something,
  * is torn down again when the app goes idle,
  * gets a new pid every time it comes back.

That is why resolution happens at every start rather than once at build time.
"""
from __future__ import annotations

from dataclasses import dataclass

from tsutawaru.logbus import get_logger

log = get_logger(__name__)

# Chromium's audio output process. Identical in Chrome and in every Electron app
# (verified against Chrome 151, Discord, and an unrelated Electron IDE).
AUDIO_SERVICE_FLAG = "--utility-sub-type=audio.mojom.AudioService"

# Any Chromium child carries --type=; the main process does not.
CHILD_FLAG = "--type="
# Discord renders call audio from its renderer rather than from Chromium's audio
# service, so "renderer" has to be a first-class role. It was originally lumped in
# with the generic --type= children, which made it unreachable by name: a `roles`
# entry of "renderer" matched nothing and Discord's real audio process sorted last.
RENDERER_FLAG = "--type=renderer"


@dataclass(frozen=True)
class Source:
    """A user-facing capture source, identified by the app that owns the audio."""

    key: str  # stable id used in config and on the CLI
    label: str  # what the UI shows
    bundles: tuple[str, ...]  # executable-path fragments identifying the app
    # Which process role actually renders this app's audio, best guess first.
    # Being Chromium-based says nothing about this, which was the surprise:
    # measured on this machine, Chrome's audio is on audio.mojom.AudioService and
    # Discord's is on its *renderer*, with Discord's audio service delivering
    # exactly zero bytes. Discord's voice engine is a native module that opens its
    # own Core Audio stream instead of going through Chromium's audio service.
    # Order is a starting point, not a promise — capture escalates through the rest
    # of the list if the first choice stays silent, so an app that moves its audio
    # in an update self-corrects instead of silently capturing nothing.
    roles: tuple[str, ...] = ("audio-service", "renderer", "main")


SOURCES: dict[str, Source] = {
    "discord": Source(
        "discord", "Discord", ("/Discord.app/",),
        roles=("renderer", "audio-service", "main"),
    ),
    "youtube": Source(
        "youtube", "YouTube (Chrome)", ("/Google Chrome.app/", "/Chromium.app/"),
        roles=("audio-service", "renderer", "main"),
    ),
}

DEFAULT_SOURCE = "discord"


@dataclass(frozen=True)
class Resolved:
    """A concrete, tappable pid."""

    pid: int
    kind: str  # "audio-service" | "main"
    source: Source

    @property
    def label(self) -> str:
        return f"{self.source.label} (pid {self.pid})"


class SourceUnavailable(RuntimeError):
    """The source has no tappable pid *right now*.

    Not a configuration error and not fatal: the overwhelmingly common cause is
    that the app simply is not playing audio yet, which is a state the capture
    layer is expected to sit and wait through.
    """

    def __init__(self, source: Source, reason: str):
        self.source, self.reason = source, reason
        super().__init__(f"{source.label}: {reason}")


def get(key: str) -> Source:
    """Look up a source by key, falling back to the default with a warning."""
    src = SOURCES.get(key)
    if src is None:
        log.warning(
            "unknown audio source %r — using %r (known: %s)",
            key, DEFAULT_SOURCE, ", ".join(sorted(SOURCES)),
        )
        return SOURCES[DEFAULT_SOURCE]
    return src


def _iter_procs():
    """psutil process iterator, or empty when psutil is absent."""
    try:
        import psutil
    except ImportError:
        log.warning("psutil is not installed — per-process capture cannot resolve pids")
        return
    # exe/cmdline raise for processes that die mid-iteration; psutil's
    # attrs= form returns None for those fields instead of throwing.
    yield from psutil.process_iter(["pid", "name", "exe", "cmdline"])


def candidates(src: Source) -> list[dict]:
    """Every live process belonging to `src`, audio service first.

    Ordering is the resolution preference, so callers can take the first entry.
    """
    out: list[dict] = []
    for p in _iter_procs():
        info = p.info
        exe = info.get("exe") or ""
        if not any(b in exe for b in src.bundles):
            continue
        cmdline = info.get("cmdline") or []
        joined = " ".join(cmdline)
        if AUDIO_SERVICE_FLAG in joined:
            kind = "audio-service"
        elif RENDERER_FLAG in joined:
            kind = "renderer"
        elif CHILD_FLAG in joined:
            kind = "helper"
        else:
            kind = "main"
        out.append({
            "pid": info["pid"],
            "name": info.get("name") or "?",
            "kind": kind,
            "exe": exe,
        })
    # "helper" is every other --type= child (gpu, network, video capture). Ranked
    # last rather than dropped: it costs nothing to keep as a final escalation
    # step, and it is the only safety net if an app renders audio from somewhere
    # none of the named roles cover.
    def rank(c: dict) -> tuple[int, int]:
        try:
            r = src.roles.index(c["kind"])
        except ValueError:
            r = len(src.roles)
        return (r, c["pid"])

    out.sort(key=rank)
    return out


def resolve_all(src: Source) -> list[Resolved]:
    """Every tappable pid for `src`, most likely to carry audio first.

    A list rather than one answer because which process renders an app's audio is
    not knowable in advance — Core Audio will not let an unentitled process
    enumerate audio objects (`kAudioHardwarePropertyProcessObjectList` returns
    'who?'), and a tap on the wrong pid opens successfully and then delivers
    silence. Trying candidates in turn is the only mechanism available.
    """
    return [Resolved(c["pid"], c["kind"], src) for c in candidates(src)]


def resolve(src: Source, index: int = 0) -> Resolved:
    """The `index`-th most likely pid for `src`. Raises SourceUnavailable.

    `index` is the escalation counter: capture bumps it when a tap stays silent, so
    a wrong first guess is temporary rather than permanent. It wraps, so escalating
    forever eventually returns to the best guess instead of dead-ending.
    """
    cands = resolve_all(src)
    if not cands:
        raise SourceUnavailable(src, "the application is not running")
    return cands[index % len(cands)]


_PERM_FIX = [
    '  -> System Settings -> Privacy & Security -> Screen Recording',
    '     -> enable "ProcTap Helper"',
    '  -> System Settings -> Privacy & Security -> Microphone',
    '     -> enable "ProcTap Helper"',
    "",
    "     The grant belongs to the signed helper app, not to python, so it",
    "     survives restarts and does not depend on which terminal launched us.",
]


def _output_device_note() -> list[str]:
    """The cause that only bites apps with their own output-device picker.

    ProcTap's helper wraps the process tap in an aggregate device whose master and
    sole subdevice is the **current system default output** (main.swift builds it
    from `defaultOutputID`). An application rendering to some *other* device is
    therefore outside the tap's graph and captures as nothing at all — while the
    audio is still perfectly audible, because it is going to a real device.

    This is why an app can be plainly audible and still produce an empty tap, and
    why browsers rarely hit it: Chrome follows the system default, whereas Discord,
    OBS, Zoom and most voice apps ship their own output selector and remember a
    device chosen long ago.
    """
    lines = [
        "     The tap is built on top of your **system default output** device, so an",
        "     app pointed at any other device is invisible to it — audible to you,",
        "     silent to us.",
    ]
    try:
        from tsutawaru.audio.devices import default_output_name

        current = default_output_name()
    except Exception:  # pragma: no cover - diagnostic must never raise
        current = None
    if current:
        lines += [
            f"     Your system default output is currently: {current!r}.",
            f"     -> set the app's own output device to {current!r} (or to 'Default')",
        ]
    else:
        lines += ["     -> set the app's own output device to match your system default"]
    return lines


def source_help(src: Source, silent_s: float, got_frames: bool = False) -> str:
    """Why a per-process tap is delivering nothing, likeliest cause first.

    Mirrors `devices.silent_stream_help`: check what can be checked and rank only
    the causes that survive. Four failure modes reach here with four different
    fixes, and a message listing all of them every time would send the user to the
    wrong one — the defect that message exists to avoid.

    `got_frames` is the discriminator that makes the last two separable, and it was
    established empirically rather than guessed:

      * no packets at all  -> the app holds an audio object but is rendering
                              nothing, i.e. it is simply idle.
      * packets, ~0 amplitude -> Core Audio built the tap without Screen Recording
                              consent and is feeding it silence. Nothing errors at
                              any layer, so this signature is the only evidence.
    """
    lines = [
        f'No audio from "{src.label}" for {silent_s:.0f}s '
        "— the per-process tap is open but silent.",
        "",
    ]
    cands = candidates(src)
    if not cands:
        lines += [
            f"  {src.label} is not running.",
            "     Start it, or switch source. Nothing else is wrong.",
        ]
        return "\n".join(lines)

    svc = [c for c in cands if c["kind"] == "audio-service"]
    if not svc:
        lines += [
            f"  {src.label} is running but has no audio process yet.",
            "     Chromium apps spawn their audio service only while something is",
            "     actually playing, and retire it again when idle. Play audio in",
            f"     {src.label} and capture starts on its own — no action needed.",
        ]
        return "\n".join(lines)

    pid = svc[0]["pid"]
    if got_frames:
        # The tap is alive and clocking; only the content is missing.
        lines += [
            f"  Tapping {src.label} pid {pid}. Audio packets ARE arriving, but every",
            "  sample is zero — the signature of a missing Screen Recording grant.",
            "",
            "  Core Audio creates a process tap successfully without that consent and",
            "  then feeds it pure silence. There is no error to catch anywhere.",
            "",
            *_PERM_FIX,
        ]
    else:
        lines += [
            f"  Tapping {src.label} pid {pid}, but no audio packets are arriving at all.",
            "",
            f"  1. {src.label} may simply not be playing anything right now.",
            "     A process tap goes quiet when its target renders nothing, and the",
            "     audio service lingers for a while after playback stops. Start",
            "     playback and capture resumes on its own.",
            "",
            f"  2. If you CAN hear {src.label}, its output device does not match yours.",
            *_output_device_note(),
            "",
            "  3. Otherwise, ProcTap's helper lacks permission.",
            *_PERM_FIX,
        ]
    lines += ["", f"  Re-test with:  python run.py --source-test {src.key}"]
    return "\n".join(lines)
