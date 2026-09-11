#!/usr/bin/env python3
"""tsutawaru — live Japanese -> English audio translation.

Entrypoint and CLI. `--help` lists the full flag surface.
"""
from __future__ import annotations

import signal
import sys
import time
from typing import Optional

import typer

from tsutawaru.logbus import get_logger, metrics, setup_logging

app = typer.Typer(add_completion=False, no_args_is_help=False,
                  help="Live Japanese -> English audio translation.")
log = get_logger("tsutawaru")


def _print_devices() -> None:
    from tsutawaru.audio.devices import VIRTUAL_HINTS, list_devices

    devs = list_devices()
    w = max((len(d["name"]) for d in devs), default=20)
    print(f"{'idx':>3}  {'name'.ljust(w)}  {'in':>3} {'out':>3}  {'rate':>6}  api")
    print("-" * (w + 34))
    for d in devs:
        virtual = any(h in d["name"].lower() for h in VIRTUAL_HINTS)
        mark = " *" if virtual and d["in"] > 0 else "  "
        print(f"{d['index']:>3}{mark}{d['name'].ljust(w)}  "
              f"{d['in']:>3} {d['out']:>3}  {d['sr']:>6}  {d['api']}")
    print("\n* = virtual loopback device (what you want for system audio)")


def _print_sources() -> None:
    """List per-process capture sources and whether each is tappable right now."""
    from tsutawaru.audio import sources

    print(f"{'key':<10} {'label':<22} status")
    print("-" * 66)
    for src in sources.SOURCES.values():
        cands = sources.resolve_all(src)
        if not cands:
            print(f"{src.key:<10} {src.label:<22} not running")
            continue
        first = cands[0]
        print(f"{src.key:<10} {src.label:<22} ready — pid {first.pid} ({first.kind})")
        # Which process carries an app's audio is not predictable from it being
        # Chromium-based: Chrome uses its audio service, Discord uses its renderer.
        # Showing the fallback order makes an escalation in the log legible.
        rest = ", ".join(f"{c.pid} ({c.kind})" for c in cands[1:4])
        if rest:
            print(f"{'':<33}fallbacks: {rest}")
    print("\nUse with:  python run.py --source discord"
          "   (per-process capture is the default on macOS 14.4+)")


def _source_test(key: str, seconds: float) -> bool:
    """Tap one source and report levels. The per-process twin of --audio-test."""
    import numpy as np

    from tsutawaru.audio import sources
    from tsutawaru.audio.capture_proc import Float32FrameDecoder
    from tsutawaru.pipeline.queues import raw_q

    src = sources.get(key)
    try:
        r = sources.resolve(src)
    except sources.SourceUnavailable as e:
        print(f"Source : {src.label}\nResult : FAIL — {e.reason}")
        return False

    from tsutawaru.audio.capture_proc import ProcessCapture

    print(f"Source : {src.label} — pid {r.pid} ({r.kind})")
    print(f"Listening for {seconds:.0f}s — play audio in {src.label} now...")

    cap = ProcessCapture(src).start()
    dec = Float32FrameDecoder(cap.native_sr, cap.channels)
    frames: list = []
    import queue as _q
    import time as _t

    end = _t.monotonic() + seconds
    while _t.monotonic() < end:
        try:
            frames.append(dec.decode(raw_q.get(timeout=0.2)))
        except _q.Empty:
            continue
    cap.stop()

    audio = np.concatenate(frames) if frames else np.zeros(0, dtype=np.float32)
    if audio.size == 0:
        print("Frames : none — the tap delivered no packets at all")
        print()
        print(sources.source_help(src, seconds, got_frames=False))
        print("\nResult : FAIL — the tap delivered nothing")
        return False

    rms = float(np.sqrt((audio ** 2).mean()))
    peak = float(np.abs(audio).max())
    print(f"RMS    : {rms:.4f}   Peak: {peak:.4f}   ({audio.size} samples @ 16 kHz)")
    if peak <= 1e-4:
        # Packets arrived but carry no signal — a different fault from no packets
        # at all, and the only observable evidence of a missing Screen Recording
        # grant, since Core Audio reports no error for it.
        print()
        print(sources.source_help(src, seconds, got_frames=True))
        print("\nResult : FAIL — packets arrived but every sample was zero")
        return False
    if rms < 0.001:
        print("Result : WARN — signal present but very quiet")
        return True
    print("Result : OK — signal present, levels healthy")
    return True


def _report_missing(missing) -> None:
    if not missing:
        return
    print("\nThe following pipeline stages are not built yet:\n", file=sys.stderr)
    for m in missing:
        print(f"  - {m.module}  ({m.section})", file=sys.stderr)
    print("", file=sys.stderr)


@app.command()
def main(
    config: Optional[str] = typer.Option(None, "--config", help="Path to config.toml."),
    device: Optional[str] = typer.Option(None, "--device", help='Index, name substring, or "auto".'),
    list_devices: bool = typer.Option(False, "--list-devices", help="List audio devices and exit."),
    audio_test: bool = typer.Option(False, "--audio-test", help="Probe the device and report levels."),
    seconds: float = typer.Option(5.0, "--seconds", help="Duration for --audio-test/--source-test."),
    audio_backend: Optional[str] = typer.Option(
        None, "--audio-backend",
        help="auto|device|process (auto = per-process on macOS 14.4+, else device)."),
    source: Optional[str] = typer.Option(
        None, "--source", help="Per-process source to tap: discord|youtube."),
    list_sources: bool = typer.Option(
        False, "--list-sources", help="List per-process capture sources and exit."),
    source_test: Optional[str] = typer.Option(
        None, "--source-test", help="Tap one source, report levels, and exit."),
    model: Optional[str] = typer.Option(
        None, "--model",
        help="tiny|base|small|medium|kotoba (Japanese-specialised), "
             "or qwen3|qwen3-small (Qwen3-ASR, Apple Silicon only)."),
    compare: Optional[str] = typer.Option(
        None, "--compare",
        help="Run a second STT model on the same audio and show both (A/B)."),
    provider: Optional[str] = typer.Option(None, "--provider", help="google|deepl|local|none"),
    sink: Optional[str] = typer.Option(None, "--sink", help="window|console|overlay|both"),
    ws_port: Optional[int] = typer.Option(None, "--ws-port", help="WebSocket port for the OBS dock."),
    plain: bool = typer.Option(False, "--plain", help="Emit each block once, fully complete."),
    log_file: Optional[str] = typer.Option(None, "--log-file", help="Append the transcript here."),
    no_romaji: bool = typer.Option(False, "--no-romaji", help="Hide the romaji tier."),
    no_breakdown: bool = typer.Option(False, "--no-breakdown", help="Hide the word breakdown."),
    no_agglutinate: bool = typer.Option(False, "--no-agglutinate", help="Raw morphemes (debug Phase 3)."),
    tokenizer: Optional[str] = typer.Option(
        None, "--tokenizer", help="janome|mecab|unidic (unidic = fugashi + UniDic)."),
    record: bool = typer.Option(False, "--record", help="Write the session to a timestamped JSONL."),
    record_to: Optional[str] = typer.Option(None, "--record-to", help="Write the session JSONL here instead."),
    export_md: Optional[str] = typer.Option(None, "--export-md", help="Write a study-sheet Markdown on exit."),
    dump_utterances: Optional[str] = typer.Option(None, "--dump-utterances", help="Write each VAD chunk to WAV."),
    stats: bool = typer.Option(False, "--stats", help="Print a latency table on exit."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug logging."),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Warnings and errors only."),
):
    setup_logging(verbose=verbose, quiet=quiet)

    if list_devices:
        _print_devices()
        raise typer.Exit(0)

    if list_sources:
        _print_sources()
        raise typer.Exit(0)

    if audio_test:
        from tsutawaru.audio.devices import audio_test as probe

        try:
            result = probe(device or "auto", seconds)
        except Exception as e:
            log.error("%s", e)
            raise typer.Exit(2)
        raise typer.Exit(0 if result["ok"] else 1)

    if source_test:
        try:
            ok = _source_test(source_test, seconds)
        except Exception as e:
            log.error("%s", e)
            raise typer.Exit(2)
        raise typer.Exit(0 if ok else 1)

    from tsutawaru.config import load

    overrides = {
        "audio": {"device": device, "backend": audio_backend, "source": source},
        "stt": {"model": model, "compare_model": compare},
        "translate": {"provider": provider},
        "nlp": {
            "agglutinate": False if no_agglutinate else None,
            "tokenizer": tokenizer,
        },
        "ui": {
            "sink": sink,
            "ws_port": ws_port,
            "plain": True if plain else None,
            "log_file": log_file,
            "show_romaji": False if no_romaji else None,
            "show_breakdown": False if no_breakdown else None,
        },
    }
    try:
        cfg = load(config, overrides)
    except FileNotFoundError as e:
        log.error("%s", e)
        raise typer.Exit(2)

    problems = cfg.validate()
    if problems:
        for p in problems:
            log.error("config: %s", p)
        raise typer.Exit(2)

    from tsutawaru.pipeline.orchestrator import Orchestrator

    # --record names no file, so one is generated; --record-to overrides it.
    # Timestamped rather than fixed, because the alternative is a flag that
    # silently destroys the previous session every time it is used.
    record_path = record_to
    if record_path is None and record:
        record_path = f"experiments/sessions/tsutawaru-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    if record_path:
        log.info("recording session to %s", record_path)

    orch = Orchestrator(cfg, dump_utterances=dump_utterances,
                        record=record_path, export_md=export_md)
    stages = orch.build()

    if stages.capture is None:
        log.error("audio capture is unavailable — cannot start")
        _report_missing(stages.missing)
        raise typer.Exit(3)

    def _sigint(signum, frame):
        orch.stop()

    signal.signal(signal.SIGINT, _sigint)
    signal.signal(signal.SIGTERM, _sigint)

    _report_missing(stages.missing)

    try:
        orch.start()
        orch.wait()
    except KeyboardInterrupt:
        pass
    finally:
        orch.stop()
        if stats:
            from tsutawaru.pipeline import queues

            print("\n" + metrics.format_table({
                "uptime": f"{metrics.uptime():.0f}s",
                "dropped": str(queues.total_dropped()),
            }), file=sys.stderr)


if __name__ == "__main__":
    app()
