#!/usr/bin/env python
"""Run the 2x2 model matrix over recorded data and write a sheet to grade.

Four combinations, two variables:

    A  qwen3  + google      C  qwen3  + local
    B  kotoba + google      D  kotoba + local

Two input modes, because the project has two kinds of recording.

`--session FILE.jsonl` (no audio needed)
    A session recorded with `compare_model` already holds both transcripts *and*
    both online translations — `jp`/`en` and `alt.jp`/`alt.en`. Cells A and B are
    therefore already on disk, and are read rather than recomputed: they are what
    the live system actually produced, which no re-run can reproduce exactly.
    Only C and D are generated, by translating the two recorded transcripts
    locally. No ASR, no network, ~130 ms per cell.

`--wav DIR` (full matrix from audio)
    Runs both ASR models and both providers from scratch. Each model transcribes
    each clip once and each distinct transcript is translated once per provider,
    so the cost is 2 ASR + 2 MT per clip, not 4 + 4.

The breakdown tier is deliberately absent from the sheet: glosses come from
JMdict in all four cells, so they are a constant here, not a variable.

Nothing is scored automatically. Latency and disagreement rates are measured;
which English is *better* is a judgement, and this writes the sheet for a human
to make it on. See tools/grade.py for the same discipline on the ASR/MT split.

Usage:
    python tools/compare4.py --session experiments/sessions/tsutawaru-20260904-183501.jsonl
    python tools/compare4.py --wav "experiments/sessions/20260822-live/wav"
    python tools/compare4.py --session S.jsonl --all -o sheet.md
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import time
import wave

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from tsutawaru.config import load  # noqa: E402

CELLS = [("A", "qwen3", "google"), ("B", "kotoba", "google"),
         ("C", "qwen3", "local"), ("D", "kotoba", "local")]


class Timed:
    """Translate through one backend, deduplicating and recording latency.

    The same short utterance ("はい", "うん") recurs constantly in a call, and
    re-translating it would inflate both the runtime and the latency sample with
    work the live pipeline's cache would never have done.
    """

    def __init__(self, backend):
        self.backend = backend
        self.seen: dict[str, str] = {}
        self.ms: list[float] = []

    def __call__(self, text: str) -> str:
        text = (text or "").strip()
        if not text:
            return ""
        if text in self.seen:
            return self.seen[text]
        t = time.perf_counter()
        try:
            out = self.backend.sentence(text)
        except Exception as e:
            out = f"[{type(e).__name__}]"
        self.ms.append((time.perf_counter() - t) * 1000)
        self.seen[text] = out
        return out


def _sample(items: list, limit: int | None) -> list:
    """Evenly spaced, not the first N — a session's opening lines are all
    greetings and would make a biased sheet."""
    if limit is None or len(items) <= limit:
        return items
    step = len(items) / limit
    return [items[int(i * step)] for i in range(limit)]


def _mk(provider: str, cfg):
    if provider == "local":
        from tsutawaru.translate.local_mlx import LocalTranslator

        return LocalTranslator(cfg.translate)
    from dataclasses import replace as dc_replace

    from tsutawaru.translate.online import OnlineTranslator

    return OnlineTranslator(dc_replace(cfg.translate, provider="google"))


def from_session(path: pathlib.Path, limit: int | None, cfg) -> tuple[list[dict], dict]:
    rows_in = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        alt = d.get("alt") or {}
        if d.get("partial") or not d.get("jp") or not alt.get("jp"):
            continue
        rows_in.append(d)

    if not rows_in:
        raise SystemExit(
            f"{path} has no lines with a comparison lane. Record one with "
            "`--model qwen3` and `[stt] compare_model = \"kotoba\"`, or use --wav."
        )
    models = (rows_in[0].get("model") or "?", (rows_in[0].get("alt") or {}).get("model") or "?")
    rows_in = _sample(rows_in, limit)

    print(f"[compare4] {len(rows_in)} lines from {path.name}; "
          f"cells A/B read from the recording ({models[0]} / {models[1]} + google)")
    local = Timed(_mk("local", cfg))

    out = []
    for i, d in enumerate(rows_in, 1):
        alt = d["alt"]
        out.append({
            "n": i,
            "ref": f"line {d.get('id', '?')}",
            "A": (d["jp"], d.get("en", "")),
            "B": (alt["jp"], alt.get("en", "")),
            "C": (d["jp"], local(d["jp"])),
            "D": (alt["jp"], local(alt["jp"])),
        })
        if i % 25 == 0:
            print(f"  {i}/{len(rows_in)}", flush=True)
    return out, {"local": local.ms, "asr": {}, "models": models}


def from_wav(d: pathlib.Path, limit: int | None, cfg) -> tuple[list[dict], dict]:
    from dataclasses import replace as dc_replace

    from tsutawaru.stt.factory import build_engine

    wavs = _sample(sorted(d.glob("*.wav")), limit)
    if not wavs:
        raise SystemExit(f"no .wav files in {d}")
    print(f"[compare4] {len(wavs)} clips from {d}")

    engines, asr_ms = {}, {}
    for name in ("qwen3", "kotoba"):
        e = build_engine(dc_replace(cfg.stt, model=name))
        e.warmup()
        engines[name] = e
        asr_ms[name] = []
    google, local = Timed(_mk("google", cfg)), Timed(_mk("local", cfg))

    out = []
    for i, w in enumerate(wavs, 1):
        with wave.open(str(w)) as fh:
            pcm = np.frombuffer(fh.readframes(fh.getnframes()), dtype=np.int16)
        audio = pcm.astype(np.float32) / 32768.0

        jp = {}
        for name, eng in engines.items():
            t = time.perf_counter()
            jp[name] = (eng.transcribe(audio).text or "").strip()
            asr_ms[name].append((time.perf_counter() - t) * 1000)
        if not any(jp.values()):
            continue  # silence or a rejected clip — nothing to compare

        out.append({
            "n": len(out) + 1,
            "ref": f"`{w.name}`",
            "A": (jp["qwen3"], google(jp["qwen3"])),
            "B": (jp["kotoba"], google(jp["kotoba"])),
            "C": (jp["qwen3"], local(jp["qwen3"])),
            "D": (jp["kotoba"], local(jp["kotoba"])),
        })
        if i % 10 == 0:
            print(f"  {i}/{len(wavs)}", flush=True)
    return out, {"local": local.ms, "google": google.ms, "asr": asr_ms, "models": ("qwen3", "kotoba")}


_PUNCT = re.compile(r"[\s。、，．！？!?,.・…~〜「」『』()（）\"']+")


def _same(a: str, b: str) -> bool:
    """Equal ignoring punctuation, spacing and case.

    Raw string equality is useless here: qwen3 terminates every utterance with 。
    and kotoba never does, so a raw comparison called the two ASR models
    different on 859 of 859 lines — a number that says nothing about either.
    Ignoring punctuation puts the real disagreement at 73%.
    """
    return _PUNCT.sub("", a).lower() == _PUNCT.sub("", b).lower()


def _pct(v: list[float], q: float) -> float:
    if not v:
        return float("nan")
    v = sorted(v)
    return v[min(int(len(v) * q), len(v) - 1)]


def write_sheet(rows: list[dict], stats: dict, src: str, path: pathlib.Path) -> None:
    asr_split = sum(1 for r in rows if not _same(r["A"][0], r["B"][0]))
    mt_split = sum(1 for r in rows
                   if not _same(r["A"][1], r["C"][1]) or not _same(r["B"][1], r["D"][1]))
    n = len(rows) or 1

    L = [
        f"# 4-way comparison — {src}",
        "",
        f"{len(rows)} lines. Cells: **A** qwen3+google · **B** kotoba+google · "
        "**C** qwen3+local · **D** kotoba+local.",
        "",
        "The breakdown tier is not compared: glosses come from JMdict in all four "
        "cells, so it is held constant.",
        "",
        "## How to grade",
        "",
        "Put an `x` in the ✓ column of the row whose **English** you would rather "
        "have had in the call. Tie between two rows — mark both. All four bad — "
        "mark none. The counts at the bottom of this file are filled in by "
        "re-running with `--tally`.",
        "",
        "Read the Japanese column too: a row can only be as good as the transcript "
        "above it, and separating those two failures is the whole point "
        "(see tools/grade.py).",
        "",
        "## Measured (not judged)",
        "",
        "| | p50 | p95 |",
        "|---|---|---|",
    ]
    for name, v in (("ASR qwen3", stats["asr"].get("qwen3")),
                    ("ASR kotoba", stats["asr"].get("kotoba")),
                    ("MT google", stats.get("google")),
                    ("MT local", stats.get("local"))):
        if v:
            L.append(f"| {name} | {_pct(v, .5):.0f} ms | {_pct(v, .95):.0f} ms |")
    L += [
        "",
        f"- The two ASR models disagreed on **{asr_split}/{len(rows)}** lines "
        f"({100 * asr_split / n:.0f}%). Where they agree, rows A/B and C/D carry "
        "the same Japanese and only the translator differs.",
        f"- The two providers produced different English on **{mt_split}/{len(rows)}** "
        f"lines ({100 * mt_split / n:.0f}%).",
        "",
        "Both comparisons ignore punctuation, spacing and case: qwen3 ends every "
        "utterance with 。 and kotoba never does, which on a raw comparison makes "
        "the two models look 100% different on every sample.",
        "",
        "---",
        "",
    ]

    for r in rows:
        L += [f"### {r['n']} · {r['ref']}", "",
              "| | ASR | MT | Japanese | English | ✓ |",
              "|---|---|---|---|---|---|"]
        for key, asr, mt in CELLS:
            jp, en = r[key]
            L.append(f"| {key} | {asr} | {mt} | {jp or '—'} | {en or '—'} |  |")
        L.append("")

    path.write_text("\n".join(L) + "\n")
    print(f"[compare4] wrote {path} ({len(rows)} lines)")


def tally(path: pathlib.Path) -> None:
    """Count the ✓ marks per cell, and per variable."""
    counts = {k: 0 for k, _, _ in CELLS}
    graded = set()
    for line in path.read_text().splitlines():
        parts = line.split("|")
        if len(parts) != 8:
            continue
        key = parts[1].strip()
        if key not in counts:
            continue
        if parts[6].strip().lower() == "x":
            counts[key] += 1
            graded.add(id(line))
    total = sum(counts.values())
    if not total:
        print("nothing marked yet — put an x in the ✓ column of the best row")
        return
    print(f"marks: {total}")
    for key, asr, mt in CELLS:
        print(f"  {key}  {asr:7} + {mt:7}  {counts[key]:4}  ({100 * counts[key] / total:.0f}%)")
    qwen = counts["A"] + counts["C"]
    google = counts["A"] + counts["B"]
    print(f"\n  by ASR: qwen3 {qwen} vs kotoba {total - qwen}")
    print(f"  by MT:  google {google} vs local {total - google}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--session", type=pathlib.Path, help="a recorded JSONL with a compare lane")
    ap.add_argument("--wav", type=pathlib.Path, help="a directory of utterance WAVs")
    ap.add_argument("-o", "--out", type=pathlib.Path, default=pathlib.Path("compare4-sheet.md"))
    ap.add_argument("-n", "--limit", type=int, default=100, help="lines to sample (default 100)")
    ap.add_argument("--all", action="store_true", help="use every line, no sampling")
    ap.add_argument("--tally", action="store_true", help="count an existing sheet's marks and exit")
    a = ap.parse_args()

    if a.tally:
        tally(a.out)
        return 0
    if bool(a.session) == bool(a.wav):
        ap.error("give exactly one of --session or --wav")

    cfg = load()
    limit = None if a.all else a.limit
    t = time.time()
    if a.session:
        rows, stats = from_session(a.session, limit, cfg)
        src = a.session.name
    else:
        rows, stats = from_wav(a.wav, limit, cfg)
        src = str(a.wav)
    write_sheet(rows, stats, src, a.out)
    print(f"[compare4] {time.time() - t:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
