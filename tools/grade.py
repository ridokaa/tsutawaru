#!/usr/bin/env python
"""Fill in the verdict column of an accuracy-check sheet, one utterance at a time.

`accuracy-check.md` pairs 40 sampled lines with what tsutawaru heard and what it
translated, and leaves a verdict column empty. Filling that column is the only
thing standing between this project and a measured answer to the question it
keeps asking: *is the Japanese wrong, or is the English wrong?* Those two answers
point at opposite halves of the pipeline, and no amount of reading output settles
which one you have.

Doing it by hand means an `afplay` invocation, a scroll back to the table, and a
markdown cell edit per line — forty times. That friction is why the column has
been empty since 2026-08-23. This plays each clip, shows the row, takes one
keystroke, and writes the file back after every answer.

Verdicts:

    j    Japanese is wrong        — ASR failed; the translator never had a chance
    e    Japanese right, English wrong — ASR fine; the translator failed
    ok   both acceptable

Counting j against e tells you which half to spend money on. A sheet that comes
back mostly `j` means a better translation model buys you nothing.

Usage:
    python tools/grade.py                        # default sheet, resume where you left off
    python tools/grade.py --file path/to/sheet.md
    python tools/grade.py --tally                # print the score, grade nothing
"""
from __future__ import annotations

import argparse
import pathlib
import shutil
import subprocess
import sys
from dataclasses import dataclass

DEFAULT_SHEET = (
    pathlib.Path(__file__).resolve().parent.parent
    / "experiments"
    / "sessions"
    / "20260822-live"
    / "accuracy-check.md"
)

VERDICTS = {"j": "J", "e": "E", "ok": "OK", "o": "OK"}

HELP = """
  j   Japanese wrong (ASR)      r   replay
  e   English wrong (MT)        b   back one row
  ok  both fine                 s   skip, decide later
                                q   save and quit
"""


@dataclass
class Row:
    """One data row of the verdict table, kept with the line it came from."""

    lineno: int          # index into the file's line list
    cells: list[str]     # the six raw cells, padding intact
    num: str
    wav: str
    jp: str
    en: str
    conf: str

    @property
    def verdict(self) -> str:
        return self.cells[5].strip()

    def set(self, v: str) -> None:
        self.cells[5] = f" {v} "

    def render(self) -> str:
        return "|" + "|".join(self.cells) + "|"


def parse(lines: list[str]) -> list[Row]:
    """Data rows of the six-column verdict table.

    Recognised by the first cell being a number, which skips the header, the
    alignment row, and the Japanese request block above them without needing to
    know where in the document the table starts.
    """
    rows = []
    for i, line in enumerate(lines):
        if not line.startswith("|"):
            continue
        parts = line.split("|")
        if len(parts) != 8:
            continue
        cells = parts[1:-1]
        if not cells[0].strip().isdigit():
            continue
        rows.append(Row(
            lineno=i,
            cells=cells,
            num=cells[0].strip(),
            wav=cells[1].strip().strip("`"),
            jp=cells[2].strip(),
            en=cells[3].strip(),
            conf=cells[4].strip(),
        ))
    return rows


def save(path: pathlib.Path, lines: list[str], rows: list[Row]) -> None:
    """Rewrite the sheet in place.

    Called after every single answer rather than at the end: forty clips is long
    enough that an interrupted session is a real possibility, and losing the
    work to a Ctrl-C would guarantee the column stays empty for another fortnight.
    """
    for r in rows:
        lines[r.lineno] = r.render()
    text = "\n".join(lines)
    # `split("\n")` on a file ending in a newline yields a trailing "" that
    # rejoining reproduces, so the newline must not be added back on top of it —
    # otherwise every save grows the sheet by one blank line.
    path.write_text(text, encoding="utf-8")


def tally(rows: list[Row]) -> dict[str, int]:
    out = {"J": 0, "E": 0, "OK": 0, "": 0}
    for r in rows:
        out[r.verdict if r.verdict in out else ""] += 1
    return out


def report(rows: list[Row]) -> None:
    """J vs E is the whole answer: whichever is larger is the weak half."""
    t = tally(rows)
    done = t["J"] + t["E"] + t["OK"]
    print(f"\n  graded {done}/{len(rows)}   J {t['J']} · E {t['E']} · OK {t['OK']}")
    if t[""]:
        print(f"  {t['']} row(s) still unanswered — rerun to continue.")


def play(wav: pathlib.Path, player: str | None) -> None:
    if player is None:
        return
    try:
        subprocess.run([player, str(wav)], check=False)
    except OSError as e:
        print(f"  (could not play: {e})")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--file", type=pathlib.Path, default=DEFAULT_SHEET,
                    help="accuracy-check sheet to fill in")
    ap.add_argument("--wav-dir", type=pathlib.Path, default=None,
                    help="directory of utterance WAVs (default: sibling wav/)")
    ap.add_argument("--tally", action="store_true",
                    help="print the current score and exit")
    args = ap.parse_args()

    if not args.file.exists():
        print(f"no such sheet: {args.file}", file=sys.stderr)
        return 1

    lines = args.file.read_text(encoding="utf-8").split("\n")
    rows = parse(lines)
    if not rows:
        print(f"no verdict table found in {args.file}", file=sys.stderr)
        return 1

    if args.tally:
        report(rows)
        return 0

    wav_dir = args.wav_dir or args.file.parent / "wav"
    player = shutil.which("afplay") or shutil.which("aplay")
    if player is None:
        print("  no audio player found (afplay/aplay) — grading from text only\n")

    todo = [i for i, r in enumerate(rows) if not r.verdict]
    if not todo:
        print("every row already has a verdict. Blank a cell to redo it.")
        report(rows)
        return 0

    print(f"{len(todo)} row(s) to grade in {args.file.name}")
    print(HELP)

    pos = 0
    while 0 <= pos < len(todo):
        r = rows[todo[pos]]
        wav = wav_dir / f"{r.wav}.wav"
        print(f"\n─── {r.num}/{len(rows)}  {r.wav}  conf {r.conf}"
              f"{'' if wav.exists() else '  [wav missing]'}")
        print(f"  JP  {r.jp}")
        print(f"  EN  {r.en}")
        if r.verdict:
            print(f"  (currently {r.verdict})")
        if wav.exists():
            play(wav, player)

        while True:
            try:
                key = input("  [j/e/ok/r/b/s/q] ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print()
                save(args.file, lines, rows)
                report(rows)
                return 0
            if key in VERDICTS:
                r.set(VERDICTS[key])
                save(args.file, lines, rows)
                pos += 1
                break
            if key == "r":
                play(wav, player) if wav.exists() else print("  (no wav)")
                continue
            if key == "s":
                pos += 1
                break
            if key == "b":
                pos = max(0, pos - 1)
                break
            if key == "q":
                save(args.file, lines, rows)
                report(rows)
                return 0
            print(HELP)

    save(args.file, lines, rows)
    report(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
