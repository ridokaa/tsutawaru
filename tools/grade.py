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
    python tools/grade.py --rows 15,28,3-9       # re-check named rows, answered or not
    python tools/grade.py --ab --file sheet.md   # blind A/B: which transcript is closer

`--ab` grades a sheet whose JP and EN columns hold two competing versions of the
same clip, A and B, sides shuffled per row so the grader cannot tell which arm is
which. The verdict is which one is closer to what the clip says; the key that
unblinds it lives beside the sheet, not in it.

`--key` writes an answer key instead of a verdict: for each clip of the sheet,
who the line is about and the English that is right, picked from the versions
already translated (shown unlabelled, shuffled) or typed — pasted and fixed for
a near-miss. (Pre-filling the line for editing was tried; macOS's libedit
readline ignores it.) A verdict only ranks the two outputs it saw; the key
scores any future output too, with tools/score.py, so a translator change no
longer needs a new sheet.

    python tools/grade.py --key                  # the 40-row sheet, resume
    python tools/grade.py --key --rows 5,30      # redo named rows
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
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

CANDIDATES = DEFAULT_SHEET.parents[2] / "reports" / "frag-eval-20261007.json"

VERDICTS = {"j": "J", "e": "E", "ok": "OK", "o": "OK"}
VERDICTS_AB = {"1": "A", "2": "B", "0": "="}
# Who the line is about. Japanese usually leaves it unsaid, and filling it in
# wrongly is the translator's most common error, so it is decided before the
# English is chosen. "skip" keeps a clip out of scoring (bad Japanese, no idea).
SUBJECTS = {"n": "none", "m": "me", "y": "you", "o": "other", "x": "skip"}


def _spec(s: str) -> set[str]:
    """Row numbers from "15,28,3-9". Empty means "every unanswered row".

    Re-grading named rows is the normal case once a sheet is full: a verdict
    reached by comparing models is provisional, and the rows worth an ear are
    the handful the notes flag as weak. Without this the only way back into a
    graded sheet is to blank cells in markdown by hand, which is the friction
    this tool exists to remove.
    """
    out: set[str] = set()
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        lo, _, hi = part.partition("-")
        out.update(str(n) for n in range(int(lo), int(hi or lo) + 1))
    return out

HELP = """
  j   Japanese wrong (ASR)      r   replay
  e   English wrong (MT)        b   back one row
  ok  both fine                 s   skip, decide later
                                q   save and quit
"""

HELP_AB = """
  1   A is closer to the audio      r   replay
  2   B is closer to the audio      b   back one row
  0   no difference / both wrong    s   skip, decide later
                                    q   save and quit
"""


HELP_KEY = """
  Who is the line about?
  n   no one — it, the situation      m   the speaker (I, we)
  y   the listener (you)              o   someone else, named or not
  x   can't tell / Japanese is wrong — leave the clip out
  r   replay   b   back one   s   skip for now   q   save and quit

  Then the English: a number takes that version as it is; anything else
  is taken as the English itself (paste a version and fix it to correct one).
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


def tally(rows: list[Row], labels=("J", "E", "OK")) -> dict[str, int]:
    out = {**dict.fromkeys(labels, 0), "": 0}
    for r in rows:
        out[r.verdict if r.verdict in out else ""] += 1
    return out


def report(rows: list[Row], labels=("J", "E", "OK")) -> None:
    """J vs E is the whole answer: whichever is larger is the weak half."""
    t = tally(rows, labels)
    done = len(rows) - t[""]
    print(f"\n  graded {done}/{len(rows)}   " + " · ".join(f"{k} {t[k]}" for k in labels))
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
    ap.add_argument("--rows", default="",
                    help="re-grade these rows even if answered, e.g. 15,28,3-9")
    ap.add_argument("--ab", action="store_true",
                    help="blind A/B sheet: which of two versions is closer")
    ap.add_argument("--key", action="store_true",
                    help="write the answer key: who each line is about, and the right English")
    ap.add_argument("--candidates", type=pathlib.Path, default=CANDIDATES,
                    help="JSON {arm: {wav: english}} offered as starting points in --key")
    ap.add_argument("--self-check", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    verdicts, help_, tags = ((VERDICTS_AB, HELP_AB, ("A", "B")) if args.ab
                             else (VERDICTS, HELP, ("JP", "EN")))
    labels = tuple(dict.fromkeys(verdicts.values()))
    keys = "/".join(verdicts) if args.ab else "j/e/ok"

    if args.self_check:
        return _selfcheck()

    if not args.file.exists():
        print(f"no such sheet: {args.file}", file=sys.stderr)
        return 1

    lines = args.file.read_text(encoding="utf-8").split("\n")
    rows = parse(lines)
    if not rows:
        print(f"no verdict table found in {args.file}", file=sys.stderr)
        return 1

    if args.tally:
        report(rows, labels)
        return 0

    wav_dir = args.wav_dir or args.file.parent / "wav"
    player = shutil.which("afplay") or shutil.which("aplay")
    if player is None:
        print("  no audio player found (afplay/aplay) — grading from text only\n")

    if args.key:
        return key_mode(args, rows, wav_dir, player)

    want = _spec(args.rows)
    todo = [i for i, r in enumerate(rows)
            if (r.num in want if want else not r.verdict)]
    if not todo:
        print("nothing to grade. Name rows to re-check with --rows 15,28"
              if not want else f"no such row(s): {args.rows}")
        report(rows, labels)
        return 0

    print(f"{len(todo)} row(s) to grade in {args.file.name}")
    print(help_)

    pos = 0
    while 0 <= pos < len(todo):
        r = rows[todo[pos]]
        wav = wav_dir / f"{r.wav}.wav"
        # In an A/B sheet the conf column carries the source line both sides
        # translate, since a verdict on two Englishes needs the Japanese.
        print(f"\n─── {r.num}/{len(rows)}  {r.wav}" + ("" if args.ab else f"  conf {r.conf}")
              + ("" if wav.exists() else "  [wav missing]"))
        if args.ab and r.conf not in ("", "—"):
            print(f"  JP  {r.conf}")
        print(f"  {tags[0]:2}  {r.jp}")
        print(f"  {tags[1]:2}  {r.en}")
        if r.verdict:
            print(f"  (currently {r.verdict})")
        if wav.exists():
            play(wav, player)

        while True:
            try:
                key = input(f"  [{keys}/r/b/s/q] ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print()
                save(args.file, lines, rows)
                report(rows, labels)
                return 0
            if key in verdicts:
                r.set(verdicts[key])
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
                report(rows, labels)
                return 0
            print(help_)

    save(args.file, lines, rows)
    report(rows, labels)
    return 0


def versions(cands: dict, wav: str) -> list[str]:
    """Every distinct English already produced for a clip, in an order that
    says nothing about which arm made it. Seeded by the clip so going back
    shows the same numbering."""
    out = list(dict.fromkeys(v[wav].strip() for v in cands.values() if (v.get(wav) or "").strip()))
    random.Random(wav).shuffle(out)
    return out


def key_mode(args, rows: list[Row], wav_dir: pathlib.Path, player: str | None) -> int:
    path = args.file.parent / "answer-key.json"
    key = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    cands = json.loads(args.candidates.read_text(encoding="utf-8")) if args.candidates.exists() else {}
    save_key = lambda: path.write_text(json.dumps(key, ensure_ascii=False, indent=1), encoding="utf-8")
    want = _spec(args.rows)
    todo = [r for r in rows if (r.num in want if want else r.wav not in key)]
    print(f"{len(todo)} clip(s) to key, {len(key)} already in {path.name}")
    print(HELP_KEY)
    pos = 0
    while 0 <= pos < len(todo):
        r = todo[pos]
        wav = wav_dir / f"{r.wav}.wav"
        opts = versions(cands, r.wav) or [r.en]
        print(f"\n─── {r.num}/{len(rows)}  {r.wav}" + ("" if wav.exists() else "  [wav missing]"))
        print(f"  JP  {r.jp}")
        for i, v in enumerate(opts, 1):
            print(f"  {i}   {v}")
        if r.wav in key:
            print(f"  (currently {key[r.wav]['subject']}: {key[r.wav].get('en', '')})")
        if wav.exists():
            play(wav, player)
        try:
            while True:
                k = input(f"  [{'/'.join(SUBJECTS)}/r/b/s/q] ").strip().lower()
                if k in SUBJECTS or k in ("b", "s", "q"):
                    break
                play(wav, player) if k == "r" and wav.exists() else print(HELP_KEY)
            if k == "q":
                break
            if k in ("b", "s"):
                pos = max(0, pos - 1) if k == "b" else pos + 1
                continue
            entry = {"jp": r.jp, "subject": SUBJECTS[k]}
            while k != "x":
                pick = input("  English (number, or type it): ").strip()
                if pick.isdigit() and 1 <= int(pick) <= len(opts):
                    pick = opts[int(pick) - 1]
                if pick:
                    entry["en"] = pick
                    break
        except (EOFError, KeyboardInterrupt):
            print()
            break
        key[r.wav] = entry
        save_key()
        pos += 1
    save_key()
    done = sum(r.wav in key for r in rows)
    print(f"\n  keyed {done}/{len(rows)} clips of {args.file.name} → {path}")
    return 0


def _selfcheck() -> int:
    assert _spec("") == set()
    assert _spec("15,28") == {"15", "28"}
    assert _spec(" 3 - 5 , 9 ") == {"3", "4", "5", "9"}
    assert _spec("7-7") == {"7"}
    # A full sheet: --rows reaches answered rows, the default reaches none.
    lines = [
        "| # | wav | JP | EN | conf | verdict |",
        "|--:|:--|:--|:--|--:|:--|",
        "| 1 | `utt_1` | あ | ah | — | OK |",
        "| 2 | `utt_2` | い | ee | — | J |",
        "| 3 | `utt_3` | う | oo | — |  |",
    ]
    rows = parse(lines)
    assert [r.num for r in rows] == ["1", "2", "3"], rows
    pick = lambda spec: [rows[i].num for i, r in enumerate(rows)
                         if (r.num in _spec(spec) if _spec(spec) else not r.verdict)]
    assert pick("") == ["3"], pick("")
    assert pick("1,2") == ["1", "2"], pick("1,2")
    assert tally(rows) == {"J": 1, "E": 0, "OK": 1, "": 1}, tally(rows)
    ab = parse(["| 1 | `utt_1` | あ | い | — | A |", "| 2 | `utt_2` | う | え | — | = |"])
    assert tally(ab, ("A", "B", "=")) == {"A": 1, "B": 0, "=": 1, "": 0}
    # Duplicates collapse, blanks drop, and the order is stable per clip.
    c = {"x": {"u1": "Hi.", "u2": ""}, "y": {"u1": " Hi. "}, "z": {"u1": "Hello."}}
    assert sorted(versions(c, "u1")) == ["Hello.", "Hi."] and versions(c, "u1") == versions(c, "u1")
    assert versions(c, "u2") == [] and versions(c, "u3") == []
    print("grade self-check OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
