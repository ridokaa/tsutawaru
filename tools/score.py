#!/usr/bin/env python
"""Score translations against the answer key — no listening needed.

The key (`grade.py --key`) holds, per clip, the English a listener approved and
who the line is about. Any translator's output can then be scored without a new
grading sheet:

    added person   the output says I/we or you where the approved English
                   does not — the invented subject the translator is known for
    chrF           character n-gram overlap with the approved English, 0–100;
                   catches added or lost content, blind to the subject

Usage:
    python tools/score.py experiments/reports/frag-eval-20261007.json
    python tools/score.py OUTPUTS.json --agree experiments/sessions/20260822-live/accuracy-check.md

OUTPUTS.json is {arm: {wav: english}}. `--agree` scores a graded sheet's EN
column and shows whether the scores separate its ear verdicts (E vs OK). Do that
before trusting a score; a scorer that cannot tell E from OK grades nothing.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import statistics
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parent.parent
KEY = ROOT / "experiments" / "sessions" / "20260822-live" / "answer-key.json"

PERSON = {
    "I/we": {"i", "i'm", "i'll", "i've", "i'd", "me", "my", "mine", "myself",
             "we", "we're", "we'll", "we've", "we'd", "us", "our", "ours", "ourselves"},
    "you": {"you", "you're", "you'll", "you've", "you'd", "your", "yours", "yourself"},
}


def persons(en: str) -> set[str]:
    words = set(re.findall(r"[a-z']+", en.lower().replace("’", "'")))
    return {p for p, ws in PERSON.items() if words & ws}


def added_person(out: str, ref: str) -> set[str]:
    return persons(out) - persons(ref)


def chrf(hyp: str, ref: str, n: int = 6, beta: float = 2.0) -> float:
    """chrF (Popović 2015): character n-gram F-score, whitespace removed."""
    h, r = re.sub(r"\s+", "", hyp), re.sub(r"\s+", "", ref)
    ps, rs = [], []
    for k in range(1, n + 1):
        hc = Counter(h[i:i + k] for i in range(len(h) - k + 1))
        rc = Counter(r[i:i + k] for i in range(len(r) - k + 1))
        if not hc or not rc:
            continue
        hit = sum((hc & rc).values())
        ps.append(hit / sum(hc.values()))
        rs.append(hit / sum(rc.values()))
    if not ps:
        return 100.0 if h == r else 0.0
    p, rr = statistics.mean(ps), statistics.mean(rs)
    return 0.0 if p + rr == 0 else 100 * (1 + beta**2) * p * rr / (beta**2 * p + rr)


def scored(key: dict) -> dict:
    return {w: k for w, k in key.items() if k.get("subject") != "skip" and k.get("en")}


def score(outputs: dict, key: dict) -> None:
    key = scored(key)
    print(f"{len(key)} keyed clips\n")
    print(f"  {'arm':8} {'clips':>5} {'added person':>13} {'chrF':>6}")
    for arm, res in outputs.items():
        ws = [w for w in key if w in res]
        if not ws:
            continue
        add = sum(bool(added_person(res[w], key[w]["en"])) for w in ws)
        c = statistics.mean(chrf(res[w], key[w]["en"]) for w in ws)
        print(f"  {arm:8} {len(ws):5} {add:13} {c:6.1f}")


def agree(sheet: pathlib.Path, key: dict) -> None:
    """Do the scores separate the ear's E rows from its OK rows?"""
    key = scored(key)
    rows = []
    for line in sheet.read_text(encoding="utf-8").split("\n"):
        c = [x.strip() for x in line.split("|")[1:-1]]
        if len(c) == 6 and c[0].isdigit() and c[5] in ("E", "OK") and c[1].strip("`") in key:
            ref = key[c[1].strip("`")]["en"]
            rows.append((c[5], chrf(c[3], ref), bool(added_person(c[3], ref))))
    e = [r for r in rows if r[0] == "E"]
    ok = [r for r in rows if r[0] == "OK"]
    if not e or not ok:
        print(f"need both E and OK rows in the key; have E {len(e)}, OK {len(ok)}")
        return
    # Chance that a random OK row outscores a random E row (0.5 = no signal).
    auc = statistics.mean((o[1] > x[1]) + 0.5 * (o[1] == x[1]) for o in ok for x in e)
    print(f"ear E {len(e)}: chrF {statistics.mean(r[1] for r in e):.1f}, added person {sum(r[2] for r in e)}")
    print(f"ear OK {len(ok)}: chrF {statistics.mean(r[1] for r in ok):.1f}, added person {sum(r[2] for r in ok)}")
    print(f"an OK row outscores an E row {auc:.0%} of the time (50% = chrF sees nothing)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("outputs", type=pathlib.Path, nargs="?", help="JSON {arm: {wav: english}}")
    ap.add_argument("--key", type=pathlib.Path, default=KEY)
    ap.add_argument("--agree", type=pathlib.Path, help="graded sheet whose EN column to check against")
    ap.add_argument("--self-check", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.self_check:
        return _selfcheck()
    if not a.key.exists():
        print(f"no answer key yet: {a.key}\nwrite it with: python tools/grade.py --key")
        return 1
    key = json.loads(a.key.read_text(encoding="utf-8"))
    if a.outputs:
        score(json.loads(a.outputs.read_text(encoding="utf-8")), key)
    if a.agree:
        agree(a.agree, key)
    return 0


def _selfcheck() -> int:
    # The approved English decides which person is "added", not the word alone.
    assert added_person("I wonder if I'll like the cake", "I wonder if the cake is good") == set()
    assert added_person("I wonder if I'll like the cake", "I wonder how the cake turned out") == set()
    assert added_person("I think it's late", "It's late, isn't it") == {"I/we"}
    assert added_person("You know, it's late", "It's late") == {"you"}
    assert added_person("We’re done", "Done") == {"I/we"}  # curly apostrophe
    assert persons("Mine is the iPhone") == {"I/we"} and persons("The Ice is cold") == set()
    assert chrf("It's late.", "It's late.") == 100.0
    assert chrf("abc", "xyz") == 0.0
    assert chrf("It's late, isn't it", "It's late") > chrf("I like trains", "It's late")
    print("score self-check OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
