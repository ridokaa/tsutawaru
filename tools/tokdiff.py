#!/usr/bin/env python
"""Compare two tokenizer backends over a fixed corpus.

Standalone on purpose. It imports tsutawaru but changes nothing in it, so it can
be deleted without leaving a trace — and so a comparison can never accidentally
become a feature.

The comparison runs the *whole* NLP stage (tokenize -> filter -> agglutinate ->
romaji), not just tokenization, because the breakdown is what a reader actually
sees. Two backends that segment identically but disagree about dictionary forms
produce different glosses, and that difference is the one that reaches the user.

Corpus, in order of preference:

    --jsonl PATH   a session recorded with `run.py --record`. Real speech, and
                   fixed forever once written, which is the only kind of sample
                   worth scoring a change against.
    --text PATH    one sentence per line.
    (default)      Japanese string literals harvested from this repository —
                   enough to smoke-test the tooling, not enough to conclude
                   anything from.

Usage:
    python tools/tokdiff.py --jsonl logs/call.jsonl
    python tools/tokdiff.py --a janome --b unidic --show 40
    python tools/tokdiff.py --jsonl logs/call.jsonl --html /tmp/tokdiff.html
"""
from __future__ import annotations

import argparse
import html as html_mod
import json
import pathlib
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from tsutawaru.config import NlpCfg          # noqa: E402
from tsutawaru.models import Segment          # noqa: E402
from tsutawaru.nlp.pipeline import annotate   # noqa: E402
from tsutawaru.nlp.tokenizer import build_tokenizer  # noqa: E402

JP = re.compile(r"[぀-ヿ一-鿿]")


@dataclass
class Row:
    text: str
    a: list          # tokens from backend A
    b: list          # tokens from backend B

    @property
    def seg_differs(self) -> bool:
        return [t.surface for t in self.a] != [t.surface for t in self.b]

    @property
    def base_differs(self) -> bool:
        if self.seg_differs:
            return False  # not comparable; counted under segmentation
        return [t.base_form for t in self.a] != [t.base_form for t in self.b]

    @property
    def reading_differs(self) -> bool:
        if self.seg_differs:
            return False
        return [t.romaji for t in self.a] != [t.romaji for t in self.b]

    @property
    def identical(self) -> bool:
        return not (self.seg_differs or self.base_differs or self.reading_differs)


def corpus_from_jsonl(path: pathlib.Path) -> list[str]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        # Segment.to_dict projects `original` as "jp" — the recorder's schema is
        # the wire format, not the model's attribute names.
        text = rec.get("jp") or rec.get("original") or ""
        if text and JP.search(text):
            out.append(text)
    return out


def corpus_from_repo() -> list[str]:
    """Japanese string literals from the source tree.

    Parsed rather than regexed: a regex over quotes swallows whole docstrings —
    including this one — and then the "corpus" is mostly Python keywords with a
    few kana in it. ast sees actual literals, and the filters below drop the
    prose ones (multi-line, long, or mostly non-Japanese).
    """
    import ast

    root = pathlib.Path(__file__).resolve().parent.parent
    found = set()
    for p in list((root / "tests").rglob("*.py")) + list((root / "tsutawaru").rglob("*.py")):
        try:
            tree = ast.parse(p.read_text(encoding="utf-8"))
        except (OSError, SyntaxError):
            continue
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
                continue
            s = node.value.strip()
            if not s or "\n" in s or len(s) > 60:
                continue
            jp = len(JP.findall(s))
            if jp >= 2 and jp / len(s) > 0.5:
                found.add(s)
    return sorted(found)


def run(backend: str, texts: list[str]) -> tuple[list[list], float]:
    cfg = NlpCfg(tokenizer=backend)
    tok = build_tokenizer(cfg)
    tok.tokenize("ウォームアップ")
    t0 = time.perf_counter()
    rows = [annotate(Segment.new(stream="cmp", original=t), tok, cfg).tokens for t in texts]
    return rows, time.perf_counter() - t0


def fmt(tokens) -> str:
    return " / ".join(f"{t.surface}={t.base_form}" for t in tokens) or "—"


def write_html(path: pathlib.Path, rows: list[Row], a: str, b: str) -> None:
    diffs = [r for r in rows if not r.identical]
    parts = [
        "<title>tokdiff</title>",
        "<style>",
        ":root{color-scheme:light dark}",
        "body{font:14px/1.6 system-ui,sans-serif;margin:24px;max-width:1200px}",
        "table{border-collapse:collapse;width:100%}",
        "td,th{border:1px solid #8884;padding:6px 8px;vertical-align:top;text-align:left}",
        "th{background:#8881}", "td.jp{font-size:16px;white-space:nowrap}",
        ".tag{font-size:11px;padding:1px 6px;border-radius:3px;background:#8882}",
        "</style>",
        f"<h1>tokdiff — {html_mod.escape(a)} vs {html_mod.escape(b)}</h1>",
        f"<p>{len(diffs)} of {len(rows)} sentences differ.</p>",
        f"<table><tr><th>sentence</th><th>{html_mod.escape(a)}</th>"
        f"<th>{html_mod.escape(b)}</th><th>what</th></tr>",
    ]
    for r in diffs:
        what = ("segmentation" if r.seg_differs
                else "base form" if r.base_differs else "reading")
        parts.append(
            f"<tr><td class='jp'>{html_mod.escape(r.text)}</td>"
            f"<td>{html_mod.escape(fmt(r.a))}</td>"
            f"<td>{html_mod.escape(fmt(r.b))}</td>"
            f"<td><span class='tag'>{what}</span></td></tr>"
        )
    parts.append("</table>")
    path.write_text("\n".join(parts), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", default="janome", help="control backend")
    ap.add_argument("--b", default="unidic", help="candidate backend")
    ap.add_argument("--jsonl", type=pathlib.Path, help="session from run.py --record")
    ap.add_argument("--text", type=pathlib.Path, help="one sentence per line")
    ap.add_argument("--show", type=int, default=25, help="example diffs to print")
    ap.add_argument("--html", type=pathlib.Path, help="write a side-by-side page")
    args = ap.parse_args()

    if args.jsonl:
        texts, src = corpus_from_jsonl(args.jsonl), str(args.jsonl)
    elif args.text:
        texts = [ln.strip() for ln in args.text.read_text(encoding="utf-8").splitlines()
                 if ln.strip() and JP.search(ln)]
        src = str(args.text)
    else:
        texts, src = corpus_from_repo(), "repo string literals (smoke test only)"

    if not texts:
        print(f"no Japanese sentences found in {src}", file=sys.stderr)
        return 2

    a_rows, a_ms = run(args.a, texts)
    b_rows, b_ms = run(args.b, texts)
    rows = [Row(t, x, y) for t, x, y in zip(texts, a_rows, b_rows)]

    n = len(rows)
    counts = Counter(
        "segmentation" if r.seg_differs else
        "base form" if r.base_differs else
        "reading" if r.reading_differs else "identical"
        for r in rows
    )

    print(f"corpus: {src}")
    print(f"        {n} sentences\n")
    print(f"{'':14s} {args.a:>12s} {args.b:>12s}")
    print(f"{'tokens':14s} {sum(len(r.a) for r in rows):12d} {sum(len(r.b) for r in rows):12d}")
    print(f"{'ms/sentence':14s} {a_ms/n*1000:12.3f} {b_ms/n*1000:12.3f}\n")
    for k in ("identical", "segmentation", "base form", "reading"):
        c = counts.get(k, 0)
        print(f"  {k:14s} {c:5d}  {c/n*100:5.1f}%")

    shown = [r for r in rows if not r.identical][:args.show]
    if shown:
        print(f"\nfirst {len(shown)} differences:\n")
        for r in shown:
            what = ("seg " if r.seg_differs else "base" if r.base_differs else "read")
            print(f"  [{what}] {r.text}")
            print(f"         {args.a:>7s}: {fmt(r.a)}")
            print(f"         {args.b:>7s}: {fmt(r.b)}")

    if args.html:
        write_html(args.html, rows, args.a, args.b)
        print(f"\nwrote {args.html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
