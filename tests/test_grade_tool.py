"""tools/grade.py writes into the project's only graded corpus.

The sheet it edits cannot be regenerated — the WAVs are a one-off dump of a real
call, and the sampled rows carry hand-written reference judgements. So the bar
for this tool is not that it grades correctly but that it never damages anything
it did not mean to change, including across repeated saves.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

_SPEC = importlib.util.spec_from_file_location(
    "grade", pathlib.Path(__file__).resolve().parents[1] / "tools" / "grade.py"
)
grade = importlib.util.module_from_spec(_SPEC)
# @dataclass resolves annotations through sys.modules[cls.__module__], so the
# module has to be registered before it executes, not after.
sys.modules[_SPEC.name] = grade
_SPEC.loader.exec_module(grade)


SHEET = """# accuracy check

Prose above the table, including a fenced block that looks tabular:

```
 1. すごいねアボルスコ
```

| # | wav | JP (tool heard) | EN (tool produced) | conf | verdict |
|--:|:--|:--|:--|--:|:--|
| 1 | `utt_00046` | すごいねアボルスコ | That's amazing Aborsko | -0.45 |  |
| 2 | `utt_00120` | コーラル | coral | -0.23 | J |

Trailing prose.
"""


def _load(tmp_path):
    p = tmp_path / "sheet.md"
    p.write_text(SHEET, encoding="utf-8")
    lines = p.read_text(encoding="utf-8").split("\n")
    return p, lines, grade.parse(lines)


def test_parse_finds_only_the_verdict_table(tmp_path):
    """The header, alignment row and the numbered request block are not data."""
    _, _, rows = _load(tmp_path)
    assert [r.num for r in rows] == ["1", "2"]
    assert [r.wav for r in rows] == ["utt_00046", "utt_00120"]
    assert rows[0].jp == "すごいねアボルスコ"
    assert rows[1].verdict == "J"


def test_saving_without_grading_changes_nothing(tmp_path):
    """The riskiest case: opening the sheet must not rewrite it."""
    p, lines, rows = _load(tmp_path)
    grade.save(p, lines, rows)
    assert p.read_text(encoding="utf-8") == SHEET


def test_repeated_saves_do_not_grow_the_file(tmp_path):
    """Regression: rejoining split lines used to append a blank line each time."""
    p, lines, rows = _load(tmp_path)
    for _ in range(5):
        grade.save(p, lines, rows)
    assert p.read_text(encoding="utf-8") == SHEET


def test_only_the_verdict_cell_is_touched(tmp_path):
    p, lines, rows = _load(tmp_path)
    rows[0].set("OK")
    grade.save(p, lines, rows)

    before, after = SHEET.split("\n"), p.read_text(encoding="utf-8").split("\n")
    changed = [i for i, (a, b) in enumerate(zip(before, after)) if a != b]
    assert len(changed) == 1
    assert after[changed[0]] == (
        "| 1 | `utt_00046` | すごいねアボルスコ | That's amazing Aborsko | -0.45 | OK |"
    )


def test_tally_counts_ungraded_rows_separately(tmp_path):
    _, _, rows = _load(tmp_path)
    assert grade.tally(rows) == {"J": 1, "E": 0, "OK": 0, "": 1}
