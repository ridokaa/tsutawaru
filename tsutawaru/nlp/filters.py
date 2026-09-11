"""Punctuation, POS filtering, and stutter deduplication for breakdown tokens."""
from __future__ import annotations

import re
from tsutawaru.models import Token

DROP_POS = {"記号", "補助記号", "空白", "フィラー", "その他"}
# ー (chōonpu) is deliberately NOT in this class: it must survive inside コーヒー.
PUNCT_RE = re.compile(r'''^[\s、。，．・「」『』（）()\[\]【】！？!?…~〜:;：；"'`]+$''')


def keep(tok: Token) -> bool:
    if not tok.surface.strip():
        return False
    if PUNCT_RE.match(tok.surface):
        return False
    if tok.pos in DROP_POS:
        return False
    return True


def dedupe(tokens: list[Token]) -> list[Token]:
    """Collapse immediate repeats (stutters: 'あ、あ、あの') for a cleaner breakdown."""
    out: list[Token] = []
    prev: str | None = None
    for t in tokens:
        if prev is not None and t.surface == prev:
            continue
        out.append(t)
        prev = t.surface
    return out
