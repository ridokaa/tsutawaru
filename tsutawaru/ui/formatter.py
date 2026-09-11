"""Multi-tier block renderer matching the exact AutoTranslationPlan format."""
from __future__ import annotations

from tsutawaru.config import UiCfg
from tsutawaru.models import Segment

SEP = "-" * 50


def render_block(seg: Segment, cfg: UiCfg) -> str:
    """Render a multi-tier transcript block.

    Example:
    [Discord]:
    1. Original (JP)  : おはようございます、いい天気ですね。
    2. Romaji         : ohayougozaimasu ii tenki desune
    3. English (Full) : Good morning, nice weather today
    4. Word Breakdown :
       • おはようございます (ohayougozaimasu) : good morning
       • いい (ii) : good
       • 天気 (tenki) : weather
       • ですね (desune) : right / isn't it
    --------------------------------------------------
    """
    lines = [
        f"[{seg.stream}]:",
        f"1. Original (JP)  : {seg.original}",
    ]
    if cfg.show_romaji:
        lines.append(f"2. Romaji         : {seg.romaji or '…'}")
    lines.append(f"3. English (Full) : {seg.english or '…'}")
    if seg.alt_model:
        # Indented, never renumbered into the tiers: those are the product,
        # this is a margin note. Stacked rather than columnar because Japanese
        # is double-width and CJK column padding breaks on the first wrap.
        lines.append(f"   --- {seg.alt_model} ---")
        lines.append(f"   Original (JP)  : {seg.alt_original or '…'}")
        if cfg.show_romaji:
            lines.append(f"   Romaji         : {seg.alt_romaji or '…'}")
        lines.append(f"   English (Full) : {seg.alt_english or '…'}")
    if cfg.show_breakdown:
        lines.append("4. Word Breakdown :")
        for t in seg.tokens:
            gloss = t.gloss if t.gloss else ("…" if t.gloss is None else "?")
            # Glosses come from base_form, so conjugation is invisible without
            # this: 行けなかった would read "i can go".
            infl = f" [{', '.join(t.infl)}]" if t.infl else ""
            lines.append(f"   • {t.surface} ({t.romaji}) : {gloss}{infl}")
        if getattr(seg, "truncated", 0):
            lines.append(f"   … (+{seg.truncated} tokens)")
    lines.append(SEP)
    return "\n".join(lines)
