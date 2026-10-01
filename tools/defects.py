"""Reference-free defect counting for a Japanese->English line.

Every rule here can be checked without knowing the correct translation, which is
the point: the 2026-08-22 Discord call has no reference transcript and never will.
The rules are deliberately conservative — each one should be something a reader
would call broken on sight, so a count of 0 means "nothing objectively wrong",
not "good".
"""
import re
import unicodedata

CJK = re.compile(r'[぀-ヿ㐀-䶿一-鿿]')
# A macron is essentially absent from English and ordinary in romanized Japanese,
# so it is a cheap, precise signal for the passthrough failure ("So so so so.").
# It is a lower bound: unmarked romaji ("desu ne") is invisible to it.
MACRON = re.compile(r'[āēīōūĀĒĪŌŪ]')
# Punctuation with no space after it and a letter following: "a moment.Yes".
GLUED = re.compile(r'[a-z][.,!?][A-Za-z]')
SENT = re.compile(r'(?<=[.!?…])[\s]+|\.{2,}\s*')

BLOWUP_SLACK = 40      # characters of headroom before any ratio applies
BLOWUP_RATIO = 4.0     # JP->EN runs ~2.5x in characters; 4x is a runaway
REPEAT_MIN_PARTS = 3
REPEAT_MAX_UNIQUE = 0.6


def _parts(en: str) -> list[str]:
    out = []
    for p in SENT.split(en.strip()):
        if p is None:
            continue
        p = unicodedata.normalize("NFKC", p).strip().lower().strip(".!?,… ")
        if p:
            out.append(p)
    return out


def defects(jp: str, en: str) -> list[str]:
    """Every objective defect in this translation, by name."""
    out = []
    e = (en or "").strip()
    if not e or e == jp.strip():
        return ["empty"]
    if CJK.search(e):
        out.append("japanese")
    if MACRON.search(e):
        out.append("romaji")
    if GLUED.search(e):
        out.append("glued")
    if len(e) > BLOWUP_RATIO * len(jp.strip()) + BLOWUP_SLACK:
        out.append("blowup")
    parts = _parts(e)
    looped = len(parts) >= REPEAT_MIN_PARTS and (
        len(set(parts)) / len(parts) < REPEAT_MAX_UNIQUE
        or max(parts.count(p) for p in set(parts)) >= 3)
    # Sentence-level repetition misses a stutter inside one sentence ("So so so
    # so."), which is the same defect at a smaller scale.
    words = [w.strip(".,!?…") for w in e.lower().split()]
    run = mx = 1
    for a, b in zip(words, words[1:]):
        run = run + 1 if a and a == b else 1
        mx = max(mx, run)
    if looped or mx >= 3:
        out.append("repetition")
    return out


def _selfcheck():
    """Synthetic lines only — no text from any recorded call or stream.

    Every case below is written to exercise one rule. Real transcript text is
    deliberately kept out of this file because tools/ is tracked and the
    recordings are not.
    """
    clean = [
        ("はい", "Yes."),
        ("そうだな", "That's right."),
        ("それは定期的にやる", "That's done periodically."),
        ("駅前の店は今日休みらしい", "The shop by the station seems to be closed today."),
    ]
    for jp, en in clean:
        assert defects(jp, en) == [], f"{en!r} -> {defects(jp, en)}"

    # The rule that misfires, documented rather than tuned away: a speaker really
    # saying a thing three times is indistinguishable from a generation loop at
    # this length, and Japanese conversation repeats constantly. Measured on 588
    # lines of one call, 8 of 11 flags were faithful translations of real
    # repetition. Treat a repetition count as an upper bound and read the lines.
    assert defects("痛い痛い痛い", "It hurts. It hurts. It hurts.") == ["repetition"]

    bad = {
        ("ほんとに", "Honto ni."): [],                      # unmarked romaji: invisible
        ("ほんとに", "Hontō ni."): ["romaji"],
        ("犬を連れて行く", "I'd take the 犬 with me"): ["japanese"],
        ("ちょっと待って、はい", "wait a moment.Yes"): ["glued"],
        ("なんだろう", "x" * 200): ["blowup"],
        ("あ", ""): ["empty"],
        ("うんうんうん", "Yeah yeah yeah yeah"): ["repetition"],
        ("そこは違うと思うんだけどね",
         "I don't think that's right. I mean... I mean... I mean..."):
            ["repetition"],
    }
    for (jp, en), want in bad.items():
        got = defects(jp, en)
        assert got == want, f"{en[:40]!r} -> {got}, expected {want}"
    print("defects self-check OK")


if __name__ == "__main__":
    _selfcheck()
