"""Whisper hallucination and repetition filtering (§6.1)."""
from __future__ import annotations

import collections


HALLUCINATIONS = {
    "ご視聴ありがとうございました",
    "ご視聴ありがとうございました。",
    "チャンネル登録お願いします",
    "最後までご視聴いただきありがとうございます",
    "字幕視聴ありがとうございました",
    "エンディング",
    "おわり",
}

# Short affirmatives are REAL speech most of the time. Only drop them when the
# acoustics also say "no speech" — never on the string alone.
#
# ごめん and bare ありがとうございました were added after the 2026-08-22 session,
# where they were confirmed by ear to be output over laughter and near-silence:
# ごめん x30 and ありがとうございました x3 in 475 lines, none of them spoken.
# ありがとうございました is the tail of ご視聴ありがとうございました above — the same
# YouTube-caption artifact, arriving without its prefix.
SHORT_AMBIGUOUS = {
    "はい", "はい。", "うん", "ええ", "。", "…", "ん",
    "ごめん", "ごめん。", "ありがとうございました", "ありがとうございました。",
}

# A SHORT_AMBIGUOUS string is not believed when its waveform peak falls this far
# below the source's own typical peak.
#
# Whisper's own scores cannot make this call: across the 2026-08-22 session the
# confirmed-junk clips ran avg_logprob -0.87..-0.35 and no_speech <= 0.42, so
# neither existing threshold (-0.9 / 0.5) ever fired — and the one *genuine* ん
# scored -0.744, worse than most of the junk.
#
# The ratio is deliberately relative. A first version used an absolute 0.12,
# tuned on that Discord call (peaks median 0.73), and it did not survive contact
# with a second source: a YouTube stream measured the next morning ran a median
# peak of 0.20, where an absolute floor suppresses ordinary speech wholesale.
# Against each source's own median the same ratio does the right thing in both:
#   Discord  median 0.73 -> floor 0.117: 28/44 junk removed, 1 real line lost
#   YouTube  median 0.20 -> floor 0.031: nothing suppressed, which is correct —
#            on that stream the hallucinations peaked 0.134-0.183 while real
#            speech ran a median of 0.104, so amplitude carries no signal at all
#            and the gate must abstain rather than guess.
PEAK_FLOOR_RATIO = 0.16
# Below this many observed utterances the running estimate is noise; abstain.
PEAK_REF_MIN_SAMPLES = 20


#: Why utterances were discarded this run, by reason. Not locked: the pipeline
#: runs exactly one `_stt_worker` and it is the only writer, the same
#: single-consumer assumption `QwenMLXEngine._ensure_session` relies on. A lock
#: here would imply a second caller is supported when it is not.
DROPS: collections.Counter = collections.Counter()


def drops() -> dict[str, int]:
    """A snapshot of `DROPS` for display. Empty when nothing was discarded."""
    return dict(DROPS)


def drop_reason(text: str, result=None, cfg=None, peak: float | None = None,
                peak_ref: float | None = None) -> str | None:
    """Why this transcription should be discarded, or None to keep it.

    `peak` is max |sample| of the utterance and `peak_ref` the source's running
    median peak, when the caller has the audio. Both are needed for the
    amplitude test: the threshold only means anything relative to how loud this
    particular source runs. Passing neither leaves behaviour unchanged.

    Returns a reason rather than a bool because a fifth of captured speech was
    being discarded with no record of which rule did it — see
    `pipeline/orchestrator.py:_emit_segment`, which counts what comes back here.
    """
    t = text.strip()
    if not t:
        return "empty"
    if t in HALLUCINATIONS:
        return "wordlist"
    if t in SHORT_AMBIGUOUS:
        if peak is not None and peak_ref:
            if peak < PEAK_FLOOR_RATIO * peak_ref:
                return "peak-gate"
        if result is None:
            return None
        if getattr(result, "no_speech_prob", 0.0) > 0.5:
            return "no-speech"
        if getattr(result, "avg_logprob", 0.0) < -0.9:
            return "low-confidence"
        return None
    if len(set(t)) <= 2 and len(t) > 6:  # ああああああ / ーーーーー
        return "single-char"
    if result and cfg and getattr(result, "avg_logprob", 0.0) < cfg.logprob_floor:
        return "low-confidence"
    if result and getattr(result, "no_speech_prob", 0.0) > 0.8:
        return "no-speech"
    for n in (3, 4, 5):  # n-gram repetition loops
        if len(t) >= n * 3 and t[:n] * 3 in t:
            return "repetition"
    return None


def is_hallucination(text: str, result=None, cfg=None, peak: float | None = None,
                     peak_ref: float | None = None) -> bool:
    """Whether to discard. `drop_reason` is the same test with the reason kept."""
    return drop_reason(text, result, cfg, peak, peak_ref) is not None


if __name__ == "__main__":  # self-check: python -m tsutawaru.stt.filters
    # The regression this guards: a reason string that no longer matches the
    # branch it names, which would make the drop histogram confidently wrong.
    class _R:
        def __init__(self, lp=0.0, ns=0.0):
            self.avg_logprob, self.no_speech_prob = lp, ns

    class _C:
        logprob_floor = -1.0

    cases = [
        ("", None, None, {}, "empty"),
        ("   ", None, None, {}, "empty"),
        ("ご視聴ありがとうございました", None, None, {}, "wordlist"),
        ("はい", _R(), None, {"peak": 0.01, "peak_ref": 0.73}, "peak-gate"),
        ("はい", _R(ns=0.9), None, {}, "no-speech"),
        ("はい", _R(lp=-1.5), None, {}, "low-confidence"),
        ("ああああああああ", None, None, {}, "single-char"),
        # Two distinct characters is still "single-char", and that branch runs
        # first — そうそう… is caught there, not by the n-gram test below.
        ("そうそうそうそうそう", None, None, {}, "single-char"),
        ("こんにちはこんにちはこんにちは", None, None, {}, "repetition"),
        ("結構早いな", _R(), _C(), {}, None),
        ("はい", None, None, {}, None),                       # no evidence, no opinion
        ("はい", _R(), None, {"peak": 0.5, "peak_ref": 0.73}, None),
    ]
    for text, res, cfg, kw, want in cases:
        got = drop_reason(text, res, cfg, **kw)
        assert got == want, f"{text!r} -> {got!r}, expected {want!r}"
        # The wrapper the existing tests assert on must agree, always.
        assert is_hallucination(text, res, cfg, **kw) is (want is not None), text

    DROPS.clear()
    for text, res, cfg, kw, _ in cases:
        r = drop_reason(text, res, cfg, **kw)
        if r:
            DROPS[r] += 1
    assert drops() == {"empty": 2, "wordlist": 1, "peak-gate": 1, "no-speech": 1,
                       "low-confidence": 1, "single-char": 2, "repetition": 1}, drops()
    DROPS.clear()

    print("filters self-check OK")
