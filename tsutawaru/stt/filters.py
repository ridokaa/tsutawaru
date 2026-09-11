"""Whisper hallucination and repetition filtering (§6.1)."""
from __future__ import annotations


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


def is_hallucination(text: str, result=None, cfg=None, peak: float | None = None,
                     peak_ref: float | None = None) -> bool:
    """Filter one transcription.

    `peak` is max |sample| of the utterance and `peak_ref` the source's running
    median peak, when the caller has the audio. Both are needed for the
    amplitude test: the threshold only means anything relative to how loud this
    particular source runs. Passing neither leaves behaviour unchanged.
    """
    t = text.strip()
    if not t:
        return True
    if t in HALLUCINATIONS:
        return True
    if t in SHORT_AMBIGUOUS:
        if peak is not None and peak_ref:
            if peak < PEAK_FLOOR_RATIO * peak_ref:
                return True
        if result is None:
            return False
        return (
            getattr(result, "no_speech_prob", 0.0) > 0.5
            or getattr(result, "avg_logprob", 0.0) < -0.9
        )
    if len(set(t)) <= 2 and len(t) > 6:  # ああああああ / ーーーーー
        return True
    if result and cfg and getattr(result, "avg_logprob", 0.0) < cfg.logprob_floor:
        return True
    if result and getattr(result, "no_speech_prob", 0.0) > 0.8:
        return True
    for n in (3, 4, 5):  # n-gram repetition loops
        if len(t) >= n * 3 and t[:n] * 3 in t:
            return True
    return False
