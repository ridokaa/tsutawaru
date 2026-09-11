"""Tests for Whisper hallucination filtering (§6.1)."""
from __future__ import annotations

from tsutawaru.config import SttCfg
from tsutawaru.stt.base import STTResult
from tsutawaru.stt.filters import is_hallucination


def test_hallucination_boilerplate_detected():
    assert is_hallucination("ご視聴ありがとうございました") is True
    assert is_hallucination("ご視聴ありがとうございました。") is True
    assert is_hallucination("チャンネル登録お願いします") is True
    assert is_hallucination("おわり") is True


def test_repeated_characters_and_ngrams():
    # Long character repetition
    assert is_hallucination("ああああああああ") is True
    assert is_hallucination("ーーーーーーーー") is True
    # 3-gram repeating 3+ times
    assert is_hallucination("abcabcabc") is True
    assert is_hallucination("こんにちはこんにちはこんにちは") is True


def test_confident_short_affirmatives_kept():
    cfg = SttCfg()
    # Confident "はい"
    res_confident = STTResult(
        text="はい",
        language="ja",
        lang_prob=0.99,
        avg_logprob=-0.3,
        no_speech_prob=0.05,
    )
    assert is_hallucination("はい", res_confident, cfg) is False

    # Confident "うん"
    res_un = STTResult(
        text="うん",
        language="ja",
        lang_prob=0.95,
        avg_logprob=-0.2,
        no_speech_prob=0.1,
    )
    assert is_hallucination("うん", res_un, cfg) is False


def test_unconfident_short_affirmatives_filtered():
    cfg = SttCfg()
    # Unconfident "はい" with high no_speech_prob
    res_noise = STTResult(
        text="はい",
        language="ja",
        lang_prob=0.5,
        avg_logprob=-0.5,
        no_speech_prob=0.7,
    )
    assert is_hallucination("はい", res_noise, cfg) is True

    # Unconfident "はい" with very low avg_logprob
    res_low_prob = STTResult(
        text="はい",
        language="ja",
        lang_prob=0.5,
        avg_logprob=-1.5,
        no_speech_prob=0.2,
    )
    assert is_hallucination("はい", res_low_prob, cfg) is True


def test_valid_speech_passes():
    cfg = SttCfg()
    res = STTResult(
        text="おはようございます、いい天気ですね。",
        language="ja",
        lang_prob=0.99,
        avg_logprob=-0.2,
        no_speech_prob=0.01,
    )
    assert is_hallucination(res.text, res, cfg) is False


# ------------------------------------------------- peak-amplitude gate (§6.1)
# Added after the 2026-08-22 live session. Ground truth is by ear: every clip
# referenced below was listened to before these numbers were written down.

def test_peak_gate_drops_stock_phrases_over_near_silence():
    """ごめん x30 and ありがとうございました x3, none of them ever spoken."""
    cfg = SttCfg()
    # Representative of the confirmed-junk clips: Whisper was *not* unconfident
    # about them, which is why the existing thresholds never fired.
    res = STTResult(text="ごめん", language="ja", lang_prob=0.9,
                    avg_logprob=-0.53, no_speech_prob=0.16)
    assert is_hallucination("ごめん", res, cfg) is False, "no audio -> no opinion"
    # Discord call: source median peak 0.73 -> floor 0.117
    assert is_hallucination("ごめん", res, cfg, peak=0.04, peak_ref=0.73) is True
    assert is_hallucination("ありがとうございました", res, cfg,
                            peak=0.09, peak_ref=0.73) is True


def test_peak_gate_keeps_a_genuinely_quiet_word():
    """utt_00574: a real ん, avg_logprob -0.744 — worse than most of the junk.

    Any confidence-based rule deletes this line and keeps the hallucinations.
    Peak amplitude is what separates them, so it must be what decides.
    """
    cfg = SttCfg()
    res = STTResult(text="ん", language="ja", lang_prob=0.8,
                    avg_logprob=-0.744, no_speech_prob=0.2)
    assert is_hallucination("ん", res, cfg, peak=0.41, peak_ref=0.73) is False


def test_peak_gate_is_conservative_about_real_speech():
    """The floor sits under the 10th percentile of real speech (peak 0.234)."""
    from tsutawaru.stt.filters import PEAK_FLOOR_RATIO

    assert PEAK_FLOOR_RATIO * 0.73 < 0.234


def test_peak_gate_adapts_to_a_quieter_source():
    """A YouTube stream ran 4x quieter; an absolute floor ate its real speech.

    Measured 2026-08-23: stream median peak 0.196, real lines median 0.104.
    Under the old absolute 0.12 those ordinary lines were suppressed; scaled to
    the source they must survive.
    """
    cfg = SttCfg()
    res = STTResult(text="うん", language="ja", lang_prob=0.9,
                    avg_logprob=-0.35, no_speech_prob=0.07)
    assert is_hallucination("うん", res, cfg, peak=0.104, peak_ref=0.196) is False
    # …while the same ratio still fires on that source's own near-silence
    assert is_hallucination("うん", res, cfg, peak=0.01, peak_ref=0.196) is True


def test_peak_gate_abstains_without_a_reference():
    """Before enough utterances are seen, amplitude gets no vote."""
    cfg = SttCfg()
    res = STTResult(text="はい", language="ja", lang_prob=0.9,
                    avg_logprob=-0.3, no_speech_prob=0.05)
    assert is_hallucination("はい", res, cfg, peak=0.001, peak_ref=None) is False


def test_peak_gate_only_applies_to_ambiguous_strings():
    """A quiet but real sentence is never dropped for being quiet."""
    cfg = SttCfg()
    res = STTResult(text="全体周知としてはいいと思う", language="ja", lang_prob=0.95,
                    avg_logprob=-0.4, no_speech_prob=0.1)
    assert is_hallucination("全体周知としてはいいと思う", res, cfg,
                            peak=0.01, peak_ref=0.73) is False
