"""Qwen3-ASR on Apple Silicon via MLX — an opt-in alternative to Whisper.

`stt.model` defaults to "qwen3", so this file is what runs on an untouched
config. Set it to kotoba/medium/small to go back to Whisper.

Why it exists
-------------
kotoba-whisper is distilled on ReazonSpeech — Japanese TV broadcast, which is
read speech. It wins on read-speech benchmarks and loses badly on spontaneous
conversation, which is the only kind this project ever sees. Two independent
2026 conversational-Japanese benchmarks put kotoba-whisper-v2.0 last (WER 0.534,
CER 0.495) and Qwen3-ASR-1.7B first (0.185 / 0.140), and Qwen wins the
code-switching set by nearly 2x — relevant here because game chat is full of
bare English nouns.

That ordering is *someone else's* measurement on someone else's audio. It is a
reason to make this switchable, not a reason to believe it about this Discord
call. accuracy-check.md and tools/grade.py exist to settle that on our own
sample; until they have, both models are hypotheses.

What it costs
-------------
Qwen3-ASR is a decoder-only ASR LLM, not a Whisper variant. It reports no
`no_speech_prob` and has no `no_speech_threshold` of its own, so the no-speech
branches of stt/filters.py abstain under it. The peak-amplitude gate, computed
from the waveform in the orchestrator, keeps working.

The library reports no `avg_logprob` either, but it scores every token before
picking one, so `_hook_logprobs` keeps that score and the engine reports the
mean as `avg_logprob`. Measured on the 600 clips of the 2026-08-22 call: the
lowest-scored lines are the misheard ones (an OK/E sheet row outscores a J row
83% of the time; 78–87% within each line-length band), and at the shipped
floors 6 lines (1%) are dropped and 44 (7%) dimmed, against a kotoba-whisper
disagreement rate of 83% and 86% there versus 36% overall.
"""
from __future__ import annotations

import logging
import re

import numpy as np

from tsutawaru.config import SttCfg
from tsutawaru.stt.base import STTEngine, STTResult

log = logging.getLogger(__name__)

# MLX conversions of the official Qwen weights. 8-bit rather than bf16 for both:
# the 1.7B bf16 build is 3.6 GB against 2.47 GB here, and on a 16 GB fanless Air
# the memory that buys is worth more than the sub-1% quantisation cost.
REPO = {
    # The recommendation. 2.47 GB.
    "qwen3": "mlx-community/Qwen3-ASR-1.7B-8bit",
    # 1.01 GB, ~3x fewer parameters. Here as the latency escape hatch: if the
    # 1.7B blows the end-of-speech budget on this machine, this is the fallback
    # to try before giving up on Qwen entirely.
    "qwen3-small": "mlx-community/Qwen3-ASR-0.6B-8bit",
}

# What an STTResult carries when the engine cannot report it. Both values are
# the sentinels models.Segment already documents for "not reported", and both
# are chosen so every confidence test in stt/filters.py abstains rather than
# guesses: 0.0 is above any logprob floor, -1.0 is below any no-speech ceiling.
# test_qwen_engine.py pins that, because a sentinel that happened to trip a
# threshold would silently discard real speech.
NO_LOGPROB = 0.0
NO_SPEECH_PROB = -1.0

# Log-probability of each token the library picks, for the utterance in flight.
# ponytail: one module-level list, safe because the pipeline runs exactly one
# STT worker (see _ensure_session); a second engine in one process would need
# its own.
_LOGPROBS: list[float] = []


def _hook_logprobs() -> bool:
    """Wrap mlx_qwen3_asr's token picker so each pick's log-probability is kept.

    The library computes the full distribution for every token and keeps only
    the argmax; there is no API for the score. `generate._sample` is private,
    which is why requirements-mac.txt pins the exact version. If it moves, the
    engine falls back to NO_LOGPROB and says so, rather than failing the call.
    """
    try:
        import mlx.core as mx
        import mlx_qwen3_asr.generate as gen

        pick = gen._sample
    except (ImportError, AttributeError):
        return False
    if getattr(pick, "keeps_logprob", False):
        return True

    def _sample(logits, temperature):
        tok = pick(logits, temperature)
        flat = logits.reshape(-1).astype(mx.float32)
        _LOGPROBS.append((flat[tok] - mx.logsumexp(flat)).item())
        return tok

    _sample.keeps_logprob = True
    gen._sample = _sample
    return True



# Qwen3-ASR transcribes its biasing context when the audio gives it too little
# to go on. Measured with a 16-name prompt over 749 recorded clips: 72 (9.6%) came
# back as the whole name list instead of 「はい。」 or a laugh — 40/600 on the call,
# 32/149 on a stream none of those people are in. Leaked clips ran 0.86 s median,
# up to 2.98 s, overlapping the 1.38 s shortest clip the prompt fixed a name in,
# so clip length cannot gate it. Rewording only trades leaks for fixes:
#
#   framing                            full-list leaks   name fixes kept
#   "Proper nouns: a、b、…"                  72/89            17/17
#   "名前：a、b、…"                          47/89            14/17
#   Japanese sentence                       52/89            13/17
#   "Context: … People in the call: …"      21/89            11/17
#
# Real speech names one or two people per clip, a leak recites the list, so
# three distinct entries in one transcript means the context came back out and
# the clip is transcribed again without it. That kept all 17 fixes and removed
# every leak. A leak of a single name — ねえ聞いて -> ハナ聞いて (shape, not a
# recorded line) — is under the threshold and gets through.
PROMPT_ECHO_MIN = 3


def _echoes_prompt(text: str, prompt: str) -> bool:
    """Whether `text` recites the prompt's entries rather than the audio.

    Everything after the first colon is the list — the framing in front of it
    ("Proper nouns:") is load-bearing for recognition, see config.example.toml.
    Single characters are not entries: one kanji turns up in ordinary speech.
    """
    body = re.split(r"[:：]", prompt, maxsplit=1)[-1]
    entries = {e for e in re.split(r"[、,，\s]+", body) if len(e) >= 2}
    return sum(e in text for e in entries) >= PROMPT_ECHO_MIN


class QwenMLXEngine(STTEngine):
    """One long-lived Session, loaded on the thread that will run inference.

    `Session` is the library's explicit model-ownership API, so the 2.5 GB of
    weights are loaded once and reused for every utterance rather than being
    resolved through a process-global cache on each call.

    The load is deferred to first use, and that is not a lazy-init habit — it
    is load-bearing. **MLX streams are thread-local.** A Session built on one
    thread cannot run inference on another; the second call raises

        RuntimeError: There is no Stream(gpu, 1) in current thread.

    `build_engine()` runs on the main thread during stage construction, while
    `transcribe()` is only ever called from the STT worker. Constructing the
    Session eagerly in `__init__` therefore put the model on the wrong thread
    and made *every* utterance raise — caught per-utterance by the worker and
    logged, so the pipeline ran on looking healthy while dropping 100% of its
    output. A 7-minute session recorded zero lines that way.

    Deferring costs nothing: `_stt_worker` calls `warmup()` as its first act,
    so the weights still load once, at startup, off the hot path — just on the
    thread that will actually use them. `mlx_whisper` never hit this because
    its module-level `transcribe()` loads inside the call and so is always on
    the caller's thread; moving the load is what introduced the bug.
    """

    def __init__(self, cfg: SttCfg):
        self.repo = REPO[cfg.model]
        self.cfg = cfg
        self.session = None
        # Said once, at startup: which filters this model turns off.
        log.warning(
            "[stt] %s reports no no_speech_prob — no_speech_thresh=%.2f is "
            "inert for this run; the peak-amplitude gate still runs.",
            cfg.model, cfg.no_speech_thresh,
        )

    def _ensure_session(self):
        """The Session, loaded on the calling thread the first time it is asked.

        Not thread-safe by design rather than by oversight: the pipeline runs
        exactly one STT worker, and a lock here would imply a second caller is
        supported when the thread-affinity constraint above means it is not.
        Two threads sharing one engine is a bug this cannot paper over.
        """
        if self.session is None:
            from mlx_qwen3_asr import Session

            if not _hook_logprobs():
                log.warning(
                    "[stt] could not read %s's token scores (mlx-qwen3-asr "
                    "changed?) — logprob_floor and the unsure dimming are inert "
                    "for this run.", self.cfg.model)
            self.session = Session(self.repo)
        return self.session

    def transcribe(self, audio: np.ndarray) -> STTResult:
        prompt = self.cfg.initial_prompt or ""
        r = self._run(audio, prompt)
        if prompt and _echoes_prompt((r.text or ""), prompt):
            from tsutawaru.stt.filters import DROPS

            DROPS["asr-prompt-echo"] += 1
            r = self._run(audio, "")
        # The last pick is the end-of-text token, which says nothing about the
        # words. Measured with the language pinned; under "detect" the
        # "language Japanese" prefix is scored too, and dilutes the mean.
        lps = _LOGPROBS[:-1]
        return STTResult(
            (r.text or "").strip(),
            r.language or self.cfg.language,
            1.0,
            sum(lps) / len(lps) if lps else NO_LOGPROB,
            NO_SPEECH_PROB,
        )

    def _run(self, audio: np.ndarray, context: str):
        _LOGPROBS.clear()
        return self._ensure_session().transcribe(
            audio,
            language=self.cfg.language if self.cfg.lang_mode == "pinned" else None,
            # Qwen3-ASR takes a free-text biasing context: the fix for the
            # proper nouns this project keeps mangling. Unlike the claim this
            # comment used to make, it *is* transcribable — see
            # PROMPT_ECHO_MIN for how often and what `transcribe` does about it.
            context=context,
        )

    def warmup(self) -> None:
        self.transcribe(np.zeros(16000, dtype=np.float32))
