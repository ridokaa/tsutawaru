"""MLX-Whisper STT engine for Apple Silicon (Metal / ANE)."""
from __future__ import annotations

import numpy as np

from tsutawaru.config import SttCfg
from tsutawaru.stt.base import STTEngine, STTResult

REPO = {
    "tiny": "mlx-community/whisper-tiny-mlx",
    "base": "mlx-community/whisper-base-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "medium": "mlx-community/whisper-medium-mlx-q4",
    # Whisper distilled for Japanese on the ReazonSpeech corpus (Japanese TV
    # broadcast): the full large-v3 encoder with a 2-layer decoder. Measured
    # against 'medium' on 80 identical utterances from a real session, it
    # fabricates far less on masked speech — where medium invented a
    # "クラウドタワー", this returned 物干し台 and プランター, which fit the
    # surrounding context. It costs ~17% throughput here (this build is not
    # quantised, 'medium' is q4) and it truncates: 5 utterances came back
    # materially shorter than medium's, the shallow decoder giving up early.
    #
    # Its avg_logprob scale differs — mean -0.33 against medium's -0.59 on the
    # same audio — so any threshold fitted to one model is meaningless for the
    # other. stt.logprob_floor is the live one: at -1.0 it culls 15-21% of
    # utterances under 'medium' but only ~2% under this, so switching quietly
    # changes how much speech is discarded before it ever reaches you.
    "kotoba": "kaiinui/kotoba-whisper-v2.0-mlx",
    # large-v3's encoder with a 4-layer decoder, 809M params. Here to isolate the
    # one variable "kotoba" confounds: kotoba is this same encoder distilled on
    # ReazonSpeech, which is Japanese TV *read* speech, while this project only
    # ever sees spontaneous conversation. Same front end, general training, and a
    # decoder deep enough not to give up early the way kotoba's 2-layer one does.
    #
    # q4 (464 MB) rather than the 8-bit or fp16 conversions, which ship
    # safetensors that this mlx_whisper cannot load ("[load_npz] Input must be a
    # zip file"). It is the same format and size class as "medium" above, so the
    # two are directly comparable; an 8-bit arm remains untested.
    #
    # MEASURED AND REJECTED, 2026-09-30. Block-scored CER over the same
    # contiguous 10-minute window of each corpus clip, all three arms in one run:
    #
    #   model     quiet  loud-game  collab  big-collab    avg     RTF
    #   qwen3     0.161      0.497   0.366      0.318   0.336   0.109
    #   kotoba    0.159      0.494   0.557      0.417   0.406   0.153
    #   turbo     0.171      1.235   0.734      0.739   0.720   0.123
    #
    # It loses on all four clips, including the quiet solo stream it was expected
    # to win. On the loud clip it emits 1.69x the reference length (1943 chars
    # against 1148) — a CER above 1.0 is insertion, not substitution, which is
    # the Whisper repetition loop on masked speech that filters.py already
    # documents for "medium". Part of that is a decode setting rather than the
    # weights: transcribe() runs temperature=0.0 with no fallback, which is the
    # configuration Whisper loops under. Chasing it is not worth it, because the
    # two collab arms sink the model on their own — give turbo qwen3's
    # loud-game score outright and its average is still 0.535.
    #
    # Kept rather than deleted so the negative result stays reproducible, the way
    # "medium" is kept with its own losing numbers. Do not make it a default.
    "turbo": "mlx-community/whisper-large-v3-turbo-q4",
}


class MLXWhisperEngine(STTEngine):
    def __init__(self, cfg: SttCfg):
        import mlx_whisper

        self._mlx = mlx_whisper
        try:
            self.repo = REPO[cfg.model]
        except KeyError:
            # Was REPO.get(cfg.model, REPO["small"]): a typo'd or unsupported
            # name quietly loaded `small` and reported nothing, so a bake-off
            # arm could measure a model nobody selected.
            raise RuntimeError(
                f"[stt] model {cfg.model!r} is not a Whisper model this engine "
                f"knows. Available: {', '.join(sorted(REPO))}."
            ) from None
        self.cfg = cfg

    def transcribe(self, audio: np.ndarray) -> STTResult:
        r = self._mlx.transcribe(
            audio,
            path_or_hf_repo=self.repo,
            language=self.cfg.language if self.cfg.lang_mode == "pinned" else None,
            temperature=0.0,
            condition_on_previous_text=False,
            initial_prompt=self.cfg.initial_prompt or None,
            no_speech_threshold=self.cfg.no_speech_thresh,
            word_timestamps=False,
            verbose=None,
        )
        segs = r.get("segments", [])
        avg = float(np.mean([s["avg_logprob"] for s in segs])) if segs else -10.0
        nsp = float(np.mean([s["no_speech_prob"] for s in segs])) if segs else 1.0
        return STTResult(
            r.get("text", "").strip(),
            r.get("language", "ja"),
            1.0,
            avg,
            nsp,
        )

    def warmup(self) -> None:
        self.transcribe(np.zeros(16000, dtype=np.float32))
