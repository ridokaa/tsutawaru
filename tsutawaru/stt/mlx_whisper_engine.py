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
}


class MLXWhisperEngine(STTEngine):
    def __init__(self, cfg: SttCfg):
        import mlx_whisper

        self._mlx = mlx_whisper
        self.repo = REPO.get(cfg.model, REPO["small"])
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
