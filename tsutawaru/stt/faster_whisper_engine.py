"""Faster-Whisper STT engine (CTranslate2 backend for CUDA / CPU)."""
from __future__ import annotations

import numpy as np
from faster_whisper import WhisperModel

from tsutawaru.config import SttCfg
from tsutawaru.stt.base import STTEngine, STTResult


# WhisperModel takes either a size name or a Hugging Face repo id, so only the
# names that are not sizes need mapping. The CTranslate2 build is a different
# artefact from the MLX one in mlx_whisper_engine.REPO — same model, converted
# for a different runtime — which is why the two tables cannot be shared.
REPO = {
    "kotoba": "kotoba-tech/kotoba-whisper-v2.0-faster",
}


class FasterWhisperEngine(STTEngine):
    def __init__(self, cfg: SttCfg, device: str = "cpu", compute_type: str = "int8"):
        self.model = WhisperModel(
            REPO.get(cfg.model, cfg.model),
            device=device,
            compute_type=compute_type,
            num_workers=1,
        )
        self.cfg = cfg

    def transcribe(self, audio: np.ndarray) -> STTResult:
        segments, info = self.model.transcribe(
            audio,
            language=self.cfg.language if self.cfg.lang_mode == "pinned" else None,
            task="transcribe",
            beam_size=self.cfg.beam_size,
            best_of=1,
            temperature=0.0,
            vad_filter=False,
            condition_on_previous_text=False,
            without_timestamps=True,
            initial_prompt=self.cfg.initial_prompt or None,
            no_speech_threshold=self.cfg.no_speech_thresh,
        )
        segs = list(segments)
        text = "".join(s.text for s in segs).strip()
        avg = float(np.mean([s.avg_logprob for s in segs])) if segs else -10.0
        nsp = float(np.mean([s.no_speech_prob for s in segs])) if segs else 1.0
        return STTResult(text, info.language, info.language_probability, avg, nsp)

    def warmup(self) -> None:
        self.transcribe(np.zeros(16000, dtype=np.float32))
