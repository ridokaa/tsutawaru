"""Factory to autodetect hardware and instantiate the appropriate STTEngine."""
from __future__ import annotations

import importlib.util
import platform

from tsutawaru.config import SttCfg
from tsutawaru.stt.base import STTEngine
from tsutawaru.stt.qwen_mlx_engine import REPO as QWEN_REPO


def build_engine(cfg: SttCfg) -> STTEngine:
    want = cfg.backend
    is_apple = platform.system() == "Darwin" and platform.machine() == "arm64"

    # Qwen3-ASR is a decoder-only ASR LLM, not a Whisper variant: CTranslate2
    # cannot load it and there is no CPU or CUDA path here. So it either works
    # or raises — falling through to a Whisper model nobody asked for would
    # make every later quality comparison meaningless.
    if cfg.model in QWEN_REPO:
        if not is_apple:
            raise RuntimeError(
                f"[stt] model '{cfg.model}' runs only on Apple Silicon "
                "(MLX). Use model='kotoba' on this machine."
            )
        if not importlib.util.find_spec("mlx_qwen3_asr"):
            raise RuntimeError(
                f"[stt] model '{cfg.model}' needs mlx-qwen3-asr. "
                "Install it with: pip install mlx-qwen3-asr"
            )
        from tsutawaru.stt.qwen_mlx_engine import QwenMLXEngine

        return QwenMLXEngine(cfg)

    if want in ("auto", "mlx") and is_apple and importlib.util.find_spec("mlx_whisper"):
        from tsutawaru.stt.mlx_whisper_engine import MLXWhisperEngine

        return MLXWhisperEngine(cfg)

    from tsutawaru.stt.faster_whisper_engine import FasterWhisperEngine

    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            ct = "float16" if cfg.compute_type == "auto" else cfg.compute_type
            return FasterWhisperEngine(cfg, device="cuda", compute_type=ct)
    except Exception:
        pass

    ct = "int8" if cfg.compute_type == "auto" else cfg.compute_type
    return FasterWhisperEngine(cfg, device="cpu", compute_type=ct)
