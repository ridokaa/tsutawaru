"""Offline sentence translation with CAT-Translate on MLX (Apple Silicon).

`cyberagent/CAT-Translate-1.4b` is a 1.4 B JA<->EN specialist (MIT, Sarashina 2.2
base) that outscores 9-12 B general models on Ja-En. `hotchpotch` publishes a
4-bit MLX conversion, so there is nothing to convert locally.

Measured on the 40-line 20260822 session corpus, M5 Air 16 GB, 4-bit:
p50 126 ms, p95 329 ms, max 340 ms — below Google's round-trip and far inside
`sentence_timeout_s`.

Only the sentence lane runs here. Glosses go to `translate/jmdict.py`, because a
bare dictionary form is a lookup, not a translation, and because a second model
on the GPU would take time away from this one and from the ASR (see
`TranslationPool` on why the compare lane is not left on during a call).
"""
from __future__ import annotations

import threading

from tsutawaru.config import TranslateCfg
from tsutawaru.logbus import get_logger
from tsutawaru.translate.base import Translator

log = get_logger(__name__)

PROMPT = "Translate the following Japanese text into English.\n\n {src}"

# Generous relative to the utterances this sees (VAD caps an utterance at 12 s),
# but it is a runaway guard, not a budget: a repetition loop on garbled ASR input
# is the only thing that ever reaches it.
MAX_TOKENS = 256


class LocalTranslator(Translator):
    """Sentence-only backend. `batch()` deliberately declines the gloss lane."""

    name = "cat-translate"

    def __init__(self, cfg: TranslateCfg):
        try:
            from mlx_lm import load
            from mlx_lm.sample_utils import make_sampler
        except ImportError as e:  # pragma: no cover - depends on the install
            raise RuntimeError(
                "provider 'local' needs mlx-lm (Apple Silicon): pip install mlx-lm"
            ) from e

        self.repo = cfg.local_model
        log.info("[translate] loading %s", self.repo)
        self.model, self.tok = load(self.repo)
        # Greedy. Translation wants the same input to give the same output every
        # time — a sampled one would make the gloss cache and any A/B run noise.
        self.sampler = make_sampler(temp=0.0)
        # One model, one GPU. mlx generation is not re-entrant, and the sentence
        # and alt lanes are separate threads that would otherwise interleave
        # into the same KV cache.
        self.lock = threading.Lock()
        self._warmup()

    def _warmup(self) -> None:
        """First generate() pays Metal kernel compilation. Do it before the call
        starts, not on the first thing anyone says."""
        self.sentence("こんにちは")

    def sentence(self, text: str) -> str:
        text = text.strip()
        if not text:
            return ""
        from mlx_lm import generate

        prompt = self.tok.apply_chat_template(
            [{"role": "user", "content": PROMPT.format(src=text)}],
            add_generation_prompt=True,
        )
        with self.lock:
            out = generate(
                self.model, self.tok, prompt,
                max_tokens=MAX_TOKENS, sampler=self.sampler, verbose=False,
            )
        return out.strip()

    def batch(self, texts: list[str]) -> list[str]:
        """Not implemented on purpose — glosses belong to JMdict.

        Returning empty strings rather than raising: `TranslationPool._glosses`
        renders those as "?", which is the honest display for a token the
        dictionary did not have. Raising would blank the whole breakdown row.
        """
        return ["" for _ in texts]
