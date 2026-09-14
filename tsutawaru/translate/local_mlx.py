"""Offline sentence translation on MLX (Apple Silicon).

`mlx-community/Qwen3-4B-Instruct-2507-4bit` is the shipped model, run one line at
a time with no context. Measured on 120 lines of tsutawaru-20260904-183501.jsonl,
M5 Air 16 GB, 4-bit, counting only objective defects (Japanese left in the
English, length blowup, glued words, empty output, degenerate repetition):

    CAT-Translate-1.4b   bare        16 defects   p50 147 ms
    Qwen3-4B             +3 lines     2 defects   p50 429 ms
    Qwen3-4B             bare         1 defect    p50 335 ms

Context is built (`local_context_lines`) and off. It does not reduce defects,
and it introduces one of its own: given a garbled ASR line the model has nothing
to translate, so with three prior lines in the prompt it writes a plausible
continuation of the conversation instead --

    不条ゆ相手がね。        ("不条ゆ" is not a word)
      bare: "Not a word to the other."
      +ctx: "Completely. The other side doesn't show up. It's just something
             that's out of the question for Grace too."   <- "Grace" is context

Fluent invention is worse for a reader than visible nonsense, because only one
of the two can be spotted. A 1.4B model cannot use context at all — it
translates the fenced block along with the line (3 defects -> 112).

This lane shares one GPU with the ASR, and the size shows there. Over 20 clips
from 20260822-live/wav, ASR p50 ran 178 ms alone, 292 ms with CAT-1.4B resident
and translating, 416 ms with Qwen3-4B. `utt_q` is drop-oldest, so if STT falls
far enough behind, utterances are lost rather than delayed.

Only the sentence lane runs here. Glosses go to `translate/jmdict.py`, because a
bare dictionary form is a lookup, not a translation, and because a second model
on the GPU would take time away from this one and from the ASR (see
`TranslationPool` on why the compare lane is not left on during a call).
"""
from __future__ import annotations

import collections
import threading

from tsutawaru.config import TranslateCfg
from tsutawaru.logbus import get_logger
from tsutawaru.translate.base import Translator

log = get_logger(__name__)

PROMPT = "Translate the following Japanese text into English.\n\n {src}"

# The fence and the "do not translate" are both load-bearing. An unfenced
# context prompt makes every model translate the whole block, which on
# 2026-09-08 looked like a model defect until the prompt was the thing that
# changed. Keep them together or the measurement stops meaning anything.
CONTEXT_PROMPT = (
    "Earlier lines of the same conversation, for context only. "
    "Do not translate anything inside <context>.\n"
    "<context>\n{ctx}\n</context>\n\n"
    "Translate only the Japanese text below into English. "
    "Output the English translation and nothing else.\n\n"
    " {src}"
)

# Generous relative to the utterances this sees (VAD caps an utterance at 12 s),
# but it is a runaway guard, not a budget: a repetition loop on garbled ASR input
# is the only thing that ever reaches it.
MAX_TOKENS = 256


def _prompt(text: str, history) -> str:
    """Empty history gives the bare prompt unchanged, not an empty fence.

    Pure, so the self-check can exercise the fencing without loading 2.1 GB of
    weights — and so that `local_context_lines = 0` is provably the old prompt.
    """
    if not history:
        return PROMPT.format(src=text)
    return CONTEXT_PROMPT.format(ctx="\n".join(history), src=text)


class LocalTranslator(Translator):
    """Sentence-only backend. `batch()` deliberately declines the gloss lane."""

    def __init__(self, cfg: TranslateCfg):
        try:
            from mlx_lm import load
            from mlx_lm.sample_utils import make_sampler
        except ImportError as e:  # pragma: no cover - depends on the install
            raise RuntimeError(
                "provider 'local' needs mlx-lm (Apple Silicon): pip install mlx-lm"
            ) from e

        self.repo = cfg.local_model
        # The exact repo, lowercased. Two quants of one model are two backends as
        # far as a cached or recorded result is concerned, so the label says
        # which one rather than which family.
        self.name = self.repo.split("/")[-1].lower()
        log.info("[translate] loading %s", self.repo)
        self.model, self.tok = load(self.repo)
        # Greedy. Translation wants the same input to give the same output every
        # time — a sampled one would make the gloss cache and any A/B run noise.
        self.sampler = make_sampler(temp=0.0)
        # One model, one GPU. mlx generation is not re-entrant, and the sentence
        # and alt lanes are separate threads that would otherwise interleave
        # into the same KV cache. The history lives under the same lock, so the
        # two lanes cannot interleave it either.
        self.lock = threading.Lock()
        self.history = collections.deque(maxlen=max(0, cfg.local_context_lines))
        self._warmup()

    def _warmup(self) -> None:
        """First generate() pays Metal kernel compilation. Do it before the call
        starts, not on the first thing anyone says."""
        self.sentence("こんにちは", remember=False)

    def sentence(self, text: str, remember: bool = True) -> str:
        """`remember=False` translates with the context but does not join it.

        For the compare lane: it is a second transcript of audio the primary
        lane already handled, so letting it in would make the next line's
        context contain the same utterance twice — in the runs this project
        measures with.
        """
        text = text.strip()
        if not text:
            return ""
        from mlx_lm import generate

        with self.lock:
            prompt = self.tok.apply_chat_template(
                [{"role": "user", "content": _prompt(text, self.history)}],
                add_generation_prompt=True,
            )
            out = generate(
                self.model, self.tok, prompt,
                max_tokens=MAX_TOKENS, sampler=self.sampler, verbose=False,
            )
            if remember:
                self.history.append(text)
        return out.strip()

    def batch(self, texts: list[str], remember: bool = True) -> list[str]:
        """Not implemented on purpose — glosses belong to JMdict.

        Returning empty strings rather than raising: `TranslationPool._glosses`
        renders those as "?", which is the honest display for a token the
        dictionary did not have. Raising would blank the whole breakdown row.
        """
        return ["" for _ in texts]


if __name__ == "__main__":  # self-check: python -m tsutawaru.translate.local_mlx
    # The regressions this guards, none of which need the weights: a context
    # block the model is not told to skip (which cost a day on 2026-09-08), a
    # context_lines=0 prompt that is no longer the old bare one, and the compare
    # lane leaking a duplicate transcript into the next line's context.
    assert _prompt("ねこ", []) == PROMPT.format(src="ねこ"), "bare prompt drifted"
    assert "<context>" not in _prompt("ねこ", [])

    p = _prompt("ねこ", ["いぬ", "とり", "さかな"])
    assert p.count("ねこ") == 1, "target line must not also sit inside the fence"
    assert p.index("</context>") < p.index("ねこ"), "target leaked into the fence"
    assert "Do not translate anything inside <context>" in p
    for line in ("いぬ", "とり", "さかな"):
        assert line in p.split("</context>")[0], f"{line} missing from context"

    t = LocalTranslator.__new__(LocalTranslator)  # no weights, no MLX
    t.history = collections.deque(maxlen=3)
    t.lock = threading.Lock()
    t.sampler = t.model = t.tok = None

    def _gen(text, remember=True):
        """The two lines of `sentence` that are not the model call."""
        _prompt(text, t.history)
        if remember:
            t.history.append(text)

    for x in ("a", "b", "c", "d"):
        _gen(x)
    assert list(t.history) == ["b", "c", "d"], t.history  # maxlen drops the oldest
    _gen("alt-lane", remember=False)
    assert list(t.history) == ["b", "c", "d"], "compare lane polluted the context"

    t.history = collections.deque(maxlen=0)  # local_context_lines = 0
    _gen("x")
    assert not t.history and _prompt("y", t.history) == PROMPT.format(src="y")

    print("local_mlx self-check OK")
