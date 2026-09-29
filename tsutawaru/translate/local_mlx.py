"""Offline sentence translation on MLX (Apple Silicon).

`mlx-community/Qwen3-4B-Instruct-2507-4bit` is the shipped model, run one line at
a time with no context. Measured on 120 lines of tsutawaru-20260904-183501.jsonl,
M5 Air 16 GB, 4-bit, counting only objective defects (Japanese left in the
English, length blowup, glued words, empty output, degenerate repetition):

    CAT-Translate-1.4b   bare        16 defects   p50 147 ms
    Qwen3-4B             +3 lines     2 defects   p50 429 ms
    Qwen3-4B             bare         1 defect    p50 335 ms

Context is on (`local_context_lines = 3`) and holds previous *English* output,
not Japanese source — a block of English cannot be mistaken for text awaiting
translation, which is how the Japanese version failed. No countable metric
separates the variants on 120 lines:

    bare                 1 defect   p50 322 ms   subject flips 55.4%
    +3 Japanese lines    2 defects  p50 403 ms   subject flips 50.0%
    +3 English lines     3 defects  p50 413 ms   subject flips 49.2%
    +English +tone ask   5 defects  p50 440 ms   subject flips 52.7%

It was settled by a human grading experiments/reports/mt-context-20260914.md:
19 lines to 16 overall, and 6 to 3 on the subset context exists for — lines
where the Japanese names no subject and the two variants supply a different one.
The Japanese states no subject on 108 of those 120 lines, so this is the failure
that reads as wrong, and none of the columns above can see it.

    朝から並んでたのに売り切れだったらしい。
      bare:   "I queued from the morning but it was apparently sold out."
      +en:    "He queued from the morning but it was apparently sold out."

Asking for tone on top was tested and rejected: it only made output longer
(1 blowup -> 5). The name rule came out of the same grading and is measured
separately — see NAME_RULE.

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

from typing import TYPE_CHECKING

from tsutawaru.logbus import get_logger
from tsutawaru.translate.base import Translator

if TYPE_CHECKING:  # `from __future__ import annotations` keeps this out of
    from tsutawaru.config import TranslateCfg  # runtime — config imports REPO.

log = get_logger(__name__)

#: Short names for `--translator`, the way `stt.REPO` backs `--asr`. A key here
#: names the model *and* implies this backend, because a named local model is
#: obviously local — spelling both would say the same thing twice. Anything not
#: listed goes through `--translator mlx`, which uses `translate.local_model`
#: verbatim, so the registry never has to be exhaustive.
REPO = {
    # Graded 37/40 against the 2B's 13 on the 20260915 sheet; the shipped
    # default. Reasons by default — `_chat` turns that off, and must.
    "qwen3.5": "mlx-community/Qwen3.5-4B-MLX-4bit",
}
# CAT-Translate-1.4b was the budget key here and was removed on 2026-09-21. It
# is 2.5x faster and leaks the prompt back as output: it translates the whole
# user turn, instructions included, because a 1.4B translation specialist has no
# instruction-following layer to separate them. All three prompt shapes were
# measured on 160 stream lines (experiments/reports/model-bakeoff-20260920.md) —
# instructions in the user turn 17 defects, in a system turn 26, omitted
# entirely 24 — against qwen3.5's 2. `local_model` still loads it by repo name
# for anyone who wants it; it is no longer offered as a choice.

# The name rule is measured, not stylistic. Without it ミドリさん comes out as "a
# green colour" and ハナさん as "a flower" — 4 of 15 name lines
# survived on the stream corpus and 3 became English dictionary words.
#
# Three things about the wording are load-bearing, and each cost a defect to
# find. **Retune it whenever the model changes**: none of the three transferred.
#
# 1. No example names. A first version read "...keep the honorific (Midori-san,
#    Hana-san)" and the model copied those two straight out of the prompt —
#    ハナさん, ソラさん and ユキさん all came back "Midori-san", in bare
#    mode where there was no context to blame. A name in a prompt is a one-shot
#    example, and these models take it.
# 2. Scoped to the honorific, not to "names". "Write personal names as romanized
#    Japanese" bled onto whole sentences: そろそろ行きますね -> "Sorosoro ikimasu
#    ne".
# 3. Shaped as a constraint on the output, not an instruction about names. The
#    scoped wording above was tuned against Qwen3 and bled again on Qwen3.5;
#    saying what the output may contain fixed it where saying what to do did not.
#
#    Qwen3.5-4B, 110 lines per corpus:     call bleed   stream bleed   names ok/bad
#      no rule                                  0             0            4 / 11
#      scoped (tuned on Qwen3)                  5             1            9 /  6
#      output-shaped (shipped)                  0             0            7 /  8
#
# The residual is not fixable here: the translator can only romanize what the
# ASR heard, and a handle like ソルトさん really does mean "salt". That one belongs to
# `stt.initial_prompt` — see config.example.toml.
NAME_RULE = (
    "Your output must be English, with no romanized Japanese in it, except for a "
    "person's name — the word directly before さん, ちゃん or くん — which you "
    "romanize and hyphenate to the honorific rather than translating."
)

PROMPT = (
    "Translate the following Japanese text into English. " + NAME_RULE +
    "\n\n {src}"
)

# The fence and the "do not translate" are both load-bearing. An unfenced
# context prompt makes every model translate the whole block, which on
# 2026-09-08 looked like a model defect until the prompt was the thing that
# changed. Keep them together or the measurement stops meaning anything.
CONTEXT_PROMPT = (
    "The conversation so far, already translated:\n{ctx}\n\n"
    "Translate the next Japanese line into English, consistent with what came "
    "before — especially who is being talked about, since Japanese usually "
    "leaves the subject unsaid. " + NAME_RULE + " Output the English only."
    "\n\n {src}"
)

# Generous relative to the utterances this sees (VAD caps an utterance at 12 s),
# but it is a runaway guard, not a budget: a repetition loop on garbled ASR input
# is the only thing that ever reaches it.
MAX_TOKENS = 256


def _prompt(text: str, history) -> str:
    """Empty history gives the bare prompt unchanged, not an empty context block.

    `history` holds previous *English output*, not Japanese source. That is the
    whole difference from the version that failed: a block of English cannot be
    mistaken for text awaiting translation, so the model has nothing to copy
    through and no fence is needed to stop it. It also carries the subjects
    already resolved, which is what the next line usually needs — Japanese
    states no subject on 108 of 120 measured lines.

    Pure, so the self-check can exercise it without loading 2.1 GB of weights —
    and so that `local_context_lines = 0` is provably the old bare prompt.
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
                "--translator mlx needs mlx-lm (Apple Silicon): pip install mlx-lm"
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
        self.max_context = max(0, cfg.local_context_lines)
        # Empty until a source says otherwise. `set_context(True)` turns it on;
        # nothing does that for a call, which is the whole point.
        self.history: collections.deque = collections.deque(maxlen=0)
        self._warmup()

    def set_context(self, on: bool) -> None:
        """Enable or disable conversational context, keeping what still fits.

        Called when the capture source changes, so a session that starts on a
        stream and switches to a call stops carrying context into the call —
        and drops what it had, because those lines were a different
        conversation entirely.
        """
        n = self.max_context if on else 0
        if n == self.history.maxlen:
            return
        with self.lock:
            self.history = collections.deque(
                list(self.history)[-n:] if n else [], maxlen=n)
        log.info("[translate] conversational context %s", f"on ({n} lines)" if n else "off")

    def _warmup(self) -> None:
        """First generate() pays Metal kernel compilation. Do it before the call
        starts, not on the first thing anyone says."""
        out = self.sentence("こんにちは", remember=False)
        # A thinking model that slipped past `_chat` shows up here and nowhere
        # else until the call has started and every line is 15 s late.
        if len(out) > 200:
            log.warning(
                "[translate] %s returned %d characters for a two-word warmup — "
                "it is probably reasoning out loud; check that its chat template "
                "honours enable_thinking=False", self.name, len(out))

    def _chat(self, msg: list[dict]) -> str:
        """Chat template with thinking off.

        Not an optimisation — a requirement. Qwen3.5 reasons by default, and on
        this lane that is 15 s and 1.8 kB of "Thinking Process: 1. Analyze the
        Request" per line against 440 ms and a translation. A model whose
        template has no such flag rejects the keyword, so the plain call is the
        fallback rather than the default: a silent 34x slowdown is the failure
        this shape exists to prevent.
        """
        try:
            return self.tok.apply_chat_template(
                msg, add_generation_prompt=True, enable_thinking=False)
        except (TypeError, ValueError, KeyError):
            return self.tok.apply_chat_template(msg, add_generation_prompt=True)

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

        msg = [{"role": "user", "content": _prompt(text, self.history)}]
        with self.lock:
            prompt = self._chat(msg)
            out = generate(
                self.model, self.tok, prompt,
                max_tokens=MAX_TOKENS, sampler=self.sampler, verbose=False,
            )
            out = out.strip()
            if remember:
                # The English, not the Japanese. See `_prompt`.
                self.history.append(out)
        return out

    def batch(self, texts: list[str], remember: bool = True) -> list[str]:
        """Not implemented on purpose — glosses belong to JMdict.

        Returning empty strings rather than raising: `TranslationPool._glosses`
        renders those as "?", which is the honest display for a token the
        dictionary did not have. Raising would blank the whole breakdown row.
        """
        return ["" for _ in texts]


if __name__ == "__main__":  # self-check: python -m tsutawaru.translate.local_mlx
    # The regressions this guards: a context_lines=0 prompt that is no longer
    # the old bare one, Japanese source leaking into a block that is supposed to
    # hold only English, and the compare lane putting a duplicate transcript
    # into the next line's context.
    assert _prompt("ねこ", []) == PROMPT.format(src="ねこ"), "bare prompt drifted"
    # Against NAME_RULE itself, not a quoted phrase: the wording is measured and
    # will be tuned again, and a hardcoded copy here would fail for the wrong
    # reason the next time it is.
    for p_ in (_prompt("ねこ", []), _prompt("ねこ", ["A."])):
        assert NAME_RULE in p_, "the name rule fell out of a branch"

    p = _prompt("ねこ", ["The dog barked.", "So did the bird.", "Then a fish."])
    assert p.count("ねこ") == 1, "target line must not also sit in the context"
    assert p.rindex("ねこ") > p.index("Then a fish."), "target leaked into the context"
    for line in ("The dog barked.", "So did the bird.", "Then a fish."):
        assert line in p, f"{line} missing from context"

    t = LocalTranslator.__new__(LocalTranslator)  # no weights, no MLX
    t.history = collections.deque(maxlen=3)
    t.lock = threading.Lock()
    t.sampler = t.model = t.tok = None

    def _gen(text, out, remember=True):
        """The two lines of `sentence` that are not the model call."""
        _prompt(text, t.history)
        if remember:
            t.history.append(out)

    for jp, en in [("あ", "A."), ("い", "B."), ("う", "C."), ("え", "D.")]:
        _gen(jp, en)
    assert list(t.history) == ["B.", "C.", "D."], t.history  # maxlen drops oldest
    assert not any("あ" in h or "え" in h for h in t.history), "Japanese in the context"
    _gen("alt", "ALT.", remember=False)
    assert list(t.history) == ["B.", "C.", "D."], "compare lane polluted the context"

    t.history = collections.deque(maxlen=0)  # local_context_lines = 0
    _gen("x", "X.")
    assert not t.history and _prompt("y", t.history) == PROMPT.format(src="y")

    print("local_mlx self-check OK")
