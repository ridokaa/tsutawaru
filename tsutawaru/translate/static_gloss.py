"""Zero-latency static gloss dictionary for particles, copulas, and common conversational words."""
from __future__ import annotations

from typing import Optional
from tsutawaru.models import Token

STATIC = {
    "は": "topic marker",
    "が": "subject marker",
    "を": "object marker",
    "に": "to / at / in",
    "で": "at / by / with",
    "と": "and / with",
    "の": "possessive / of",
    "も": "also / too",
    "へ": "toward",
    "から": "from",
    "まで": "until",
    "より": "than",
    "ね": "right? / isn't it",
    "よ": "you know (emphasis)",
    "な": "casual emphasis",
    "か": "question marker",
    "けど": "but / although",
    "ので": "because",
    "です": "is (polite copula)",
    "ですね": "right / isn't it",
    "ですよ": "you know (polite)",
    "ますね": "right? (polite verb)",
    "ます": "polite verb ending",
    "でした": "was (polite past)",
    "だ": "is (plain copula)",
    "ある": "to exist (inanimate)",
    "いる": "to exist (animate)",
    "する": "to do",
    "なる": "to become",
    "ちょっと": "a little / somewhat",
    # JMdict's first sense is "terrible / dreadful", which is correct and wrong:
    # in speech this is almost always admiration. Observed on line 1 of the
    # 20260822 corpus (すごいねアボルスコ).
    "すごい": "amazing / wow",
    "そう": "so / that way",
    "でも": "but / however",
    "じゃ": "well then",
    "えーと": "um…",
    "あの": "um / that",
    "はい": "yes",
    "いいえ": "no",
    "ええ": "yeah",
    "うん": "mm-hmm",
    "まあ": "well…",
    "おはようございます": "good morning",
    "ありがとうございます": "thank you",
    "こんにちは": "hello",
    "すみません": "excuse me / sorry",
}


def static_gloss(tok: Token) -> Optional[str]:
    """Merged surface first, then the head's dictionary form.

    Order matters: ですね (merged) must beat です (base_form) so the breakdown
    reads 'right / isn't it' rather than 'is (polite copula)'.
    """
    return STATIC.get(tok.surface) or STATIC.get(tok.base_form)
