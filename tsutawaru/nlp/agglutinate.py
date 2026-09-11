"""Agglutination pass: merge bound morphemes into their head.

Learner-focused segmentation that groups bound morphemes (e.g. です+ね -> ですね,
おはよう+ござい+ます -> おはようございます) while keeping head base_form for dictionary lookups.
"""
from __future__ import annotations

from dataclasses import dataclass

from tsutawaru.models import Token

# Heads that a bound morpheme may attach to.
ATTACH_HEADS = {"動詞", "形容詞", "形容動詞", "助動詞", "感動詞"}

# 補助動詞 — verbs that follow て/で carrying grammar rather than meaning.
#
# This list exists because neither dictionary's tag can be trusted to mark it.
# IPADIC files all of these under 動詞,非自立 — but it files 続ける, 始める, 直す
# and 上げる there too, and those are ordinary verbs with meanings a learner
# needs. Merging on the tag alone produced 変えて続ける as one word glossed
# 変える ("to change"), losing 続ける entirely, and 立ち上げて glossed 立つ
# ("to stand") for a word that means "to launch". UniDic is no better: its
# 非自立可能 states that a verb *may* be auxiliary, never that it is, so keying
# on that merges nothing at all and 見ている comes apart into two words.
#
# So the decision is made here, explicitly, and is the same under either
# backend. Membership is by dictionary form, which is stable across conjugation.
#
# Deliberately excluded: the aspectual compounds 始める / 終わる / 直す / 続ける
# / 出す. They are auxiliary-like in grammar but lexical in meaning — 食べ始める
# is "start eating", and folding it into 食べる throws away the half that says
# "start". Splitting them shows both words, which is what the breakdown is for.
AUX_VERBS = frozenset({
    "いる", "てる", "でる", "おる",        # progressive (てる/でる are IPADIC's
                                          # separate entries for 見てる/踏んでる)
    "ある",                               # resultative ～てある
    "しまう",                             # completive ～てしまう
    "おく",                               # preparatory ～ておく
    "くれる", "くださる", "ください",        # benefactive, inward
    "もらう", "いただく",                   # benefactive, received
    "あげる", "やる",                      # benefactive, outward
    "みる",                               # attemptive ～てみる
    "いく", "行く", "くる", "来る",         # directional ～ていく / ～てくる
    # UniDic writes the same lemmas in kanji — 居る for いる, 仕舞う for しまう.
    # Both spellings are listed so one rule covers both backends; see the note
    # in tokenizer.py on UniDic's archaic lemma orthography.
    "居る", "有る", "仕舞う", "置く", "下さる", "貰う", "頂く", "上げる", "遣る",
    "呉れる", "見る",
})

# Connective forms an auxiliary attaches across. ちゃ / じゃ are IPADIC's
# analysis of the spoken contraction: なっちゃいました comes out as
# なっ + ちゃ[接続助詞] + い(いる) + まし + た, with no て anywhere in it.
TE_FORMS = ("て", "で", "ちゃ", "じゃ")

# Auxiliaries that swallowed the connective whole, leaving nothing to check:
# ておく -> とく, てしまう -> ちゃう. They have no lexical use to be confused
# with, so they attach on part of speech alone.
FUSED_AUX = frozenset({"ちゃう", "じゃう", "とく", "どく"})


SMALL_TSU = ("っ", "ッ")

# Grammar that agglutination would otherwise erase.
#
# A merged token keeps the HEAD's dictionary form so gloss lookups hit, but the
# gloss is then the plain affirmative: 行けなかった carries base_form 行ける and
# glossed as "i can go" — the exact inverse of what was said. Negation and tense
# live only in the folded-in morphemes, so record them as they are absorbed.
#
# Keyed on the auxiliary's dictionary form, which is stable across conjugation
# (なかっ, なく, ない all report base_form ない).
INFLECTION_LABELS = {
    "ない": "negative",
    "ぬ": "negative",
    "ん": "negative",
    "た": "past",
    "ます": "polite",
    "です": "polite",
    "たい": "want to",
    "う": "volitional",
    "よう": "volitional",
    "れる": "passive/potential",
    "られる": "passive/potential",
    "せる": "causative",
    "させる": "causative",
    # Spoken contractions of ～ている are separate IPADIC entries, not forms of
    # いる: 踏んでる yields base_form でる, 見てる yields てる.
    "いる": "progressive",
    "てる": "progressive",
    "でる": "progressive",
    "しまう": "completive",
    "ちゃう": "completive",
    "おく": "in advance",
    "ください": "please",
    "らしい": "seems",
}


def _inflection_label(tok: Token) -> str | None:
    """Grammatical label for a bound morpheme, or None if it carries none.

    Gated on part of speech, not on the surface alone: ん is the negative
    auxiliary in 分からん but a nominaliser (名詞,非自立) in 何か入れるのある，
    and labelling the latter "negative" would invert a correct reading.
    """
    if tok.pos == "助動詞" or (tok.pos == "動詞" and tok.pos1 in ("非自立", "接尾")):
        return INFLECTION_LABELS.get(tok.base_form)
    return None


@dataclass
class _Group:
    """Bookkeeping for the group currently being built.

    Kept beside the tokens rather than on them: `Token` is a slots dataclass,
    and parser internals do not belong in the public model anyway.
    """

    numeric: bool = False       # group opened with a 名詞,数 (a numeral)
    has_suffix: bool = False    # a 名詞,接尾 has already been absorbed


def _dangling_sokuon(tok: Token) -> bool:
    """Does this token end in a bare geminate marker?

    A trailing っ/ッ is never a word ending in Japanese — it geminates the
    consonant that *starts the next mora*, so it cannot be romanised alone.
    pykakasi falls back to spelling it out: もっ -> "motsu" rather than the
    "moc-" of もっち -> "motchi". Whenever IPADIC leaves one dangling (common on
    names and coinages it does not know) the fragment must rejoin its neighbour.
    """
    src = tok.reading or tok.surface
    return bool(src) and src[-1] in SMALL_TSU


def _attaches(head: Token | None, t: Token, group: "_Group | None" = None) -> bool:
    """Should token `t` merge into the group headed by `head`?"""
    if head is None:
        return False
    # Structural, not grammatical: a dangling geminate marker is always an
    # artefact of segmentation and must absorb the following token regardless
    # of parts of speech.
    if _dangling_sokuon(head):
        return True
    if t.pos == "助動詞":  # ござい/ます/です/た
        return head.pos in ATTACH_HEADS  # NOT after 名詞
    if t.pos == "動詞" and t.pos1 == "接尾":  # 〜がる / 〜めく
        return head.pos in ATTACH_HEADS
    if t.pos == "動詞" and t.base_form in FUSED_AUX:  # ...ちゃう / ...とく
        return head.pos in ATTACH_HEADS
    if t.pos == "動詞" and t.base_form in AUX_VERBS:  # ...ている / ...ください
        # Gated on the て/で the auxiliary attaches across, not on either
        # dictionary's subtype tag. `head` accumulates as the group grows, so by
        # the time いる is judged in 見ている the head reads 見て.
        #
        # The gate is what makes the list safe to widen: 見る, 上げる and ある are
        # all ordinary verbs too, and only ～てみる, ～てあげる and ～てある are
        # auxiliary uses. Requiring the て means the entry cannot fire on the
        # lexical one. It also makes this rule backend-independent — it never
        # consults 非自立 or 非自立可能, so it behaves the same under IPADIC and
        # UniDic, which disagree about that tag in opposite directions.
        #
        # The contractions carry their own て: 見てる segments as 見 + てる, so
        # the head is bare 見 and the connective is inside the auxiliary. Both
        # sides are checked for that reason.
        if not head.pos in ATTACH_HEADS:
            return False
        return head.surface.endswith(TE_FORMS) or t.surface.startswith(TE_FORMS)
    if t.pos == "助詞" and t.pos1 == "終助詞":  # ね / よ / な
        return head.pos in ATTACH_HEADS
    if t.pos == "名詞" and t.pos1 == "接尾":  # 〜さん / 〜たち / counters
        if head.pos != "名詞":
            return False
        # A numeral takes exactly ONE suffix — its counter (一+回 -> 一回).
        # Anything further is IPADIC guessing: in 一回家帰ろう it tags 家 as the
        # profession suffix 〜家 (reading カ), producing 一回家 "ichikaika"
        # instead of 一回 / 家 "ikkai / uchi". Stop the chain after the counter.
        if group is not None and group.numeric:
            return not group.has_suffix
        return True
    if t.pos == "助詞" and t.pos1 == "接続助詞" and t.surface in TE_FORMS:
        return head.pos in ("動詞", "形容詞", "助動詞")
    return False


def agglutinate(tokens: list[Token]) -> list[Token]:
    """Merge bound morphemes into their head.

    Surface and reading concatenate; base_form stays the HEAD's dictionary form
    so gloss lookups still hit (食べ+た -> surface 食べた, base 食べる).
    """
    out: list[Token] = []
    heads: list[Token] = []
    groups: list[_Group] = []  # per-group bookkeeping; Token has slots=True
    for t in tokens:
        head = heads[-1] if heads else None
        group = groups[-1] if groups else None
        if _attaches(head, t, group):
            out[-1].surface += t.surface
            # Fall back to the SURFACE for any part with no dictionary reading.
            # Plain concatenation silently drops those morphemes from the
            # pronunciation: 妖(reading "") + ちゃん(チャン) yielded "チャン",
            # so the romaji read "chan" and 妖 vanished from the line entirely.
            # pykakasi reads kanji, so a mixed 妖チャン romanises correctly.
            if out[-1].reading or t.reading:
                out[-1].reading = (
                    (out[-1].reading or out[-1].surface[: -len(t.surface)])
                    + (t.reading or t.surface)
                )
            if t.pos == "名詞" and t.pos1 == "接尾" and group is not None:
                group.has_suffix = True
            # base_form deliberately unchanged: 食べ+た -> surface 食べた, base 食べる
            # ...which is exactly why the grammar it carried has to be kept.
            label = _inflection_label(t)
            if label and label not in out[-1].infl:
                out[-1].infl.append(label)
        else:
            tok_copy = Token(
                surface=t.surface,
                base_form=t.base_form,
                reading=t.reading,
                romaji=t.romaji,
                pos=t.pos,
                pos1=t.pos1,
                gloss=t.gloss,
            )
            out.append(tok_copy)
            heads.append(tok_copy)
            groups.append(_Group(numeric=(t.pos == "名詞" and t.pos1 == "数")))
    return out
