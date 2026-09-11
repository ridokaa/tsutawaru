"""Japanese morphological tokenization preserving POS subtype (pos1)."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from tsutawaru.config import NlpCfg
from tsutawaru.models import Token

#: Where an unpacked NINJAL UniDic release is looked for. Outside the project on
#: purpose: the dictionary is ~1.2 GB unpacked, and the backup snapshots this
#: project uses instead of version control would copy it every time.
UNIDIC_HOME = Path(
    os.environ.get("TSUTAWARU_UNIDIC_HOME", "~/.local/share/tsutawaru")
).expanduser()


def resolve_unidic_dir(explicit: str = "") -> Optional[Path]:
    """Locate a UniDic dictionary directory, or None.

    An explicit path wins and is not second-guessed. Otherwise the newest
    unidic-* directory under UNIDIC_HOME is used — NINJAL versions its releases
    by date (unidic-cwj-202512), so a plain sort puts the newest last.
    """
    if explicit:
        p = Path(explicit).expanduser()
        return p if (p / "sys.dic").exists() else None
    if not UNIDIC_HOME.is_dir():
        return None
    found = sorted(
        d for d in UNIDIC_HOME.glob("unidic-*") if (d / "sys.dic").exists()
    )
    return found[-1] if found else None

# IPADIC readings that are right for the tag and wrong for conversation.
#
# Keyed on (surface, pos, pos1), never on surface alone, because the surface is
# not what goes wrong — the tag is. IPADIC holds several readings per kanji and
# the Viterbi path commits to one per token, so 下 reads シタ on its own and モト
# as soon as a determiner precedes it and a 並立助詞 follows: 「この下とか」 comes
# out "kono moto toka". Keying on surface would rewrite the many places the same
# kanji is already read correctly. Note the bound-noun tag itself is innocent —
# 上・中・事・物・所 all take 非自立 with the right reading; only these two flip.
#
# Both entries trade a formal reading for a conversational one, knowingly:
#   下/非自立 is genuinely モト in 「法の下」(ほうのもと) — legal register.
#   後/非自立 is genuinely ノチ in 「数年の後」(すうねんののち) — literary.
# Against that, 「この下」(shita) and 「この後」(ato) are ordinary speech, and
# speech is the entire input domain. The trade is only correct in this direction;
# a tool that read documents would want the opposite table.
READING_FIXES: dict[tuple[str, str, str], str] = {
    ("下", "名詞", "非自立"): "シタ",
    ("後", "名詞", "非自立"): "アト",
}


def fix_reading(tok: Token) -> Token:
    """Apply READING_FIXES in place. Returns the token so it can wrap a call.

    Runs at tokenization time rather than in the NLP pipeline so that both
    backends are covered by one table, and — the part that matters — so it lands
    *before* agglutination, which concatenates readings. A correction applied
    after the merge would have to unpick 潜って's reading from its parts.
    """
    fixed = READING_FIXES.get((tok.surface, tok.pos, tok.pos1))
    if fixed:
        tok.reading = fixed
    return tok


class JanomeTokenizer:
    def __init__(self):
        from janome.tokenizer import Tokenizer

        self._t = Tokenizer()

    def tokenize(self, text: str) -> list[Token]:
        out = []
        for t in self._t.tokenize(text):
            p = t.part_of_speech.split(",")  # e.g. 助詞,終助詞,*,*
            out.append(fix_reading(
                Token(
                    surface=t.surface,
                    base_form=t.base_form if t.base_form != "*" else t.surface,
                    reading=t.reading if t.reading != "*" else "",
                    pos=p[0],
                    pos1=p[1] if len(p) > 1 else "",
                )
            ))
        return out


class MeCabTokenizer:
    def __init__(self):
        import ipadic
        import MeCab

        self._t = MeCab.Tagger(ipadic.MECAB_ARGS)

    def tokenize(self, text: str) -> list[Token]:
        out, node = [], self._t.parseToNode(text)
        while node:
            f = node.feature.split(",")
            if node.surface:
                out.append(fix_reading(
                    Token(
                        surface=node.surface,
                        base_form=f[6] if len(f) > 6 and f[6] != "*" else node.surface,
                        reading=f[7] if len(f) > 7 and f[7] != "*" else "",
                        pos=f[0],
                        pos1=f[1] if len(f) > 1 else "",
                    )
                ))
            node = node.next
        return out


# UniDic's own tagset, mapped onto the IPADIC one the rest of the pipeline
# speaks. Everything downstream — agglutinate.py's merge rules, NlpCfg.drop_pos,
# romaji.py's particle overrides — is keyed on IPADIC tags, so translating here
# keeps this a drop-in swap: exactly one variable changes, which is the only way
# a comparison between the two means anything. Unmapped pairs pass through.
#
# Two mappings are deliberately lossy and worth knowing about:
#
#   非自立可能 -> 自立.  UniDic marks verbs and adjectives that *may* be
#     auxiliary (いる, ある, いく, おく, しまう, いい). It is a statement of
#     possibility. IPADIC's 非自立 asserts the word IS bound in this position,
#     which is a different claim and one UniDic does not make here. Mapping them
#     together would make agglutinate.py fold 行け into whatever preceded it —
#     行けなかった tags as 動詞-非自立可能 + 助動詞 + 助動詞. So it maps to 自立
#     and the merge happens (or does not) through the 助動詞 rules instead.
#
#   形状詞 -> 名詞-形容動詞語幹.  Close but not identical: UniDic promoted
#     na-adjectives to a top-level class, IPADIC kept them as a noun subtype.
_UNIDIC_TO_IPADIC: dict[tuple[str, str], tuple[str, str]] = {
    ("名詞", "普通名詞"): ("名詞", "一般"),
    ("名詞", "固有名詞"): ("名詞", "固有名詞"),
    ("名詞", "数詞"): ("名詞", "数"),
    ("名詞", "助動詞語幹"): ("名詞", "非自立"),
    ("代名詞", ""): ("名詞", "代名詞"),
    ("形状詞", "一般"): ("名詞", "形容動詞語幹"),
    ("形状詞", "タリ"): ("名詞", "形容動詞語幹"),
    ("形状詞", "助動詞語幹"): ("名詞", "形容動詞語幹"),
    ("接尾辞", "名詞的"): ("名詞", "接尾"),
    ("接尾辞", "形状詞的"): ("名詞", "接尾"),
    ("接尾辞", "動詞的"): ("動詞", "接尾"),
    ("接尾辞", "形容詞的"): ("形容詞", "接尾"),
    ("接頭辞", ""): ("接頭詞", "名詞接続"),
    ("動詞", "一般"): ("動詞", "自立"),
    ("動詞", "非自立可能"): ("動詞", "自立"),
    ("形容詞", "一般"): ("形容詞", "自立"),
    ("形容詞", "非自立可能"): ("形容詞", "自立"),
    ("助動詞", ""): ("助動詞", ""),
    ("副詞", ""): ("副詞", "一般"),
    ("感動詞", "一般"): ("感動詞", ""),
    ("感動詞", "フィラー"): ("フィラー", ""),
    ("補助記号", "句点"): ("記号", "句点"),
    ("補助記号", "読点"): ("記号", "読点"),
}
# 助詞 subtypes carry the same names in both dictionaries, and agglutination
# keys on 終助詞 and 接続助詞 specifically, so they pass through untouched.


def _unidic_pos(pos1: str, pos2: str) -> tuple[str, str]:
    p2 = "" if pos2 == "*" else pos2
    mapped = _UNIDIC_TO_IPADIC.get((pos1, p2))
    if mapped is None and p2:
        mapped = _UNIDIC_TO_IPADIC.get((pos1, ""))
    pos, sub = mapped or (pos1, p2)
    # Janome passes IPADIC's literal "*" through for an absent subtype. Match it
    # rather than normalising to "": the two tokenizers have to differ *only*
    # where they genuinely disagree, or the comparison drowns in noise — an
    # empty-vs-"*" mismatch would fire on every 助動詞 in the corpus.
    return pos, sub or "*"


def _unidic_lemma(lemma: str, surface: str) -> str:
    """語彙素 minus its 細分類 suffix.

    UniDic disambiguates homographs and marks loanword origins inside the lemma
    itself — コーヒー is stored as "コーヒー-coffee", 行く as "行く" but 開く as
    "開く-ヒラク". Everything downstream treats base_form as a dictionary form
    to look a gloss up on, so the annotation has to come off.
    """
    if not lemma or lemma == "*":
        return surface
    return lemma.split("-", 1)[0] or surface


class FugashiTokenizer:
    """UniDic via fugashi (MeCab), emitting IPADIC-shaped tags.

    Kept deliberately parallel to JanomeTokenizer rather than sharing code with
    it: the point of this class is to be swappable and removable, and the moment
    the two share a base class the swap stops being one variable.
    """

    #: UniDic's 29 feature columns, in order. Named here rather than indexed
    #: inline so a dictionary that reorders them fails loudly instead of quietly
    #: reading readings out of the accent column.
    FIELDS = (
        "pos1 pos2 pos3 pos4 cType cForm lForm lemma orth pron orthBase "
        "pronBase goshu iType iForm fType fForm iConType fConType type kana "
        "kanaBase form formBase aType aConType aModType lid lemma_id"
    ).split()

    def __init__(self, dicdir: str = ""):
        import fugashi

        path = resolve_unidic_dir(dicdir)
        if path is None:
            raise RuntimeError(
                "no UniDic found. Set [nlp] unidic_dir, or unpack a NINJAL "
                f"release into {UNIDIC_HOME}/unidic-cwj-<version>/"
            )
        # MeCab insists on a resource file and looks for it at a compiled-in
        # prefix that Homebrew's layout does not have, so it dies with
        # "no such file or directory: /usr/local/etc/mecabrc" before it ever
        # reads -d. An empty file next to the dictionary satisfies it; every
        # real setting we care about is passed on the command line anyway.
        rc = path / "mecabrc"
        if not rc.exists():
            rc.touch()
        self.dicdir = path
        self._t = fugashi.GenericTagger(f"-r {rc} -d {path}")

    def tokenize(self, text: str) -> list[Token]:
        out = []
        for node in self._t(text):
            f = dict(zip(self.FIELDS, list(node.feature)))
            pos, pos1 = _unidic_pos(f.get("pos1", ""), f.get("pos2", ""))
            kana = f.get("kana", "")
            out.append(fix_reading(
                Token(
                    surface=node.surface,
                    base_form=_unidic_lemma(f.get("lemma", ""), node.surface),
                    # 仮名形出現形 — the kana of the form as written, which is
                    # what IPADIC's 読み is too. Not 発音形 (pron), which already
                    # applies は->ワ and would double up with romaji.py's own
                    # Hepburn particle overrides.
                    reading="" if kana in ("", "*") else kana,
                    pos=pos,
                    pos1=pos1,
                )
            ))
        return out


def build_tokenizer(cfg: NlpCfg):
    if cfg.tokenizer == "unidic":
        # No silent fallback. Choosing unidic is an experiment, and an
        # experiment that quietly runs the control instead is worse than one
        # that fails.
        return FugashiTokenizer(getattr(cfg, "unidic_dir", ""))
    if cfg.tokenizer == "mecab":
        try:
            return MeCabTokenizer()
        except Exception:
            pass  # missing dict / compiler -> fallback to Janome
    return JanomeTokenizer()
