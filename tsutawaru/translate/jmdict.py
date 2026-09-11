"""Offline JMdict gloss lookup for the breakdown tier.

The gloss lane asks a question no translation model answers well: what does the
*dictionary form* 食べる mean, on its own, with no sentence around it. An MT model
handed a bare word returns a sentence — it invents a subject and a tense because
that is what MT models do. A dictionary returns "to eat".

So the breakdown tier reads JMdict (the EDRDG Japanese-English dictionary,
CC BY-SA 4.0) instead of the network. Lookups are a single indexed SQLite hit —
no GPU, no round-trip, no rate limit — which also means the gloss lane stops
competing with the sentence lane for anything at all.

Build the database once:

    python -m tsutawaru.translate.jmdict

Without it `gloss()` returns None for everything and the pool falls back to the
online backend exactly as before, so a missing database degrades rather than
breaks.
"""
from __future__ import annotations

import json
import sqlite3
import tarfile
import threading
import urllib.request
from pathlib import Path
from typing import Optional

from platformdirs import user_cache_dir

from tsutawaru.logbus import get_logger
from tsutawaru.models import Token

log = get_logger(__name__)

RELEASES = "https://api.github.com/repos/scriptin/jmdict-simplified/releases/latest"

# IPADIC major POS -> our normalised class. Same coarse buckets the JMdict tags
# are folded into below, so a 動詞 token can only ever match a verb sense.
_IPADIC = {
    "名詞": "n",
    "動詞": "v",
    "形容詞": "adj",
    "副詞": "adv",
    "連体詞": "adj",
    "接続詞": "conj",
    "感動詞": "int",
    "助詞": "prt",
    "助動詞": "aux",
    "接頭詞": "pref",
    "接尾辞": "suf",
}

# JMdict part-of-speech tags -> the same classes. Verb tags are open-ended
# (v1, v5k, v5aru, vk, vs-i, …) so they are matched by prefix, everything else
# by exact tag.
_JMDICT = {
    "n": "n", "pn": "n", "n-pref": "n", "n-suf": "n", "num": "n",
    "adj-i": "adj", "adj-na": "adj", "adj-no": "adj", "adj-pn": "adj",
    "adj-t": "adj", "adj-f": "adj", "adj-ix": "adj",
    "adv": "adv", "adv-to": "adv",
    "conj": "conj", "int": "int", "prt": "prt", "aux-v": "aux",
    "aux": "aux", "aux-adj": "aux", "cop": "aux",
    "pref": "pref", "suf": "suf", "ctr": "suf", "exp": "exp",
}

# Senses tagged like this are real but wrong for a live conversation aid —
# "archaic" beating the everyday meaning is how a learner tool teaches nonsense.
# Penalised rather than dropped: for a rare word it may be the only sense there is.
_DEMOTE = {"arch", "obs", "obsc", "rare", "derog", "vulg", "sl", "X"}


# Godan potential forms (聞き取れる, 会える) are productive, so JMdict lists almost
# none of them — the tokenizer hands us one as its own dictionary form and the
# lookup misses. Undo the conjugation by walking the final kana back from the
# e-row to the u-row: 会える -> 会う, 聞き取れる -> 聞き取る.
_POTENTIAL = str.maketrans("えけせてねへめれげぜでべ", "うくすつぬふむるぐずづぶ")


def _depotential(base: str) -> Optional[str]:
    if len(base) > 2 and base.endswith("る") and base[-2] in "えけせてねへめれげぜでべ":
        return base[:-2] + base[-2].translate(_POTENTIAL)
    return None


def _pos_of(tok: Token) -> Optional[str]:
    """The token's word class. `pos1` decides where IPADIC's major POS is a lie:
    a na-adjective (静か, 大丈夫) is filed under 名詞 with 形容動詞語幹 as its subtype."""
    if tok.pos1 == "形容動詞語幹":
        return "adj"
    return _IPADIC.get(tok.pos)


def _pos_class(tags: list[str]) -> Optional[str]:
    for t in tags:
        if t in _JMDICT:
            return _JMDICT[t]
        if t.startswith("v"):  # v1, v5r, vk, vs-i, vt/vi never appear alone
            return "v"
    return None


def db_path() -> Path:
    return Path(user_cache_dir("tsutawaru")) / "jmdict.db"


def _latest_asset() -> tuple[str, str]:
    """(url, name) of the newest full English jmdict-simplified JSON tarball."""
    with urllib.request.urlopen(RELEASES, timeout=30) as r:
        rel = json.load(r)
    for a in rel["assets"]:
        n = a["name"]
        if n.startswith("jmdict-eng-") and n.endswith(".json.tgz") and "common" not in n:
            return a["browser_download_url"], n
    raise RuntimeError(f"no jmdict-eng .json.tgz in release {rel.get('tag_name')!r}")


def build(dest: Optional[Path] = None) -> Path:
    """Download the newest JMdict release and index it. Idempotent-ish: replaces.

    The release file is one JSON object per line inside a `"words": [ … ]` array,
    so it is parsed a line at a time. Reading it with json.load() instead costs
    1.7 GB of peak RSS for a 112 MB file — on a 16 GB machine that is a real risk
    of evicting the ASR weights, to save about ten lines of code.
    """
    dest = dest or db_path()
    dest.parent.mkdir(parents=True, exist_ok=True)
    url, name = _latest_asset()
    log.info("[jmdict] downloading %s", name)

    tmp = dest.with_suffix(".tgz")
    urllib.request.urlretrieve(url, tmp)

    tmp_db = dest.with_suffix(".building")
    tmp_db.unlink(missing_ok=True)
    db = sqlite3.connect(str(tmp_db))
    db.execute("PRAGMA journal_mode=OFF")
    db.execute("PRAGMA synchronous=OFF")
    db.execute("CREATE TABLE jm(k TEXT, pos TEXT, gloss TEXT, prio INT)")

    rows, entries = [], 0
    with tarfile.open(tmp) as tf:
        member = next(m for m in tf.getmembers() if m.name.endswith(".json"))
        fh = tf.extractfile(member)
        assert fh is not None
        decoder = json.JSONDecoder()
        for raw in fh:
            line = raw.strip()
            if not line.startswith(b'{"id"'):
                continue  # header keys and the array's brackets
            # raw_decode rather than loads: every entry line carries a trailing
            # comma, and the final one closes the array and the root object too
            # (`…}]}`). Stopping at the first complete object handles all three.
            e, _ = decoder.raw_decode(line.decode())
            entries += 1

            keys = [k["text"] for k in e["kanji"] if "sK" not in k["tags"]]
            keys += [k["text"] for k in e["kana"] if "sk" not in k["tags"]]
            common = any(k.get("common") for k in e["kanji"] + e["kana"])

            seen: set[str] = set()
            for i, s in enumerate(e["sense"]):
                cls = _pos_class(s["partOfSpeech"])
                if cls is None or cls in seen:
                    continue  # first sense of each class wins; the rest are noise here
                seen.add(cls)
                texts = [g["text"] for g in s["gloss"] if g["lang"] == "eng"][:2]
                if not texts:
                    continue
                gloss = " / ".join(texts)
                prio = i + (0 if common else 20) + (50 if _DEMOTE & set(s["misc"]) else 0)
                rows.extend((k, cls, gloss, prio) for k in keys)

            if len(rows) >= 50_000:
                db.executemany("INSERT INTO jm VALUES(?,?,?,?)", rows)
                rows.clear()

    db.executemany("INSERT INTO jm VALUES(?,?,?,?)", rows)
    db.execute("CREATE INDEX jm_k ON jm(k, prio)")
    db.commit()
    db.close()
    tmp.unlink()
    tmp_db.replace(dest)

    size = dest.stat().st_size / 1e6
    log.info("[jmdict] %d entries indexed -> %s (%.0f MB)", entries, dest, size)
    return dest


class _Lookup:
    """Thread-local read-only connections; the gloss lane runs N workers."""

    def __init__(self) -> None:
        self._local = threading.local()
        self._warned = False

    @property
    def db(self) -> Optional[sqlite3.Connection]:
        conn = getattr(self._local, "db", None)
        if conn is None:
            p = db_path()
            if not p.exists():
                if not self._warned:
                    self._warned = True
                    log.warning(
                        "[jmdict] %s not built — glosses fall back to the online "
                        "backend. Build it with: python -m tsutawaru.translate.jmdict", p
                    )
                return None
            conn = sqlite3.connect(f"file:{p}?mode=ro", uri=True, check_same_thread=False)
            self._local.db = conn
        return conn

    def gloss(self, tok: Token) -> Optional[str]:
        db = self.db
        if db is None:
            return None
        # A *preference*, not a filter. IPADIC and JMdict disagree about word
        # classes often enough that filtering hard is worse than not filtering
        # at all: 大丈夫 is 名詞 to IPADIC and adj-na to JMdict, so `WHERE pos='n'`
        # returns the archaic "great man" and hides "safe / secure" entirely.
        # The penalty is wide enough to beat sense order, narrow enough that a
        # common sense in the "wrong" class still outranks a rare one in the right.
        cls = _pos_of(tok) or ""
        q = "SELECT gloss FROM jm WHERE k=? ORDER BY prio + (pos<>?)*4 LIMIT 1"
        row = db.execute(q, (tok.base_form, cls)).fetchone()
        if row:
            return row[0]
        # Only after a miss, so a verb JMdict does list can never be mangled into
        # a different word — 教える is found directly and never becomes 教う.
        stem = _depotential(tok.base_form) if cls == "v" else None
        if stem:
            row = db.execute(q, (stem, cls)).fetchone()
            if row:
                return " / ".join(
                    "can " + g.removeprefix("to ") for g in row[0].split(" / ")
                )
        return None


_lookup = _Lookup()


def gloss(tok: Token) -> Optional[str]:
    """Dictionary gloss for a token's base form, or None if unknown/unbuilt."""
    return _lookup.gloss(tok)


if __name__ == "__main__":  # pragma: no cover - one-off setup command
    import logging

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    build()
