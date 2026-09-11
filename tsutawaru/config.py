"""Typed configuration: defaults -> config.toml -> CLI overrides.

Every component constructor takes its own section object (`cfg.stt`, `cfg.vad`,
…) rather than the whole config, so a unit test can build one section without
constructing the world.

Unknown keys are warned about rather than ignored — a silently-typo'd key that
does nothing is worse than a noisy one.
"""
from __future__ import annotations

import sys
import tomllib
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from tsutawaru.logbus import get_logger

log = get_logger(__name__)

DEFAULT_CONFIG_NAMES = ("config.toml", "config.example.toml")


def _installed(module: str) -> bool:
    """True if `module` can be imported, without importing it.

    importlib.util.find_spec keeps validate() free of side effects — actually
    importing proctap would spawn nothing, but it does pull numpy/scipy and pyobjc
    into a process that may only be running --help.
    """
    import importlib.util

    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):  # pragma: no cover - malformed install
        return False


# Core Audio process taps were introduced in macOS 14.4. Below that the API does
# not exist at all, so there is nothing to fall back *from* — it is a hard gate,
# not a preference.
PROCTAP_MIN_MACOS = (14, 4)
PROCTAP_DEPS = (("proctap", "proc-tap"), ("psutil", "psutil"))


def proctap_unavailable() -> str:
    """Why per-process capture cannot run here, or "" if it can.

    A single answer for both callers below, so the auto-resolver and validate()
    can never disagree about what "available" means.
    """
    if sys.platform != "darwin":
        return f"per-process capture needs macOS; this is {sys.platform}"
    import platform

    ver = platform.mac_ver()[0]
    try:
        parts = tuple(int(p) for p in ver.split(".")[:2])
    except ValueError:  # pragma: no cover - unparseable mac_ver
        parts = ()
    if parts and parts < PROCTAP_MIN_MACOS:
        want = ".".join(str(p) for p in PROCTAP_MIN_MACOS)
        return f"Core Audio process taps need macOS {want}+; this is {ver}"
    missing = [pip for mod, pip in PROCTAP_DEPS if not _installed(mod)]
    if missing:
        return (
            f"missing the {', '.join(repr(m) for m in missing)} package(s) — "
            f"pip install {' '.join(missing)}"
        )
    return ""


def resolve_audio_backend(backend: str) -> str:
    """Turn "auto" into a concrete backend; pass anything else through.

    "auto" prefers per-process capture, because tapping one application excludes
    every other sound on the machine and signal-to-noise is what actually decides
    whether a line transcribes correctly — a game running alongside a voice call
    is fused into the system mix permanently, and no model can unfuse it.

    It degrades to "device" rather than failing when the platform or the optional
    packages cannot support a tap. An explicit `backend = "process"` does not
    degrade: asking for something specific and silently getting something else is
    how you spend an evening debugging the wrong backend.
    """
    if backend != "auto":
        return backend
    why = proctap_unavailable()
    if why:
        log.info("[audio] backend 'auto' -> 'device' (%s)", why)
        return "device"
    return "process"


@dataclass
class AudioCfg:
    device: str | int = "auto"  # "auto" | index | substring of device name
    samplerate: int = 16000  # target rate; capture runs at device-native rate
    blocksize_ms: int = 20
    channels: int = 1
    loopback: bool = False  # Windows only: PyAudioWPatch loopback backend
    # "auto" resolves to per-process capture on macOS 14.4+ with the ProcTap
    # packages present, and to the virtual-loopback device path everywhere else.
    # Read it through `effective_backend`, never directly — the raw value can be
    # "auto", which no downstream branch knows how to handle.
    backend: str = "auto"  # "auto" | "device" (virtual loopback) | "process" (ProcTap)
    source: str = "discord"  # backend="process" only: which app to tap

    @property
    def effective_backend(self) -> str:
        """The concrete backend this config will actually capture with."""
        return resolve_audio_backend(self.backend)


@dataclass
class VadCfg:
    backend: str = "silero"  # "silero" | "webrtc"
    threshold: float = 0.55
    aggressiveness: int = 2  # webrtc 0..3
    preroll_ms: int = 320  # rounded to whole 32 ms windows
    hangover_ms: int = 448  # rounded to whole 32 ms windows
    min_utterance_ms: int = 350
    max_utterance_ms: int = 12000


@dataclass
class SttCfg:
    backend: str = "auto"  # "auto" | "mlx" | "faster"
    model: str = "kotoba"
    language: str = "ja"
    lang_mode: str = "pinned"  # "pinned" | "detect"
    beam_size: int = 1
    compute_type: str = "auto"
    no_speech_thresh: float = 0.6
    logprob_floor: float = -1.0
    initial_prompt: str = ""
    # A second model over the same audio, shown beside the first. Empty (the
    # default) is off: it loads a second set of weights and both engines then
    # contend for one GPU, so it is a testing tool, not a thing to leave on.
    compare_model: str = ""


@dataclass
class NlpCfg:
    tokenizer: str = "janome"  # "janome" | "mecab" | "unidic"
    # backend="unidic" only. Empty = newest unidic-* under ~/.local/share/tsutawaru
    # (override that root with TSUTAWARU_UNIDIC_HOME).
    unidic_dir: str = ""
    drop_pos: list[str] = field(
        default_factory=lambda: ["記号", "補助記号", "空白", "フィラー", "その他"]
    )
    agglutinate: bool = True
    max_tokens: int = 24
    particle_romaji: str = "hepburn"  # "hepburn" (は->wa) | "literal" (は->ha)


@dataclass
class TranslateCfg:
    provider: str = "google"  # "google" | "deepl" | "local" | "none"
    deepl_api_key: str = ""
    # provider="local" only: the MLX repo for the sentence lane. 1.4b is the
    # quality pick; swap in CAT-Translate-0.8b-mlx-q4 on a tighter memory budget.
    local_model: str = "hotchpotch/CAT-Translate-1.4b-mlx-q4"
    gloss_workers: int = 3  # sentence translation gets its own dedicated worker
    sentence_timeout_s: float = 4.0
    token_timeout_s: float = 3.0
    lru_size: int = 20000
    persist_cache: bool = True


@dataclass
class UiCfg:
    sink: str = "window"  # "window" | "console" | "overlay"
    ws_port: int = 8765
    max_lines: int = 200
    show_romaji: bool = True
    show_breakdown: bool = True
    plain: bool = False
    log_file: str = ""


@dataclass
class Config:
    audio: AudioCfg = field(default_factory=AudioCfg)
    vad: VadCfg = field(default_factory=VadCfg)
    stt: SttCfg = field(default_factory=SttCfg)
    nlp: NlpCfg = field(default_factory=NlpCfg)
    translate: TranslateCfg = field(default_factory=TranslateCfg)
    ui: UiCfg = field(default_factory=UiCfg)

    source_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("source_path", None)
        return d

    def validate(self) -> list[str]:
        """Return a list of human-readable problems. Empty means OK.

        Only checks that cannot be expressed as a simple choice/type constraint
        live here — the rest are handled during parsing.
        """
        errs: list[str] = []
        v = self.vad
        if v.min_utterance_ms >= v.max_utterance_ms:
            errs.append(
                f"[vad] min_utterance_ms ({v.min_utterance_ms}) must be below "
                f"max_utterance_ms ({v.max_utterance_ms})"
            )
        if not 0.0 < v.threshold < 1.0:
            errs.append(f"[vad] threshold must be between 0 and 1, got {v.threshold}")
        if v.hangover_ms < 32:
            errs.append("[vad] hangover_ms below one 32 ms window — utterances will not close")
        if self.audio.blocksize_ms <= 0:
            errs.append("[audio] blocksize_ms must be positive")
        if self.nlp.max_tokens < 1:
            errs.append("[nlp] max_tokens must be at least 1")
        if self.translate.gloss_workers < 1:
            errs.append("[translate] gloss_workers must be at least 1")
        if self.translate.provider == "deepl" and not self.translate.deepl_api_key:
            errs.append("[translate] provider is 'deepl' but deepl_api_key is empty")
        if self.translate.provider == "local" and sys.platform != "darwin":
            # Same rule as the qwen3 STT models: an MLX-only path names its
            # requirement rather than quietly running something else.
            errs.append(
                f"[translate] provider 'local' is MLX-only (Apple Silicon); "
                f"this is {sys.platform} — use 'google' or 'deepl'"
            )
        if self.audio.backend == "process":
            # Checked here rather than left to stage construction: an unusable
            # backend would otherwise surface as "audio capture is unavailable —
            # cannot start", which names neither the cause nor the fix.
            #
            # Deliberately keyed on the *raw* value, not `effective_backend`.
            # "auto" resolving away from a tap is a normal outcome and reports
            # itself in the log; "process" spelled out is a request that has to be
            # honoured or refused, never quietly downgraded.
            why = proctap_unavailable()
            if why:
                errs.append(f"[audio] backend 'process' is unavailable: {why}")
        if self.stt.compare_model and self.stt.compare_model == self.stt.model:
            # Silently useless: two identical columns that look exactly like
            # a working comparison.
            errs.append(
                f"[stt] compare_model is the same as model ({self.stt.model!r}) "
                "— pick a different model to compare against, or leave it empty"
            )
        if self.vad.backend == "webrtc" and sys.version_info >= (3, 14):
            errs.append(
                "[vad] backend 'webrtc' is unavailable on Python 3.14 — "
                "webrtcvad-wheels publishes no cp314 wheel. Use backend='silero', "
                "or build on Python 3.11 (see plan §3.1)."
            )
        return errs


_SECTIONS = {
    "audio": AudioCfg,
    "vad": VadCfg,
    "stt": SttCfg,
    "nlp": NlpCfg,
    "translate": TranslateCfg,
    "ui": UiCfg,
}

def _source_keys() -> set[str]:
    """Valid [audio].source values, taken from the registry rather than duplicated.

    Imported lazily so a bad/absent optional dependency can never break config
    loading, which every code path including --help goes through.
    """
    try:
        from tsutawaru.audio.sources import SOURCES

        return set(SOURCES)
    except Exception:  # pragma: no cover - defensive
        return {"discord", "youtube"}


_CHOICES = {
    ("audio", "backend"): {"auto", "device", "process"},
    ("audio", "source"): _source_keys(),
    ("vad", "backend"): {"silero", "webrtc"},
    ("stt", "backend"): {"auto", "mlx", "faster"},
    # "qwen3"/"qwen3-small" are Qwen3-ASR via MLX and are Apple-Silicon only;
    # stt/factory.py raises rather than falling back, so selecting one on
    # other hardware fails loudly instead of running a different model.
    ("stt", "model"): {"tiny", "base", "small", "medium", "kotoba",
                       "qwen3", "qwen3-small"},
    ("stt", "lang_mode"): {"pinned", "detect"},
    ("nlp", "tokenizer"): {"janome", "mecab", "unidic"},
    ("nlp", "particle_romaji"): {"hepburn", "literal"},
    ("translate", "provider"): {"google", "deepl", "local", "none"},
    ("ui", "sink"): {"window", "console", "overlay", "both"},
}
# Same names as `model`, plus "" for off. Derived so the two cannot drift.
_CHOICES[("stt", "compare_model")] = _CHOICES[("stt", "model")] | {""}


def _coerce(want_type: Any, val: Any, where: str) -> Any:
    """Coerce TOML scalars to the dataclass field type where it is safe."""
    # `device` is intentionally str|int — pass through untouched.
    if want_type in (Any, "str | int", "int | str"):
        return val
    if want_type is bool:
        if isinstance(val, bool):
            return val
        raise ValueError(f"{where}: expected true/false, got {val!r}")
    if want_type is int and isinstance(val, bool):
        raise ValueError(f"{where}: expected an integer, got a boolean")
    if want_type is int:
        return int(val)
    if want_type is float:
        return float(val)
    if want_type is str:
        return str(val)
    return val


def _build_section(cls, data: dict, name: str):
    obj = cls()
    known = {f.name: f for f in fields(cls)}
    for key, val in data.items():
        if key not in known:
            log.warning("config [%s]: unknown key %r — ignored", name, key)
            continue
        ftype = known[key].type
        try:
            if key == "device":
                setattr(obj, key, val)
            elif isinstance(getattr(obj, key), list):
                setattr(obj, key, list(val))
            else:
                setattr(obj, key, _coerce(_basetype(ftype), val, f"[{name}].{key}"))
        except (TypeError, ValueError) as e:
            log.warning("config [%s].%s: %s — using default", name, key, e)
        choices = _CHOICES.get((name, key))
        if choices and getattr(obj, key) not in choices:
            log.warning(
                "config [%s].%s = %r is not one of %s — using default %r",
                name, key, getattr(obj, key), sorted(choices), getattr(cls(), key),
            )
            setattr(obj, key, getattr(cls(), key))
    return obj


def _basetype(ftype: Any) -> Any:
    """Dataclass field types arrive as strings under `from __future__ import annotations`."""
    if isinstance(ftype, str):
        return {
            "int": int, "float": float, "bool": bool, "str": str,
        }.get(ftype.strip(), Any)
    return ftype


def find_config(explicit: str | None = None) -> Path | None:
    if explicit:
        p = Path(explicit).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"config file not found: {p}")
        return p
    for name in DEFAULT_CONFIG_NAMES:
        p = Path.cwd() / name
        if p.is_file():
            return p
    return None


def load(path: str | None = None, overrides: dict[str, dict[str, Any]] | None = None) -> Config:
    """Load config: dataclass defaults <- TOML file <- CLI overrides."""
    cfg_path = find_config(path)
    raw: dict[str, Any] = {}
    if cfg_path is not None:
        with cfg_path.open("rb") as fh:
            raw = tomllib.load(fh)
        log.info("loaded config from %s", cfg_path)
    else:
        log.info("no config.toml found — using built-in defaults")

    for key in raw:
        if key not in _SECTIONS:
            log.warning("config: unknown section [%s] — ignored", key)

    cfg = Config(source_path=cfg_path)
    for name, cls in _SECTIONS.items():
        setattr(cfg, name, _build_section(cls, raw.get(name, {}) or {}, name))

    for name, kv in (overrides or {}).items():
        section = getattr(cfg, name, None)
        if section is None:
            continue
        for key, val in kv.items():
            if val is None:
                continue  # unset CLI flag — do not clobber the file value
            if not hasattr(section, key):
                log.warning("override %s.%s: no such setting", name, key)
                continue
            # CLI flags get the same choice check the file gets. Without it a
            # retired value (--provider marian) is accepted and then quietly
            # resolves to something else, so the run disagrees with the command
            # that started it. Keep the existing value rather than reverting to
            # the class default: a bad flag must not also discard config.toml.
            choices = _CHOICES.get((name, key))
            if choices and val not in choices:
                log.warning(
                    "--%s %r is not one of %s — keeping %r",
                    key.replace("_", "-"), val, sorted(choices), getattr(section, key),
                )
                continue
            setattr(section, key, val)

    return cfg
