"""Online translation provider (Google Translate / DeepL via deep-translator)."""
from __future__ import annotations

import threading
from deep_translator import DeeplTranslator, GoogleTranslator
from tenacity import retry, stop_after_attempt, wait_exponential_jitter

from tsutawaru.config import TranslateCfg
from tsutawaru.logbus import get_logger
from tsutawaru.translate.base import Translator

log = get_logger(__name__)

# deep-translator GETs translate.google.com/m with no headers (google.py:67
# forwards only `proxies`), so it announces itself as "python-requests/2.x" to
# an endpoint that blocks non-browser agents. The block is intermittent and
# total when it lands: measured raw success on one fixed 56-line corpus was 0%,
# 23% and 71% on three consecutive days, while the same corpus with a browser
# User-Agent scored 100% in every one of those states.
#
# There is no supported hook for request headers, so rather than vendor their
# HTML parsing or patch `requests` process-wide, we rebind the `requests` name
# that deep_translator.google resolves against. Nothing else in the process
# sees the shim.
_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)

_patch_lock = threading.Lock()


class _BrowserUA:
    """Stands in for `requests` inside deep_translator.google, adding a UA."""

    def __init__(self, real):
        self._real = real

    def get(self, *args, **kwargs):
        kwargs.setdefault("headers", {"User-Agent": _UA})
        return self._real.get(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._real, name)


def _install_browser_ua() -> None:
    """Idempotent; safe to call from any thread."""
    with _patch_lock:
        from deep_translator import google as _g

        if not isinstance(_g.requests, _BrowserUA):
            _g.requests = _BrowserUA(_g.requests)
            log.debug("google: browser User-Agent shim installed")


class OnlineTranslator(Translator):
    def __init__(self, cfg: TranslateCfg):
        if cfg.provider == "deepl" and cfg.deepl_api_key:
            self.name = "deepl"
            self._mk = lambda: DeeplTranslator(
                api_key=cfg.deepl_api_key, source="ja", target="en"
            )
        else:
            # Named for what it *is*, not what was asked for: 'deepl' without a
            # key silently lands here, and the gloss cache must not label those
            # results 'deepl'.
            self.name = "google"
            _install_browser_ua()
            self._mk = lambda: GoogleTranslator(source="ja", target="en")
        self._local = threading.local()

    @property
    def t(self):
        if not hasattr(self._local, "t"):
            self._local.t = self._mk()
        return self._local.t

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential_jitter(initial=0.2, max=2.0),
        reraise=True,
    )
    def sentence(self, text: str) -> str:
        return self.t.translate(text)

    def batch(self, texts: list[str]) -> list[str]:
        """One round-trip for N tokens."""
        if not texts:
            return []
        try:
            return self.t.translate_batch(texts)
        except Exception:
            return [self.sentence(x) for x in texts]
