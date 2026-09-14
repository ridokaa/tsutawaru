"""Online translation provider (Google Translate / DeepL via deep-translator)."""
from __future__ import annotations

import threading
from deep_translator import DeeplTranslator, GoogleTranslator
from deep_translator.exceptions import RequestError, TooManyRequests
from tenacity import (retry, retry_if_not_exception_type, stop_after_attempt,
                      wait_exponential_jitter)

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
        self._warned = False

    @property
    def t(self):
        if not hasattr(self._local, "t"):
            self._local.t = self._mk()
        return self._local.t

    # Retry timeouts and dropped connections, never a refusal. A 429 is the
    # CAPTCHA interstitial ("unusual traffic from your computer network"), which
    # no second attempt fixes — it is keyed on the IP and caused by volume, so
    # retrying it triples the traffic that earned the block in the first place.
    @retry(
        retry=retry_if_not_exception_type((TooManyRequests, RequestError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential_jitter(initial=0.2, max=2.0),
        reraise=True,
    )
    def _translate(self, text: str) -> str:
        return self.t.translate(text)

    def sentence(self, text: str, remember: bool = True) -> str:
        # `remember` is meaningless here: each request is independent and the
        # backend keeps nothing between them. Accepted so the two providers
        # stay interchangeable.
        try:
            return self._translate(text)
        except TooManyRequests:
            if not self._warned:
                self._warned = True
                log.warning(
                    "%s is refusing requests from this IP (429 CAPTCHA). It "
                    "clears on its own in a few hours; until then run with "
                    "--provider local for offline translation.", self.name)
            raise

    def batch(self, texts: list[str], remember: bool = True) -> list[str]:
        """N round-trips, not one: neither backend has a batch endpoint.

        `deep_translator.translate_batch` is a for-loop over `translate`
        (base.py:171). Per item rather than one try/except over the list,
        because that shape discards the items that already succeeded and then
        re-requests every one of them through the retrying path.
        """
        out = []
        for x in texts:
            try:
                out.append(self.sentence(x))
            except Exception:
                out.append("")  # `_glosses` renders this as "?"
        return out


if __name__ == "__main__":  # self-check: python -m tsutawaru.translate.online
    # The regression this guards: a refusal costing more requests than a
    # success. Counts calls rather than asserting on timing, because the whole
    # defect is a count.
    class _Stub:
        def __init__(self, exc):
            self.exc, self.calls = exc, 0

        def translate(self, text: str) -> str:
            self.calls += 1
            raise self.exc()

    def _mk(exc) -> tuple[OnlineTranslator, _Stub]:
        t = OnlineTranslator.__new__(OnlineTranslator)
        t.name, t._warned = "google", False
        stub = _Stub(exc)
        t._local = threading.local()
        t._local.t = stub
        return t, stub

    t, stub = _mk(TooManyRequests)
    try:
        t.sentence("こんにちは")
    except TooManyRequests:
        pass
    assert stub.calls == 1, f"429 retried {stub.calls}x — that is the bug"

    t, stub = _mk(TooManyRequests)
    assert t.batch(["a", "b", "c"]) == ["", "", ""]
    assert stub.calls == 3, f"3 tokens cost {stub.calls} requests"

    # Narrowed, not removed: a transient fault still gets its three attempts.
    t, stub = _mk(TimeoutError)
    try:
        t.sentence("x")
    except TimeoutError:
        pass
    assert stub.calls == 3, f"timeout attempted {stub.calls}x, expected 3"

    print("online self-check OK")
