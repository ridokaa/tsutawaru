"""Running a second ASR model beside the first, on the same audio.

The lane exists to answer "is qwen3 actually better than kotoba on my calls",
and the only way it can answer that is by changing nothing about the thing it
is measuring. Most of these tests are about that constraint rather than about
the feature: the primary transcript, its latency, and its completion signal all
have to be bit-for-bit what they would have been with comparison switched off.

The other half is that it must stay off unless asked. It loads a second set of
weights and both engines then contend for one GPU, so an accidental default
would be an expensive and near-invisible regression.
"""
from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from tsutawaru.config import Config, load
from tsutawaru.models import Segment, Token, Utterance
from tsutawaru.pipeline.queues import compare_q, ui_q


class _Engine:
    """A stand-in ASR model that returns whatever it was told to."""

    def __init__(self, text="カフェテリア", fail_warmup=False):
        self.text, self.fail_warmup = text, fail_warmup
        self.seen = 0

    def warmup(self):
        if self.fail_warmup:
            raise RuntimeError("no Stream(gpu, 1) in current thread")

    def transcribe(self, audio):
        self.seen += 1
        return type("R", (), {"text": self.text, "language": "ja"})()


def _orch(compare_model="kotoba", engine=None):
    from tsutawaru.pipeline.orchestrator import Orchestrator

    cfg = load(None, {"stt": {"model": "qwen3", "compare_model": compare_model}})
    orch = Orchestrator(cfg)
    orch.stages.stt_compare = engine or _Engine()
    return orch


def _drain(q):
    out = []
    while True:
        try:
            out.append(q.get_nowait())
        except Exception:
            return out


@pytest.fixture(autouse=True)
def _clean():
    _drain(compare_q)
    _drain(ui_q)
    yield
    _drain(compare_q)
    _drain(ui_q)


# --------------------------------------------------------------- off by default


def test_comparison_is_off_unless_asked():
    """A second model is a testing tool, not a default: it doubles the GPU load."""
    assert Config().stt.compare_model == ""
    assert not Config().validate()


def test_an_ordinary_segment_carries_no_comparison_fields():
    """Absence in the JSONL is the honest signal that nothing was compared."""
    d = Segment.new(stream="main", original="海が怖い").to_dict()
    assert "alt" not in d


def test_a_compared_segment_records_both_readings():
    seg = Segment.new(stream="main", original="カフェテリアでブランを発見")
    seg.model = "qwen3"
    seg.alt_model, seg.alt_original = "kotoba", "カベテリアでブランを発見"
    seg.alt_romaji = "kaberatia de buran wo hakken"
    seg.alt_english = "Found Bran at Caveteria"
    d = seg.to_dict()
    # Both engines are named, so a graded JSONL says which model produced which
    # column without needing the session metadata alongside it.
    assert d["model"] == "qwen3"
    assert d["alt"] == {"model": "kotoba",
                        "jp": "カベテリアでブランを発見",
                        "romaji": "kaberatia de buran wo hakken",
                        "en": "Found Bran at Caveteria"}


def test_comparing_a_model_with_itself_is_rejected():
    """Twice the load, two identical columns, and it would look like it worked."""
    errs = load(None, {"stt": {"model": "qwen3", "compare_model": "qwen3"}}).validate()
    assert any("compare_model is the same as model" in e for e in errs)


def test_a_valid_pair_is_accepted():
    cfg = load(None, {"stt": {"model": "qwen3", "compare_model": "kotoba"}})
    assert cfg.stt.compare_model == "kotoba"
    assert not cfg.validate()


# ------------------------------------------------------------------- the worker


def _run_worker(orch, timeout=2.0):
    t = threading.Thread(target=orch._stt_compare_worker, daemon=True)
    t.start()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and compare_q.qsize():
        time.sleep(0.01)
    time.sleep(0.05)
    orch.stop_evt.set()
    t.join(timeout=1.0)
    return t


def test_the_second_transcript_is_attached_to_the_same_line():
    orch = _orch(engine=_Engine("カフェテリアでブランを発見"))
    seg = Segment.new(stream="main", original="カベテリアでブランを発見")
    utt = Utterance.new(audio=np.zeros(1600, dtype=np.float32), t_start=0.0, t_end=0.1)
    compare_q.put_latest((utt, seg))
    _run_worker(orch)

    assert seg.alt_model == "kotoba"
    assert seg.alt_original == "カフェテリアでブランを発見"


def test_the_comparison_never_edits_the_primary_transcript():
    """The measurement must not change the thing measured."""
    orch = _orch(engine=_Engine("まったく違う文"))
    seg = Segment.new(stream="main", original="カベテリアでブランを発見")
    seg.english = "Found Bran at Caveteria"
    seg.tokens = [Token(surface="発見", base_form="発見", gloss="discovery")]
    utt = Utterance.new(audio=np.zeros(1600, dtype=np.float32), t_start=0.0, t_end=0.1)
    compare_q.put_latest((utt, seg))
    _run_worker(orch)

    assert seg.original == "カベテリアでブランを発見"
    assert seg.english == "Found Bran at Caveteria"
    assert [t.gloss for t in seg.tokens] == ["discovery"]
    # Completion belongs to the gloss lane. A slow second model must not hold a
    # line open, or the study sheet's latency figures measure the diagnostic.
    assert seg.partial is True
    assert seg.t_complete == 0.0


def test_a_broken_comparison_model_does_not_stop_the_session():
    """Opposite of the primary lane, and deliberately so.

    A primary engine that cannot warm up stops the run, because nothing useful
    can follow. This lane produces nothing anyone joined a call for, so it
    disables itself and leaves the transcript running.
    """
    orch = _orch(engine=_Engine(fail_warmup=True))
    orch._stt_compare_worker()

    assert not orch.stop_evt.is_set(), "a diagnostic must not take down the run"
    assert orch.stages.stt_compare is None, "the broken lane should switch off"


def test_the_fanout_only_happens_when_a_comparison_is_configured():
    from tsutawaru.pipeline.orchestrator import Orchestrator

    orch = Orchestrator(load())
    assert orch.stages.stt_compare is None
    utt = Utterance.new(audio=np.zeros(1600, dtype=np.float32), t_start=0.0, t_end=0.1)
    res = type("R", (), {"text": "海が怖い", "avg_logprob": -0.2,
                         "no_speech_prob": 0.01, "language": "ja"})()
    orch._emit_segment(utt, res)
    assert compare_q.qsize() == 0


def test_the_fanout_passes_the_segment_so_the_pair_stays_together():
    orch = _orch()
    utt = Utterance.new(audio=np.zeros(1600, dtype=np.float32), t_start=0.0, t_end=0.1)
    res = type("R", (), {"text": "海が怖い", "avg_logprob": -0.2,
                         "no_speech_prob": 0.01, "language": "ja"})()
    orch._emit_segment(utt, res)

    assert compare_q.qsize() == 1
    got_utt, got_seg = compare_q.get_nowait()
    assert got_utt is utt
    assert got_seg.original == "海が怖い"


# ------------------------------------------------------------------- rendering


def _seg_with_alt():
    seg = Segment.new(stream="main", original="ディスコードで流すぐらいかな")
    seg.english = "I guess I'd just play it on Discord"
    seg.romaji = "disukoodo de nagasu gurai kana"
    seg.model = "qwen3"
    seg.alt_model = "kotoba"
    seg.alt_original = "ディスコートで流すぐらいかな"
    seg.alt_romaji = "disukooto de nagasu gurai kana"
    seg.alt_english = "I think it's easy to wash away with Discourt."
    return seg


def test_the_comparison_transcript_gets_a_reading_too():
    """Romaji from the same tokenizer path as the primary, not a shortcut.

    `reading()` and `annotate()` share `_reading_tokens`, so both columns apply
    the particle rule (は -> wa). Romanizing the comparison with a plainer
    converter would make the columns differ by romanization on lines where the
    models actually agreed.
    """
    from tsutawaru.config import NlpCfg
    from tsutawaru.nlp.pipeline import annotate, reading
    from tsutawaru.nlp.tokenizer import build_tokenizer

    cfg = NlpCfg()
    tok = build_tokenizer(cfg)
    seg = Segment.new(stream="main", original="海が怖い")
    annotate(seg, tok, cfg)

    assert reading("海が怖い", tok, cfg) == seg.romaji


def test_the_comparison_lane_romanizes_what_it_transcribed():
    orch = _orch(engine=_Engine("海が怖い"))
    seg = Segment.new(stream="main", original="膿が怖い")
    utt = Utterance.new(audio=np.zeros(1600, dtype=np.float32), t_start=0.0, t_end=0.1)
    compare_q.put_latest((utt, seg))
    _run_worker(orch, timeout=20.0)

    assert seg.alt_original == "海が怖い"
    assert seg.alt_romaji == "umi ga kowai"


def test_a_reading_failure_never_costs_the_transcript():
    """Cosmetic loss vs. losing the line it belongs to.

    (build_tokenizer falls back on an unknown name rather than raising, so the
    failure has to be injected at the tokenizer itself to be real.)
    """
    class _Broken:
        def tokenize(self, text):
            raise RuntimeError("tagger exploded")

    orch = _orch()
    orch._alt_tokenizer = _Broken()
    assert orch._alt_reading("海が怖い") == ""


def test_an_empty_comparison_transcript_needs_no_tokenizer():
    """The common case for a dropped line — must not pay to build one."""
    orch = _orch()
    assert orch._alt_reading("") == ""
    assert orch._alt_tokenizer is None


def test_the_console_block_shows_both_readings():
    from tsutawaru.ui.formatter import render_block

    out = render_block(_seg_with_alt(), Config().ui)
    assert "--- kotoba ---" in out
    assert "ディスコート" in out and "ディスコード" in out
    # The numbered tiers are the product; the comparison is a margin note.
    assert "4. Original" not in out


def test_an_uncompared_block_is_unchanged():
    from tsutawaru.ui.formatter import render_block

    seg = Segment.new(stream="main", original="海が怖い")
    seg.english = "the sea is scary"
    out = render_block(seg, Config().ui)
    # Not a bare "---" check: every block ends with the 50-dash separator.
    assert "Original (JP)" in out
    assert out.count("Original (JP)") == 1, "an alt tier leaked into a plain block"


def test_the_window_puts_the_two_models_in_columns():
    """QTextBrowser has no flexbox, so the layout is a table. Pin that."""
    from tsutawaru.ui.window_qt import _fmt_block

    html = _fmt_block(_seg_with_alt(), Config().ui)
    assert "<table" in html and html.count('width="50%"') == 2
    assert "qwen3" in html and "kotoba" in html
    assert "ディスコード" in html and "ディスコート" in html


def test_both_columns_carry_their_own_romaji():
    from tsutawaru.ui.window_qt import _fmt_block

    html = _fmt_block(_seg_with_alt(), Config().ui)
    assert "disukoodo de nagasu gurai kana" in html
    assert "disukooto de nagasu gurai kana" in html
    assert html.count('class="romaji"') == 2


def test_hiding_romaji_hides_it_in_both_columns():
    from tsutawaru.ui.window_qt import _fmt_block
    from tsutawaru.config import UiCfg

    html = _fmt_block(_seg_with_alt(), UiCfg(show_romaji=False))
    assert "disukoodo" not in html and "disukooto" not in html


def test_an_uncompared_line_keeps_the_single_column_layout():
    """The default path must not grow a table it does not need."""
    from tsutawaru.ui.window_qt import _fmt_block

    seg = Segment.new(stream="main", original="海が怖い")
    seg.romaji, seg.english = "umi ga kowai", "the sea is scary"
    html = _fmt_block(seg, Config().ui)
    assert "<table" not in html
    assert 'class="jp"' in html and 'class="romaji"' in html


def test_a_comparison_still_waiting_renders_placeholders_per_tier():
    from tsutawaru.ui.window_qt import _fmt_block

    seg = _seg_with_alt()
    seg.alt_romaji = seg.alt_english = ""
    assert _fmt_block(seg, Config().ui).count('class="waiting"') == 2
