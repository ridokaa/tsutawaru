"""--record and --export-md, which until now were accepted and then ignored.

Both flags parsed, both were stored on the Orchestrator, and nothing ever read
either one — a session ran to completion and wrote no file, with no error to
say so. These tests pin the behaviour the flags advertise.

The subtle requirement is *when* a segment may be rendered. Segments are mutated
in place by later stages, and no single event means "finished": the gloss lane
clears `partial` while the sentence lane is still fetching English. The recorder
therefore holds references and renders at close, which is what the last two
tests are really checking.
"""
from __future__ import annotations

import json


from tsutawaru.models import Segment, Token
from tsutawaru.pipeline.recorder import SessionRecorder


def _seg(jp: str, en: str = "", tokens=()) -> Segment:
    s = Segment.new(stream="main", original=jp)
    s.english = en
    s.tokens = list(tokens)
    return s


def _tok(surface, base, romaji="", gloss="", pos="名詞") -> Token:
    return Token(surface=surface, base_form=base, romaji=romaji,
                 gloss=gloss, pos=pos, pos1="一般")


def test_disabled_recorder_is_inert(tmp_path):
    rec = SessionRecorder()
    assert not rec.enabled
    rec.add(_seg("テスト"))
    rec.close()
    assert not list(tmp_path.iterdir())


def test_jsonl_holds_one_object_per_line(tmp_path):
    p = tmp_path / "s.jsonl"
    rec = SessionRecorder(jsonl=str(p))
    rec.add(_seg("海が怖い", "the sea is scary"))
    rec.add(_seg("潜ってみた", "i dove in"))
    rec.close()

    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()]
    assert [r["jp"] for r in rows] == ["海が怖い", "潜ってみた"]
    assert [r["en"] for r in rows] == ["the sea is scary", "i dove in"]


def test_late_mutation_still_reaches_the_file(tmp_path):
    """The point of holding references rather than serialising on add()."""
    p = tmp_path / "s.jsonl"
    rec = SessionRecorder(jsonl=str(p))
    seg = _seg("後で行く")
    rec.add(seg)          # registered with no translation at all
    seg.english = "i'll go later"   # ... which arrives afterwards
    seg.partial = False
    rec.close()

    row = json.loads(p.read_text(encoding="utf-8").splitlines()[0])
    assert row["en"] == "i'll go later"
    assert row["partial"] is False


def test_awaiting_english_counts_untranslated_segments(tmp_path):
    rec = SessionRecorder(jsonl=str(tmp_path / "s.jsonl"))
    a, b = _seg("一"), _seg("二", "two")
    rec.add(a)
    rec.add(b)
    assert rec.awaiting_english() == 1
    a.english = "one"
    assert rec.awaiting_english() == 0


def test_study_sheet_has_both_tiers(tmp_path):
    p = tmp_path / "s.md"
    rec = SessionRecorder(markdown=str(p), meta={"model": "kotoba"})
    seg = _seg("海が怖い", "the sea is scary",
               [_tok("海", "海", "umi", "sea"), _tok("怖い", "怖い", "kowai", "scary")])
    seg.romaji = "umi ga kowai"
    rec.add(seg)
    rec.close()

    md = p.read_text(encoding="utf-8")
    assert "kotoba" in md
    assert "## Vocabulary" in md and "## Transcript" in md
    assert "| 海 | umi | sea | 1 |" in md
    assert "umi ga kowai" in md          # the romaji tier
    assert "the sea is scary" in md      # the english tier


def test_vocabulary_groups_inflections_under_the_dictionary_form(tmp_path):
    p = tmp_path / "s.md"
    rec = SessionRecorder(markdown=str(p))
    for surface in ("潜って", "潜った"):
        rec.add(_seg("x", "y", [_tok(surface, "潜る", "mogutte", "dive", pos="動詞")]))
    rec.close()

    md = p.read_text(encoding="utf-8")
    row = next(l for l in md.splitlines() if l.startswith("| 潜る"))
    assert "| 2 |" in row                      # counted together
    assert "潜って" in row and "潜った" in row   # both surfaces kept
    # The headword's own reading, never an inflected token's — the Word column
    # says 潜る, so the Romaji column must not say "mogutte".
    assert "moguru" in row and "mogutte" not in row


def test_particles_stay_out_of_the_vocabulary(tmp_path):
    p = tmp_path / "s.md"
    rec = SessionRecorder(markdown=str(p))
    rec.add(_seg("海が", "", [_tok("海", "海", "umi", "sea"),
                              _tok("が", "が", "ga", "", pos="助詞")]))
    rec.close()

    vocab = p.read_text(encoding="utf-8").split("## Transcript")[0]
    assert "| 海 " in vocab
    assert "| が " not in vocab


def test_empty_session_writes_a_sheet_that_says_so(tmp_path):
    p = tmp_path / "s.md"
    rec = SessionRecorder(markdown=str(p))
    rec.close()
    assert "Nothing was transcribed" in p.read_text(encoding="utf-8")


def test_the_orchestrator_registers_every_emitted_segment(tmp_path):
    """The wiring, not the recorder: a live run is the only other way to see it.

    `_emit_segment` is the one chokepoint every transcript line passes through,
    which is why registration lives there. Nothing else in the suite would
    notice if that call were dropped.
    """
    from tsutawaru.config import load
    from tsutawaru.models import Utterance
    from tsutawaru.pipeline.orchestrator import Orchestrator
    import numpy as np

    cfg = load()
    orch = Orchestrator(cfg, record=str(tmp_path / "s.jsonl"))

    utt = Utterance.new(audio=np.zeros(1600, dtype=np.float32),
                        t_start=0.0, t_end=0.1)
    res = type("R", (), {"text": "海が怖い", "avg_logprob": -0.2,
                         "no_speech_prob": 0.01, "language": "ja"})()
    orch._emit_segment(utt, res)

    assert [s.original for s in orch.recorder._segs] == ["海が怖い"]


def test_an_unwritable_path_does_not_take_down_the_pipeline(tmp_path):
    """A recorder that kills the session it observes is worse than a lost log."""
    bad = tmp_path / "not-a-dir"
    bad.write_text("i am a file")
    rec = SessionRecorder(jsonl=str(bad / "s.jsonl"), markdown=str(bad / "s.md"))
    rec.add(_seg("テスト", "test"))
    rec.close()  # must not raise


# ------------------------------------------------------- per-line timestamps


def _timed(jp: str, audio_end: float, stt: float, done: float) -> Segment:
    s = _seg(jp)
    s.t_audio_end, s.t_stt_done, s.t_complete = audio_end, stt, done
    return s


def test_timing_measures_from_end_of_speech():
    """The budget is what a listener waits *after* someone stops talking.

    Measuring from the start of the utterance would score a long sentence worse
    than a short one purely for being long, which is not the question anyone is
    asking of this pipeline.
    """
    t = _timed("あ", audio_end=100.0, stt=100.4, done=101.2).timing()
    assert t["stt_ms"] == 400.0
    assert t["line_ms"] == 1200.0


def test_timing_reports_nothing_for_stages_never_reached():
    """0.0 is the unset marker, not a reading — monotonic zero is arbitrary."""
    t = _seg("あ").timing()
    assert t == {"stt_ms": None, "line_ms": None}


def test_timing_reaches_the_recorded_line(tmp_path):
    """Regression: the stamps lived on Segment but never reached any file.

    A session recorded before this could not answer the latency question even
    with the audio still on disk, because nothing in the log said when a line
    arrived relative to the speech that produced it.
    """
    p = tmp_path / "s.jsonl"
    rec = SessionRecorder(jsonl=str(p))
    rec.add(_timed("海が怖い", audio_end=10.0, stt=10.35, done=10.9))
    rec.close()

    row = json.loads(p.read_text(encoding="utf-8").splitlines()[0])
    assert row["t"]["stt_ms"] == 350.0
    assert row["t"]["line_ms"] == 900.0


def test_study_sheet_header_carries_the_latency_percentiles(tmp_path):
    p = tmp_path / "study.md"
    rec = SessionRecorder(markdown=str(p))
    for i, ms in enumerate((0.4, 0.8, 1.2, 3.0)):
        rec.add(_timed(f"線{i}", audio_end=100.0, stt=100.1, done=100.0 + ms))
    rec.close()

    header = p.read_text(encoding="utf-8").splitlines()[2]
    assert "p50 800 ms" in header  # nearest-rank, not interpolated
    assert "p95 3000 ms" in header


def test_untimed_session_omits_the_latency_figure(tmp_path):
    """Better a header with no number than one reporting 0 ms."""
    p = tmp_path / "study.md"
    rec = SessionRecorder(markdown=str(p))
    rec.add(_seg("テスト"))
    rec.close()
    assert "p50" not in p.read_text(encoding="utf-8")
