"""Model selection, including the Japanese-specialised kotoba-whisper."""
from __future__ import annotations

import pytest

from tsutawaru.config import Config, SttCfg, load


def test_kotoba_is_an_accepted_model_name():
    cfg = load(None, {"stt": {"model": "kotoba"}})
    assert cfg.stt.model == "kotoba"
    assert not cfg.validate()


def test_unknown_model_names_are_still_rejected():
    cfg = load(None, {"stt": {"model": "enormous"}})
    assert cfg.stt.model == SttCfg().model  # fell back to the default


def test_default_model_is_unchanged():
    assert Config().stt.model == "kotoba"


def test_mlx_maps_kotoba_to_the_mlx_build():
    pytest.importorskip("mlx_whisper")
    from tsutawaru.stt.mlx_whisper_engine import REPO

    assert REPO["kotoba"] == "kaiinui/kotoba-whisper-v2.0-mlx"
    assert REPO["medium"] == "mlx-community/whisper-medium-mlx-q4"


def test_faster_maps_kotoba_to_the_ctranslate2_build():
    """A different artefact from the MLX one — same model, other runtime."""
    pytest.importorskip("faster_whisper")
    from tsutawaru.stt.faster_whisper_engine import REPO

    assert REPO["kotoba"] == "kotoba-tech/kotoba-whisper-v2.0-faster"


def test_faster_passes_plain_size_names_through_untouched():
    """WhisperModel takes size names directly; only non-sizes need mapping."""
    pytest.importorskip("faster_whisper")
    from tsutawaru.stt.faster_whisper_engine import REPO

    for size in ("tiny", "base", "small", "medium"):
        assert REPO.get(size, size) == size


def test_every_choice_resolves_on_the_mlx_path():
    """A name accepted by config must not blow up at engine construction.

    There are now two MLX runtimes rather than one, so the check is that every
    selectable name is claimed by exactly one of them. "Claimed by neither"
    means a config the validator accepts and the factory cannot build;
    "claimed by both" means the factory's routing is ambiguous and which model
    actually ran would depend on branch order.
    """
    pytest.importorskip("mlx_whisper")
    from tsutawaru.config import _CHOICES
    from tsutawaru.stt.mlx_whisper_engine import REPO
    from tsutawaru.stt.qwen_mlx_engine import REPO as QWEN

    for name in _CHOICES[("stt", "model")]:
        assert (name in REPO) != (name in QWEN), (
            f"{name} is selectable but resolves to "
            f"{'both runtimes' if name in REPO else 'no runtime'}"
        )
