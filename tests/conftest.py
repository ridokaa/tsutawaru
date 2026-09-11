import sys
from pathlib import Path

import pytest

# Add project root to sys.path
root_dir = Path(__file__).resolve().parents[1]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))


@pytest.fixture(autouse=True)
def _no_jmdict(tmp_path, monkeypatch):
    """Point the JMdict gloss lookup at a database that is not there.

    `~/Library/Caches/tsutawaru/jmdict.db` is built by the user, not the repo, so
    a test that reaches it passes or fails depending on whose machine it runs on
    — the gloss-lane tests went green in CI and red locally the moment JMdict
    landed. Tests that want the dictionary build their own fixture and override
    this; everything else sees a clean miss and exercises the backend path.
    """
    from tsutawaru.translate import jmdict

    monkeypatch.setattr(jmdict, "db_path", lambda: tmp_path / "no-jmdict.db")
    monkeypatch.setattr(jmdict, "_lookup", jmdict._Lookup())
