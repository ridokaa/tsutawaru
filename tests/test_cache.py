"""Tests for L1 LRU and L2 SQLite GlossCache."""
from __future__ import annotations

from tsutawaru.translate.cache import LRU, GlossCache


def test_lru_eviction():
    lru = LRU(cap=2)
    lru.put("a", "1")
    lru.put("b", "2")
    assert lru.get("a") == "1"
    lru.put("c", "3")  # Evicts "b" because "a" was accessed
    assert lru.get("b") is None
    assert lru.get("a") == "1"
    assert lru.get("c") == "3"


def test_gloss_cache_persistence_across_instances(tmp_path):
    db_file = str(tmp_path / "gloss_test.db")
    cache1 = GlossCache(cap=10, persist=True, db_path=db_file)
    cache1.put("google:食べる:動詞", "to eat")
    cache1.flush()

    # New cache instance without L1 in-memory warm state
    cache2 = GlossCache(cap=10, persist=True, db_path=db_file)
    assert cache2.get("google:食べる:動詞") == "to eat"


def test_gloss_cache_flush_persists_uncommitted_batch(tmp_path):
    db_file = str(tmp_path / "gloss_batch.db")
    cache1 = GlossCache(cap=10, persist=True, db_path=db_file)
    cache1.put("google:犬:名詞", "dog")
    # Before flush, directly check SQLite table — may still be in _pending
    cache1.flush()

    cache2 = GlossCache(cap=10, persist=True, db_path=db_file)
    assert cache2.get("google:犬:名詞") == "dog"


def test_provider_scoped_keys_do_not_collide(tmp_path):
    db_file = str(tmp_path / "gloss_scope.db")
    cache = GlossCache(cap=10, persist=True, db_path=db_file)
    cache.put("google:端:名詞", "edge")
    cache.put("deepl:端:名詞", "end")
    assert cache.get("google:端:名詞") == "edge"
    assert cache.get("deepl:端:名詞") == "end"
