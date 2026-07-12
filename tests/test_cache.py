from pathlib import Path

from mas_deepr.tools.cache import WebCache


def test_set_get_roundtrip(tmp_path: Path) -> None:
    cache = WebCache(tmp_path / "cache.sqlite3")
    key = cache.make_key("search", query="hello", max_results=5)
    assert cache.get(key) is None

    cache.set(key, "search", {"hits": [{"title": "t", "url": "u", "snippet": "s"}]})
    assert cache.get(key) == {"hits": [{"title": "t", "url": "u", "snippet": "s"}]}


def test_make_key_stable_and_order_independent() -> None:
    k1 = WebCache.make_key("search", query="a", max_results=5)
    k2 = WebCache.make_key("search", max_results=5, query="a")
    assert k1 == k2


def test_make_key_distinguishes_kind_and_params() -> None:
    k1 = WebCache.make_key("search", query="a")
    k2 = WebCache.make_key("fetch", query="a")
    k3 = WebCache.make_key("search", query="b")
    assert len({k1, k2, k3}) == 3


def test_cache_persists_across_instances(tmp_path: Path) -> None:
    db_path = tmp_path / "cache.sqlite3"
    key = WebCache.make_key("fetch", url="http://example.com")
    WebCache(db_path).set(key, "fetch", {"text": "hello"})
    assert WebCache(db_path).get(key) == {"text": "hello"}
