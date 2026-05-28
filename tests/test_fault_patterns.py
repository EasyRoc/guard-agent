# tests/test_fault_patterns.py
from src.knowledge.fault_patterns import load_fault_patterns, FaultPattern, match_patterns


def test_load_fault_patterns():
    patterns = load_fault_patterns()
    assert len(patterns) == 3
    assert all(isinstance(p, FaultPattern) for p in patterns)


def test_pattern_ids():
    patterns = load_fault_patterns()
    ids = {p.id for p in patterns}
    assert ids == {"connection_pool_exhaustion", "oom_cache_unbounded", "config_change_error"}


def test_match_patterns_by_keyword():
    patterns = load_fault_patterns()
    matches = match_patterns("数据库连接数超过阈值，连接池即将耗尽", patterns)
    assert len(matches) > 0
    assert matches[0].id == "connection_pool_exhaustion"


def test_match_patterns_no_match():
    patterns = load_fault_patterns()
    matches = match_patterns("unknown weird error XYZ123", patterns)
    assert len(matches) == 0
