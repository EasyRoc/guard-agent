# tests/test_operation_levels.py
from src.safety.operation_levels import classify_operation, get_level_description, is_operation_allowed


def test_classify_known_level1():
    assert classify_operation("query_metrics") == 1
    assert classify_operation("query_logs") == 1
    assert classify_operation("query_traces") == 1


def test_classify_known_level2():
    assert classify_operation("restart_pod") == 2
    assert classify_operation("scale_up") == 2
    assert classify_operation("adjust_rate_limit") == 2


def test_classify_known_level3():
    assert classify_operation("modify_config") == 3
    assert classify_operation("execute_ddl") == 3
    assert classify_operation("execute_dml") == 3


def test_classify_known_level4():
    assert classify_operation("drop_database") == 4
    assert classify_operation("drop_table") == 4
    assert classify_operation("rm_rf") == 4


def test_classify_unknown_returns_none():
    assert classify_operation("some_weird_action") is None


def test_is_operation_allowed():
    assert is_operation_allowed("query_metrics") is True     # Level 1
    assert is_operation_allowed("restart_pod") is True       # Level 2 (allowed with confirm)
    assert is_operation_allowed("execute_ddl") is True       # Level 3 (allowed with approval)
    assert is_operation_allowed("drop_database") is False    # Level 4 (blocked)


def test_get_level_description():
    desc = get_level_description(1)
    assert "自动" in desc
    desc2 = get_level_description(2)
    assert "确认" in desc2
    desc4 = get_level_description(4)
    assert "禁止" in desc4
