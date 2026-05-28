from src.tools.diagnostic import (
    set_scenario,
    query_metrics,
    query_logs,
    query_traces,
    query_change_events,
    query_processlist,
    query_explain,
    query_heap_dump,
    query_config_diff,
)
from src.tools.mock_data import get_alert


def test_set_scenario_and_query_metrics():
    set_scenario("connection_pool")
    result = query_metrics.invoke({"metric_names": "connections_active,qps"})
    assert "connections_active" in result
    assert "148" in result
    assert "qps" in result


def test_query_logs():
    set_scenario("connection_pool")
    result = query_logs.invoke({"keyword": "timeout"})
    assert "timeout" in result.lower()


def test_query_traces():
    set_scenario("connection_pool")
    result = query_traces.invoke({"service_name": "api-server"})
    assert "SELECT * FROM orders" in result


def test_query_change_events():
    set_scenario("config_change")
    result = query_change_events.invoke({"hours": 24})
    assert "bank_api_timeout" in result


def test_query_processlist_connection_pool():
    set_scenario("connection_pool")
    result = query_processlist.invoke({})
    assert "SELECT * FROM orders" in result
    assert "Sending data" in result


def test_query_explain():
    set_scenario("connection_pool")
    result = query_explain.invoke({"query_text": "SELECT * FROM orders WHERE status='pending'"})
    assert "ALL" in result
    assert "idx_orders_status" in result


def test_query_heap_dump():
    set_scenario("oom")
    result = query_heap_dump.invoke({})
    assert "ConcurrentHashMap" in result
    assert "unlimited" in result


def test_query_config_diff():
    set_scenario("config_change")
    result = query_config_diff.invoke({})
    assert "30000" in result
    assert "3000" in result


def test_get_alert():
    alert = get_alert("connection_pool")
    assert alert.id == "alert-001"
    assert alert.resource == "mysql-orders-db"
    assert alert.current_value == 148.0


def test_get_alert_oom():
    alert = get_alert("oom")
    assert alert.id == "alert-002"
    assert "OOM" in alert.title
