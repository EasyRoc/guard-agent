from langchain_core.tools import tool
from src.tools.mock_data import (
    get_mock_metrics,
    get_mock_logs,
    get_mock_traces,
    get_mock_change_events,
    get_mock_extra,
)


_current_scenario: str = "connection_pool"


def set_scenario(scenario: str) -> None:
    global _current_scenario
    _current_scenario = scenario


def get_scenario() -> str:
    return _current_scenario


@tool
def query_metrics(metric_names: str) -> str:
    """查询指标数据。参数 metric_names: 逗号分隔的指标名列表，如 'connections_active,qps,error_rate'。返回指标名和值。"""
    metrics = get_mock_metrics(_current_scenario)
    requested = [m.strip() for m in metric_names.split(",")]
    result = {}
    for name in requested:
        if name in metrics:
            result[name] = metrics[name]
    if not result:
        return f"No metrics found for: {metric_names}. Available: {list(metrics.keys())}"
    return "\n".join(f"{k}: {v}" for k, v in result.items())


@tool
def query_logs(keyword: str, limit: int = 20) -> str:
    """查询日志。参数 keyword: 搜索关键词。参数 limit: 最大返回条数。返回匹配的日志行。"""
    logs = get_mock_logs(_current_scenario)
    matched = [log for log in logs if keyword.lower() in log.lower()]
    if not matched:
        matched = logs
    return "\n".join(matched[:limit])


@tool
def query_traces(service_name: str) -> str:
    """查询服务调用链。参数 service_name: 服务名称。返回调用链记录。"""
    traces = get_mock_traces(_current_scenario)
    matched = [t for t in traces if service_name.lower() in t.lower()]
    if not matched:
        matched = traces
    return "\n".join(matched)


@tool
def query_change_events(hours: int = 24) -> str:
    """查询变更事件。参数 hours: 查询最近多少小时的变更记录。返回变更事件列表。"""
    events = get_mock_change_events(_current_scenario)
    return "\n".join(
        f"[{e['time']}] {e['type']}: {e.get('service', '')} - {e.get('description', '')}"
        for e in events
    )


@tool
def query_processlist() -> str:
    """查询数据库当前进程列表 (SHOW PROCESSLIST)。返回当前活跃的连接和查询。"""
    data = get_mock_extra(_current_scenario, "mock_processlist")
    if not data:
        return "No processlist data available for this scenario."
    lines = []
    for p in data:
        lines.append(
            f"ID:{p['id']} User:{p['user']} Host:{p['host']} DB:{p['db']} "
            f"Command:{p['command']} Time:{p['time']}s State:{p['state']} Info:{p.get('info', '')}"
        )
    return "\n".join(lines)


@tool
def query_explain(query_text: str) -> str:
    """获取SQL查询的执行计划(EXPLAIN)。参数 query_text: SQL语句。返回执行计划。"""
    import json
    data = get_mock_extra(_current_scenario, "mock_explain")
    if not data:
        return "No EXPLAIN data available for this scenario."
    plan = data.get("plan", {})
    with_index = data.get("with_index_plan", {})
    return (
        f"当前执行计划:\n{json.dumps(plan, indent=2, ensure_ascii=False)}\n\n"
        f"如果添加索引后的执行计划:\n{json.dumps(with_index, indent=2, ensure_ascii=False)}"
    )


@tool
def query_heap_dump() -> str:
    """查询JVM堆内存分析数据。返回堆中占用最多的类。"""
    import json
    data = get_mock_extra(_current_scenario, "mock_heap_dump")
    if not data:
        return "No heap dump data available for this scenario."
    return json.dumps(data, indent=2, ensure_ascii=False)


@tool
def query_config_diff() -> str:
    """查询最近一次配置变更的差异。返回配置文件的变更内容。"""
    data = get_mock_extra(_current_scenario, "mock_config_diff")
    if not data:
        return "No config diff data available for this scenario."
    lines = [f"文件: {data.get('file', 'unknown')}"]
    for change in data.get("changes", []):
        lines.append(
            f"  {change['key']}: {change['old_value']} → {change['new_value']}"
            + (f"  // {change['note']}" if change.get("note") else "")
        )
    return "\n".join(lines)


DIAGNOSTIC_TOOLS = [
    query_metrics,
    query_logs,
    query_traces,
    query_change_events,
    query_processlist,
    query_explain,
    query_heap_dump,
    query_config_diff,
]
