# src/safety/operation_levels.py
from typing import Optional


OPERATION_LEVELS: dict[str, int] = {
    # Level 1: Auto-approved (read-only operations)
    "query_metrics": 1,
    "query_logs": 1,
    "query_traces": 1,
    "query_topology": 1,
    "query_change_events": 1,
    "query_processlist": 1,
    "search_fault_patterns": 1,
    "compare_configs": 1,

    # Level 2: Need human confirmation
    "restart_pod": 2,
    "scale_up": 2,
    "scale_down": 2,
    "adjust_rate_limit": 2,
    "adjust_circuit_breaker": 2,
    "clear_cache": 2,
    "run_diagnostic_command": 2,
    "add_index": 2,

    # Level 3: Need human approval with detailed plan
    "modify_config": 3,
    "execute_ddl": 3,
    "execute_dml": 3,
    "deploy_rollback": 3,
    "modify_route": 3,
    "adjust_connection_pool": 3,

    # Level 4: Completely forbidden
    "drop_database": 4,
    "drop_table": 4,
    "truncate_table": 4,
    "rm_rf": 4,
    "kill_process": 4,
    "iptables_modify": 4,
    "shutdown_service": 4,
}


LEVEL_DESCRIPTIONS: dict[int, str] = {
    1: "自动批准 - Agent可直接执行，无需人工介入",
    2: "需人工确认 - Agent生成建议，等待人工确认后执行",
    3: "需人工审批 - Agent生成详细方案，必须人工审批后执行",
    4: "完全禁止 - Agent不会提议此操作",
}


def classify_operation(action_name: str) -> Optional[int]:
    """Return the safety level (1-4) for an operation, or None if unknown."""
    return OPERATION_LEVELS.get(action_name)


def is_operation_allowed(action_name: str) -> bool:
    """Check if operation is allowed at all (Level 4 is not)."""
    level = classify_operation(action_name)
    if level is None:
        return False  # unknown operations are not allowed
    return level < 4


def get_level_description(level: int) -> str:
    return LEVEL_DESCRIPTIONS.get(level, f"未知等级: {level}")


def get_operations_by_level(level: int) -> list[str]:
    return [op for op, lv in OPERATION_LEVELS.items() if lv == level]
