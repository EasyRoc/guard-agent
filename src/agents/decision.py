from datetime import datetime
from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from src.models.diagnosis import DiagnosisReport
from src.models.decision import DecisionProposal, OperationPlan
from src.safety.operation_levels import classify_operation, is_operation_allowed, get_level_description


_ACTION_PATTERNS: list[tuple[str, str, str, str]] = [
    # (keyword, action_name, description_template, verification)
    ("restart", "restart_pod", "重启Pod", "kubectl get pods 确认Pod状态为Running"),
    ("重启", "restart_pod", "重启Pod", "kubectl get pods 确认Pod状态为Running"),
    ("scale up", "scale_up", "扩容实例", "检查实例数和负载指标是否恢复正常"),
    ("scale down", "scale_down", "缩容实例", "检查剩余实例负载是否在安全范围内"),
    ("扩容", "scale_up", "扩容实例", "检查实例数和负载指标是否恢复正常"),
    ("限流", "adjust_rate_limit", "调整限流阈值", "检查错误率和QPS是否恢复正常"),
    ("熔断", "adjust_circuit_breaker", "调整熔断参数", "检查服务调用成功率"),
    ("clear cache", "clear_cache", "清理缓存", "检查缓存命中率和内存使用"),
    ("清除缓存", "clear_cache", "清理缓存", "检查缓存命中率和内存使用"),
    ("add index", "add_index", "添加数据库索引", "EXPLAIN验证查询计划已使用新索引"),
    ("添加索引", "add_index", "添加数据库索引", "EXPLAIN验证查询计划已使用新索引"),
    ("create index", "add_index", "添加数据库索引", "EXPLAIN验证查询计划已使用新索引"),
    ("回滚配置", "deploy_rollback", "回滚配置到上一版本", "检查错误率是否恢复到部署前水平"),
    ("deploy rollback", "deploy_rollback", "回滚配置到上一版本", "检查错误率是否恢复到部署前水平"),
    ("rollback", "deploy_rollback", "回滚部署", "检查服务指标是否恢复正常"),
    ("modify config", "modify_config", "修改配置", "检查配置生效并验证服务行为"),
    ("修改配置", "modify_config", "修改配置", "检查配置生效并验证服务行为"),
    ("调整连接池", "adjust_connection_pool", "调整数据库连接池参数", "检查连接池使用率和等待队列"),
    # Level 4 operations (detected but rejected)
    ("drop database", "drop_database", None, None),
    ("drop table", "drop_table", None, None),
    ("truncate", "truncate_table", None, None),
    ("rm -rf", "rm_rf", None, None),
    ("kill -9", "kill_process", None, None),
]


def _extract_actions(hypothesis: str) -> list[tuple[str, str, str, str]]:
    actions = []
    hypothesis_lower = hypothesis.lower()
    for keyword, action_name, desc_template, verification in _ACTION_PATTERNS:
        if keyword in hypothesis_lower:
            actions.append((keyword, action_name, desc_template, verification))
    return actions


def run_decision(
    report: DiagnosisReport,
    llm: Optional[ChatOpenAI] = None,
) -> DecisionProposal:
    actions = _extract_actions(report.repair_hypothesis)
    proposals: list[OperationPlan] = []

    for keyword, action_name, desc_template, verification in actions:
        level = classify_operation(action_name)

        if level is None:
            if llm:
                level = _llm_classify(llm, action_name, report.repair_hypothesis)
            else:
                continue

        if level == 4:
            # Level 4 operations are completely rejected - do not include in proposals
            continue

        if desc_template is None:
            continue

        proposals.append(OperationPlan(
            action=action_name,
            level=level,
            description=desc_template,
            expected_impact=f"执行{desc_template}操作",
            rollback_plan="如失败则恢复原状态" if level >= 2 else "无需回滚",
            verification=verification or "检查相关指标是否恢复",
        ))

    if not proposals and report.root_cause:
        proposals.append(OperationPlan(
            action="query_metrics",
            level=1,
            description=f"观察: {report.root_cause}",
            expected_impact="持续监控指标变化",
            rollback_plan="无",
            verification="持续观察30分钟",
        ))

    requires_confirmation = any(p.level >= 2 for p in proposals)

    parts = []
    if proposals:
        parts.append(f"诊断报告置信度: {report.confidence:.0%}")
        parts.append(f"根因: {report.root_cause}")
        for p in proposals:
            parts.append(f"[Level {p.level}] {p.description} - {get_level_description(p.level)}")

    return DecisionProposal(
        report_id=report.alert_id,
        timestamp=datetime.now(),
        proposals=proposals,
        requires_confirmation=requires_confirmation,
        reasoning="\n".join(parts),
    )


def _llm_classify(llm: ChatOpenAI, action_name: str, context: str) -> int:
    system_prompt = """你是一个运维安全专家。请判断以下操作的安全等级。

等级定义：
1 - 只读操作/查询，无副作用
2 - 有副作用的操作，但影响范围可控，需人工确认
3 - 高风险操作，可能影响核心业务，需人工审批
4 - 极度危险操作，绝对不能执行

请只回复数字1-4。"""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"操作: {action_name}\n上下文: {context}\n\n请判断安全等级 (1-4):")
    ])

    try:
        level = int(response.content.strip()[0])
        return max(1, min(4, level))
    except (ValueError, IndexError):
        return 3
