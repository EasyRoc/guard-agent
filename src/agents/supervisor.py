from typing import Any
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from src.models.alert import AlertEvent
from src.models.diagnosis import DiagnosisReport
from src.models.decision import DecisionProposal
from src.knowledge.fault_patterns import FaultPattern
from src.agents.diagnosis import create_diagnosis_agent, run_diagnosis
from src.agents.decision import run_decision


_pending_report: DiagnosisReport | None = None
_pending_proposal: DecisionProposal | None = None


def create_supervisor(
    llm: ChatOpenAI,
    fault_patterns: list[FaultPattern],
) -> Any:
    """Create the Supervisor Agent that orchestrates diagnosis -> decision -> confirmation."""

    diagnosis_agent = create_diagnosis_agent(llm, fault_patterns)

    @tool
    async def run_diagnosis_tool(alert_json: str) -> str:
        """调用诊断Agent执行四阶段故障诊断。参数 alert_json: AlertEvent的JSON字符串。
        必须在任何决策之前先调用此工具。返回诊断报告的JSON。"""
        import json
        alert_dict = json.loads(alert_json)
        from datetime import datetime
        alert = AlertEvent(
            id=alert_dict["id"],
            timestamp=datetime.fromisoformat(alert_dict["timestamp"]),
            source=alert_dict["source"],
            severity=alert_dict["severity"],
            title=alert_dict["title"],
            resource=alert_dict["resource"],
            metric=alert_dict["metric"],
            current_value=alert_dict["current_value"],
            threshold=alert_dict["threshold"],
            labels=alert_dict.get("labels", {}),
            raw_data=alert_dict.get("raw_data", {}),
        )
        global _pending_report
        _pending_report = await run_diagnosis(diagnosis_agent, alert)
        return json.dumps({
            "alert_id": _pending_report.alert_id,
            "root_cause": _pending_report.root_cause,
            "confidence": _pending_report.confidence,
            "evidence_chain": _pending_report.evidence_chain,
            "repair_hypothesis": _pending_report.repair_hypothesis,
            "hypothesis_validated": _pending_report.hypothesis_validated,
            "phases_completed": _pending_report.phases_completed,
        }, ensure_ascii=False, default=str)

    @tool
    async def run_decision_tool(unused: str = "") -> str:
        """调用决策Agent，基于诊断报告生成操作决策。
        参数 unused: 占位参数，传入空字符串即可。
        返回决策方案的JSON。"""
        import json
        global _pending_report, _pending_proposal
        if _pending_report is None:
            return json.dumps({"error": "没有诊断报告，请先运行诊断"})
        _pending_proposal = run_decision(_pending_report, llm=llm)
        return json.dumps({
            "requires_confirmation": _pending_proposal.requires_confirmation,
            "reasoning": _pending_proposal.reasoning,
            "proposals": [
                {
                    "action": p.action,
                    "level": p.level,
                    "description": p.description,
                    "expected_impact": p.expected_impact,
                    "rollback_plan": p.rollback_plan,
                }
                for p in _pending_proposal.proposals
            ],
        }, ensure_ascii=False)

    @tool
    def request_human_confirmation(summary: str) -> str:
        """当决策需要人工确认时调用此工具。参数 summary: 操作摘要。
        返回 'confirmed' 或 'rejected'。"""
        print("\n" + "=" * 60)
        print(" [人工确认请求]")
        print("=" * 60)
        print(summary)
        print("-" * 60)
        response = input("输入 confirm 确认执行 / reject 拒绝: ").strip().lower()
        if response == "confirm":
            return "confirmed"
        return "rejected"

    system_prompt = """你是一个SRE运维 Supervisor Agent。你的职责是编排故障诊断和决策流程。

**必须严格遵循的工作流**：
1. 收到告警 -> 立即调用 run_diagnosis_tool 进行诊断（将告警信息以JSON格式传入）
2. 收到诊断报告 -> 立即调用 run_decision_tool 生成决策方案
3. 收到决策方案 -> 如果 requires_confirmation=true，调用 request_human_confirmation
4. 收到确认后 -> 输出最终总结
5. 如果 requires_confirmation=false（Level 1操作），直接输出总结

**重要规则**：
- 绝不能在诊断之前调用决策
- 绝不能在决策之前请求确认
- 每一步完成后立即进行下一步
- 最终输出完整的处理总结"""

    tools = [run_diagnosis_tool, run_decision_tool, request_human_confirmation]

    return create_react_agent(
        llm,
        tools,
        state_modifier=system_prompt,
        checkpointer=MemorySaver(),
    )


async def run_supervisor(
    supervisor,
    alert: AlertEvent,
    auto_confirm: bool = False,
) -> dict[str, Any]:
    """Run the supervisor for a given alert."""
    import json

    alert_json = json.dumps({
        "id": alert.id,
        "timestamp": alert.timestamp.isoformat(),
        "source": alert.source,
        "severity": alert.severity,
        "title": alert.title,
        "resource": alert.resource,
        "metric": alert.metric,
        "current_value": alert.current_value,
        "threshold": alert.threshold,
        "labels": alert.labels,
        "raw_data": alert.raw_data,
    }, ensure_ascii=False, default=str)

    config = {"configurable": {"thread_id": alert.id}}

    prompt = (
        f"收到一个运维告警，请立即开始诊断流程。\n\n"
        f"告警信息（JSON格式）:\n{alert_json}\n\n"
        f"请调用 run_diagnosis_tool 开始诊断。"
    )

    result = await supervisor.ainvoke(
        {"messages": [{"role": "user", "content": prompt}]},
        config,
    )

    global _pending_report, _pending_proposal

    return {
        "alert_id": alert.id,
        "diagnosis_report": _pending_report,
        "decision_proposal": _pending_proposal,
        "supervisor_output": result,
    }
