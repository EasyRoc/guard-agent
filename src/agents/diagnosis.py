import time
import json
from datetime import datetime
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.models.alert import AlertEvent
from src.models.diagnosis import DiagnosisState, DiagnosisReport
from src.knowledge.fault_patterns import FaultPattern, match_patterns
from src.tools.diagnostic import DIAGNOSTIC_TOOLS


def _format_patterns(patterns: list[FaultPattern]) -> str:
    lines = []
    for p in patterns:
        lines.append(f"- [{p.id}] {p.name}")
        lines.append(f"  症状: {', '.join(p.symptoms)}")
        lines.append(f"  常见根因: {', '.join(p.common_causes)}")
        lines.append(f"  修复策略: {', '.join(p.repair_strategies)}")
    return "\n".join(lines)


def create_diagnosis_agent(
    llm: ChatOpenAI,
    fault_patterns: list[FaultPattern],
):
    """Create the 4-phase Diagnosis Agent as a compiled LangGraph StateGraph."""

    phase1_tools = DIAGNOSTIC_TOOLS[:4]  # metrics, logs, traces, change_events
    phase2_tools = DIAGNOSTIC_TOOLS[4:6]  # processlist, explain
    phase4_tools = DIAGNOSTIC_TOOLS[5:8]  # explain, heap_dump, config_diff

    async def phase1_collect(state: DiagnosisState) -> dict:
        alert = state["alert"]
        system_prompt = f"""你是一个SRE故障诊断专家。当前处于 **Phase 1: 信息收集** 阶段。

告警信息：
- 标题: {alert.title}
- 资源: {alert.resource}
- 指标: {alert.metric} = {alert.current_value} (阈值: {alert.threshold})
- 来源: {alert.source}
- 严重度: {alert.severity}

**任务**：调用工具全面收集告警相关的数据。必须至少调用:
1. query_metrics - 查询相关指标
2. query_logs - 查询相关错误日志
3. query_change_events - 查询最近变更

收集完数据后，用中文总结你收集到了什么，格式为JSON:
{{"summary": "数据摘要", "key_findings": ["发现1", "发现2"]}}

只输出JSON，不要其他内容。"""
        llm_with_tools = llm.bind_tools(phase1_tools)
        response = await llm_with_tools.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"开始Phase 1信息收集。告警ID: {alert.id}")
        ])
        return {
            "collected_data": {"phase1_output": response.content, "tool_calls_done": True},
            "messages": state.get("messages", []) + [response],
        }

    async def phase2_trace(state: DiagnosisState) -> dict:
        alert = state["alert"]
        collected = state.get("collected_data", {})
        system_prompt = f"""你是一个SRE故障诊断专家。当前处于 **Phase 2: 数据追踪** 阶段。

告警: {alert.title} on {alert.resource}
Phase 1 收集的数据: {json.dumps(collected, ensure_ascii=False, default=str)}

**任务**：追溯异常指标的来源和传播链路。深入调查：
- 哪些进程/查询占用了资源？
- 异常是从哪个组件开始传播的？
- 调用必要的工具深入分析（如query_processlist、query_explain等）

输出JSON格式:
{{"trace_chain": "异常传播链路描述", "affected_components": ["组件1", "组件2"], "bottleneck": "瓶颈所在"}}

只输出JSON。"""
        llm_with_tools = llm.bind_tools(phase2_tools)
        response = await llm_with_tools.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content="开始Phase 2数据追踪。")
        ])
        return {
            "trace_result": {"phase2_output": response.content},
            "messages": state.get("messages", []) + [response],
        }

    async def phase3_root_cause(state: DiagnosisState) -> dict:
        alert = state["alert"]
        collected = state.get("collected_data", {})
        trace = state.get("trace_result", {})

        matched = match_patterns(f"{alert.title} {alert.metric}", fault_patterns)
        matched_text = _format_patterns(matched) if matched else "无匹配的已知故障模式"

        system_prompt = f"""你是一个SRE故障诊断专家。当前处于 **Phase 3: 根因定位** 阶段。

告警: {alert.title} on {alert.resource}
Phase 1 数据: {json.dumps(collected, ensure_ascii=False, default=str)}
Phase 2 追踪: {json.dumps(trace, ensure_ascii=False, default=str)}

已知故障模式库（相似案例）:
{matched_text}

**任务**：定位根本原因。严格按以下格式输出JSON:
{{
    "root_cause": "根因的完整描述（一句话）",
    "evidence_chain": ["证据1", "证据2", "证据3"],
    "confidence": 0.85,
    "repair_hypothesis": "修复假设描述",
    "matched_pattern": "匹配到的故障模式ID（如有，否则为null）"
}}

要求：
- evidence_chain 必须包含至少3条具体证据
- confidence: 基于证据充分程度给出0-1的置信度
- 如果 confidence < 0.6，说明证据不足，标记为低置信度

只输出JSON。"""
        response = await llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content="开始Phase 3根因定位。")
        ])

        try:
            result = json.loads(response.content.strip().removeprefix("```json").removesuffix("```").strip())
        except json.JSONDecodeError:
            result = {
                "root_cause": response.content,
                "evidence_chain": [],
                "confidence": 0.3,
                "repair_hypothesis": "",
                "matched_pattern": None,
            }

        return {
            "root_cause": result.get("root_cause", ""),
            "evidence_chain": result.get("evidence_chain", []),
            "confidence": float(result.get("confidence", 0.5)),
            "repair_hypothesis": result.get("repair_hypothesis", ""),
            "messages": state.get("messages", []) + [response],
        }

    async def phase4_validate(state: DiagnosisState) -> dict:
        hypothesis = state.get("repair_hypothesis", "")
        root_cause = state.get("root_cause", "")
        evidence = state.get("evidence_chain", [])

        system_prompt = f"""你是一个SRE故障诊断专家。当前处于 **Phase 4: 验证修复** 阶段。

根因: {root_cause}
证据链: {json.dumps(evidence, ensure_ascii=False)}
修复假设: {hypothesis}

**任务**：在逻辑上验证修复假设的正确性。不执行实际操作，而是推演验证逻辑。
- 调用相关工具获取验证所需的数据（如EXPLAIN结果、配置对比等）
- 判断修复假设是否合理

输出JSON格式:
{{
    "hypothesis_validated": true/false,
    "validation_detail": "验证过程和结论",
    "revised_hypothesis": "如果验证失败，修正后的假设（否则为null）"
}}

只输出JSON。"""
        llm_with_tools = llm.bind_tools(phase4_tools)
        response = await llm_with_tools.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content="开始Phase 4验证修复。")
        ])

        try:
            result = json.loads(response.content.strip().removeprefix("```json").removesuffix("```").strip())
        except json.JSONDecodeError:
            result = {
                "hypothesis_validated": True,
                "validation_detail": response.content,
                "revised_hypothesis": None,
            }

        return {
            "hypothesis_validated": bool(result.get("hypothesis_validated", True)),
            "validation_detail": str(result.get("validation_detail", "")),
            "messages": state.get("messages", []) + [response],
        }

    async def generate_report(state: DiagnosisState) -> dict:
        now = datetime.now()
        report = DiagnosisReport(
            alert_id=state["alert"].id,
            timestamp=now,
            root_cause=state.get("root_cause", ""),
            evidence_chain=state.get("evidence_chain", []),
            confidence=float(state.get("confidence", 0.0)),
            repair_hypothesis=state.get("repair_hypothesis", ""),
            hypothesis_validated=bool(state.get("hypothesis_validated", False)),
            validation_detail=str(state.get("validation_detail", "")),
            diagnosis_duration_s=0.0,
            phases_completed=["phase1", "phase2", "phase3"],
        )
        if state.get("hypothesis_validated") is not None:
            report.phases_completed.append("phase4")

        return {"diagnosis_report": report}

    def should_validate(state: DiagnosisState) -> str:
        confidence = state.get("confidence", 0.0)
        if confidence >= 0.6:
            return "phase4"
        return "generate_report"

    workflow = StateGraph(DiagnosisState)

    workflow.add_node("phase1", phase1_collect)
    workflow.add_node("phase2", phase2_trace)
    workflow.add_node("phase3", phase3_root_cause)
    workflow.add_node("phase4", phase4_validate)
    workflow.add_node("generate_report", generate_report)

    workflow.set_entry_point("phase1")
    workflow.add_edge("phase1", "phase2")
    workflow.add_edge("phase2", "phase3")
    workflow.add_conditional_edges(
        "phase3",
        should_validate,
        {"phase4": "phase4", "generate_report": "generate_report"},
    )
    workflow.add_edge("phase4", "generate_report")
    workflow.add_edge("generate_report", END)

    return workflow.compile(checkpointer=MemorySaver())


async def run_diagnosis(agent, alert: AlertEvent) -> DiagnosisReport:
    """Run the diagnosis agent and return the report."""
    start = time.time()
    config = {"configurable": {"thread_id": alert.id}}
    initial_state: DiagnosisState = {
        "alert": alert,
        "collected_data": {},
        "trace_result": {},
        "root_cause": "",
        "evidence_chain": [],
        "confidence": 0.0,
        "repair_hypothesis": "",
        "validation_detail": "",
        "diagnosis_report": None,
        "messages": [],
    }

    final_state = await agent.ainvoke(initial_state, config)
    report = final_state.get("diagnosis_report")
    if report is None:
        raise RuntimeError("Diagnosis agent did not produce a report")
    report.diagnosis_duration_s = time.time() - start
    return report
