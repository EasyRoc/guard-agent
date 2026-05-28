"""Guard Agent CLI - Intelligent SRE Operations Assistant."""
import argparse
import asyncio
import json
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

from src.tools.diagnostic import set_scenario
from src.tools.mock_data import get_alert
from src.knowledge.fault_patterns import load_fault_patterns
from src.utils.llm import create_llm
from src.utils.logging import setup_logging
from src.agents.supervisor import create_supervisor, run_supervisor


def main():
    parser = argparse.ArgumentParser(
        description="Guard Agent - 智能运维故障诊断助手",
    )
    parser.add_argument(
        "scenario",
        nargs="?",
        choices=["connection_pool", "oom", "config_change"],
        default="connection_pool",
        help="故障场景 (默认: connection_pool)",
    )
    parser.add_argument(
        "--auto-confirm",
        action="store_true",
        help="自动确认所有Level 2/3操作（跳过人工确认）",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="输出JSON报告到文件",
    )

    args = parser.parse_args()
    setup_logging()

    print(f"\n{'='*60}")
    print(f" Guard Agent - 智能运维故障诊断")
    print(f" 场景: {args.scenario}")
    print(f" 时间: {datetime.now().isoformat()}")
    print(f"{'='*60}\n")

    set_scenario(args.scenario)
    alert = get_alert(args.scenario)
    patterns = load_fault_patterns()
    llm = create_llm()

    print(f"[告警] {alert.title}")
    print(f"[资源] {alert.resource}")
    print(f"[指标] {alert.metric} = {alert.current_value} (阈值: {alert.threshold})")
    print(f"\n开始诊断...\n")

    async def run():
        supervisor = create_supervisor(llm, patterns)
        result = await run_supervisor(supervisor, alert, auto_confirm=args.auto_confirm)

        report = result["diagnosis_report"]
        proposal = result["decision_proposal"]

        print(f"\n{'='*60}")
        print(f" 诊断报告")
        print(f"{'='*60}")
        print(f"完成阶段: {', '.join(report.phases_completed)}")
        print(f"根因: {report.root_cause}")
        print(f"置信度: {report.confidence:.0%}")
        print(f"证据链:")
        for i, ev in enumerate(report.evidence_chain, 1):
            print(f"  {i}. {ev}")
        print(f"修复假设: {report.repair_hypothesis}")
        print(f"假设验证: {'通过' if report.hypothesis_validated else '未通过'}")
        print(f"诊断耗时: {report.diagnosis_duration_s:.1f}s")

        if proposal:
            print(f"\n{'='*60}")
            print(f" 决策方案")
            print(f"{'='*60}")
            print(f"需要人工确认: {'是' if proposal.requires_confirmation else '否'}")
            print(f"决策理由: {proposal.reasoning}")
            for i, p in enumerate(proposal.proposals, 1):
                print(f"\n 方案{i}: [{p.action}] Level {p.level}")
                print(f"   描述: {p.description}")
                print(f"   预期影响: {p.expected_impact}")
                print(f"   回滚方案: {p.rollback_plan}")
                print(f"   验证方法: {p.verification}")

        if args.output:
            output_data = {
                "scenario": args.scenario,
                "alert": {"id": alert.id, "title": alert.title, "resource": alert.resource},
                "diagnosis": {
                    "root_cause": report.root_cause,
                    "confidence": report.confidence,
                    "evidence_chain": report.evidence_chain,
                    "repair_hypothesis": report.repair_hypothesis,
                    "phases_completed": report.phases_completed,
                },
            }
            if proposal:
                output_data["decision"] = {
                    "requires_confirmation": proposal.requires_confirmation,
                    "reasoning": proposal.reasoning,
                    "proposals": [
                        {"action": p.action, "level": p.level, "description": p.description}
                        for p in proposal.proposals
                    ],
                }
            with open(args.output, "w") as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2, default=str)
            print(f"\n报告已保存到: {args.output}")

        return result

    asyncio.run(run())


if __name__ == "__main__":
    main()
