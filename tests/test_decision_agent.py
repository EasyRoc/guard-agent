import pytest
from datetime import datetime
from src.agents.decision import run_decision
from src.models.diagnosis import DiagnosisReport
from src.models.decision import DecisionProposal
from src.utils.llm import create_llm


def make_report(
    root_cause: str = "缺少索引导致全表扫描",
    repair_hypothesis: str = "在orders表的status字段上添加索引",
    confidence: float = 0.9,
    hypothesis_validated: bool = True,
) -> DiagnosisReport:
    return DiagnosisReport(
        alert_id="alert-001",
        timestamp=datetime.now(),
        root_cause=root_cause,
        evidence_chain=["证据1", "证据2", "证据3"],
        confidence=confidence,
        repair_hypothesis=repair_hypothesis,
        hypothesis_validated=hypothesis_validated,
        validation_detail="EXPLAIN验证通过",
        diagnosis_duration_s=45.0,
        phases_completed=["phase1", "phase2", "phase3", "phase4"],
    )


def test_decision_level1_auto_approve():
    """Read-only operations should be auto-approved."""
    report = make_report(
        root_cause="连接数异常",
        repair_hypothesis="需要查询metrics和logs进一步分析",
    )
    proposal = run_decision(report)
    assert isinstance(proposal, DecisionProposal)
    assert not proposal.requires_confirmation or len(proposal.proposals) == 0


def test_decision_level2_requires_confirmation():
    """add_index is Level 2, should require human confirmation."""
    report = make_report(
        repair_hypothesis="在orders表的status字段上添加索引 CREATE INDEX idx_orders_status ON orders(status)"
    )
    proposal = run_decision(report)
    assert proposal.requires_confirmation
    assert len(proposal.proposals) > 0
    for p in proposal.proposals:
        assert p.level in (1, 2, 3), f"Unexpected level: {p.level}"


def test_decision_level4_rejected():
    """DROP operations should be rejected."""
    report = make_report(
        repair_hypothesis="需要删除orders表重建: DROP TABLE orders"
    )
    proposal = run_decision(report)
    for p in proposal.proposals:
        assert p.level < 4, f"Level 4 operation leaked: {p.action}"


def test_decision_restart_pod_level2():
    """Restart pod is Level 2."""
    report = make_report(
        repair_hypothesis="需要重启Pod释放内存: restart_pod recommendation-svc-pod-3"
    )
    proposal = run_decision(report)
    assert len(proposal.proposals) > 0
    assert proposal.requires_confirmation


def test_decision_config_rollback_level3():
    """Config rollback is Level 3."""
    report = make_report(
        repair_hypothesis="需要回滚payment-gateway配置: deploy_rollback to v3.1.0"
    )
    proposal = run_decision(report)
    assert len(proposal.proposals) > 0
    assert proposal.requires_confirmation
    assert any(p.level == 3 for p in proposal.proposals)
