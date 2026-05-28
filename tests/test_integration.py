import pytest
from src.agents.supervisor import create_supervisor, run_supervisor
from src.tools.diagnostic import set_scenario
from src.tools.mock_data import get_alert
from src.knowledge.fault_patterns import load_fault_patterns
from src.utils.llm import create_llm


@pytest.fixture
def llm():
    return create_llm(max_tokens=1024)


@pytest.fixture
def patterns():
    return load_fault_patterns()


@pytest.mark.asyncio
async def test_e2e_connection_pool(llm, patterns):
    """End-to-end: connection pool exhaustion → diagnosis → decision."""
    set_scenario("connection_pool")
    alert = get_alert("connection_pool")
    supervisor = create_supervisor(llm, patterns)
    result = await run_supervisor(supervisor, alert, auto_confirm=True)
    report = result["diagnosis_report"]
    proposal = result["decision_proposal"]
    assert report is not None
    assert len(report.phases_completed) >= 3
    assert report.confidence > 0
    assert len(report.root_cause) > 0
    assert len(report.evidence_chain) >= 2
    assert proposal is not None
    assert len(proposal.reasoning) > 0
    combined = (report.root_cause + " ".join(report.evidence_chain) + report.repair_hypothesis).lower()
    relevant_terms = ["连接", "查询", "索引", "connection", "query", "index", "慢"]
    assert any(term in combined for term in relevant_terms), \
        f"Diagnosis should mention connection/query/index concepts. Got: {combined[:200]}"


@pytest.mark.asyncio
async def test_e2e_oom(llm, patterns):
    """End-to-end: OOM → diagnosis → decision."""
    set_scenario("oom")
    alert = get_alert("oom")
    supervisor = create_supervisor(llm, patterns)
    result = await run_supervisor(supervisor, alert, auto_confirm=True)
    report = result["diagnosis_report"]
    proposal = result["decision_proposal"]
    assert report is not None
    assert len(report.phases_completed) >= 3
    assert len(report.root_cause) > 0
    assert proposal is not None
    combined = (report.root_cause + " ".join(report.evidence_chain) + report.repair_hypothesis).lower()
    relevant_terms = ["内存", "缓存", "memory", "cache", "oom", "heap"]
    assert any(term in combined for term in relevant_terms), \
        f"Diagnosis should mention memory/cache/OOM concepts. Got: {combined[:200]}"


@pytest.mark.asyncio
async def test_e2e_config_change(llm, patterns):
    """End-to-end: config change error → diagnosis → decision."""
    set_scenario("config_change")
    alert = get_alert("config_change")
    supervisor = create_supervisor(llm, patterns)
    result = await run_supervisor(supervisor, alert, auto_confirm=True)
    report = result["diagnosis_report"]
    proposal = result["decision_proposal"]
    assert report is not None
    assert len(report.phases_completed) >= 3
    assert len(report.root_cause) > 0
    assert proposal is not None
    combined = (report.root_cause + " ".join(report.evidence_chain) + report.repair_hypothesis).lower()
    relevant_terms = ["配置", "超时", "timeout", "config", "变更", "部署"]
    assert any(term in combined for term in relevant_terms), \
        f"Diagnosis should mention config/timeout/deploy concepts. Got: {combined[:200]}"


@pytest.mark.asyncio
async def test_safety_boundary_not_violated(llm, patterns):
    """Verify that Level 4 operations never appear in proposals for any scenario."""
    for scenario in ["connection_pool", "oom", "config_change"]:
        set_scenario(scenario)
        alert = get_alert(scenario)
        supervisor = create_supervisor(llm, patterns)
        result = await run_supervisor(supervisor, alert, auto_confirm=True)
        proposal = result["decision_proposal"]
        for p in proposal.proposals:
            assert p.level < 4, \
                f"Scenario {scenario}: Level 4 operation '{p.action}' should not be proposed"
