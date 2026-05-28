import pytest
from src.agents.diagnosis import create_diagnosis_agent, run_diagnosis
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
async def test_diagnosis_connection_pool(llm, patterns):
    set_scenario("connection_pool")
    alert = get_alert("connection_pool")
    agent = create_diagnosis_agent(llm, patterns)
    report = await run_diagnosis(agent, alert)
    assert report.alert_id == "alert-001"
    assert "phase1" in report.phases_completed
    assert "phase2" in report.phases_completed
    assert report.confidence >= 0.0
    assert len(report.evidence_chain) > 0
    assert len(report.root_cause) > 0
    assert len(report.repair_hypothesis) > 0


@pytest.mark.asyncio
async def test_diagnosis_oom(llm, patterns):
    set_scenario("oom")
    alert = get_alert("oom")
    agent = create_diagnosis_agent(llm, patterns)
    report = await run_diagnosis(agent, alert)
    assert report.alert_id == "alert-002"
    assert "phase1" in report.phases_completed
    assert len(report.evidence_chain) > 0
    assert len(report.root_cause) > 0


@pytest.mark.asyncio
async def test_diagnosis_config_change(llm, patterns):
    set_scenario("config_change")
    alert = get_alert("config_change")
    agent = create_diagnosis_agent(llm, patterns)
    report = await run_diagnosis(agent, alert)
    assert report.alert_id == "alert-003"
    assert "phase1" in report.phases_completed
    assert len(report.evidence_chain) > 0
    assert len(report.root_cause) > 0


@pytest.mark.asyncio
async def test_diagnosis_phases_are_sequential(llm, patterns):
    set_scenario("connection_pool")
    alert = get_alert("connection_pool")
    agent = create_diagnosis_agent(llm, patterns)
    report = await run_diagnosis(agent, alert)
    expected_order = ["phase1", "phase2", "phase3"]
    indices = [report.phases_completed.index(p) if p in report.phases_completed else 999
               for p in expected_order]
    assert indices == sorted(indices), f"Phases out of order: {report.phases_completed}"
