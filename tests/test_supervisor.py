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
async def test_supervisor_full_flow_connection_pool(llm, patterns):
    set_scenario("connection_pool")
    alert = get_alert("connection_pool")
    supervisor = create_supervisor(llm, patterns)
    result = await run_supervisor(supervisor, alert, auto_confirm=True)
    assert result["alert_id"] == "alert-001"
    assert result["diagnosis_report"] is not None
    assert result["decision_proposal"] is not None
    assert result["diagnosis_report"].confidence > 0


@pytest.mark.asyncio
async def test_supervisor_full_flow_oom(llm, patterns):
    set_scenario("oom")
    alert = get_alert("oom")
    supervisor = create_supervisor(llm, patterns)
    result = await run_supervisor(supervisor, alert, auto_confirm=True)
    assert result["diagnosis_report"] is not None
    assert result["decision_proposal"] is not None
    assert result["diagnosis_report"].alert_id == "alert-002"


@pytest.mark.asyncio
async def test_supervisor_full_flow_config_change(llm, patterns):
    set_scenario("config_change")
    alert = get_alert("config_change")
    supervisor = create_supervisor(llm, patterns)
    result = await run_supervisor(supervisor, alert, auto_confirm=True)
    assert result["diagnosis_report"] is not None
    assert result["decision_proposal"] is not None
    assert result["diagnosis_report"].alert_id == "alert-003"
