import json
from pathlib import Path
from datetime import datetime
from src.models.alert import AlertEvent


FIXTURES_DIR = Path(__file__).parent.parent.parent / "tests" / "fixtures"

SCENARIO_FILES = {
    "connection_pool": "connection_pool.json",
    "oom": "oom.json",
    "config_change": "config_change.json",
}


def _load_scenario(name: str) -> dict:
    filename = SCENARIO_FILES.get(name)
    if not filename:
        raise ValueError(f"Unknown scenario: {name}. Available: {list(SCENARIO_FILES.keys())}")
    with open(FIXTURES_DIR / filename) as f:
        return json.load(f)


def get_alert(scenario: str) -> AlertEvent:
    data = _load_scenario(scenario)["alert"]
    return AlertEvent(
        id=data["id"],
        timestamp=datetime.fromisoformat(data["timestamp"]),
        source=data["source"],
        severity=data["severity"],
        title=data["title"],
        resource=data["resource"],
        metric=data["metric"],
        current_value=data["current_value"],
        threshold=data["threshold"],
        labels=data.get("labels", {}),
        raw_data=data.get("raw_data", {}),
    )


def get_mock_metrics(scenario: str) -> dict:
    return _load_scenario(scenario).get("mock_metrics", {})


def get_mock_logs(scenario: str) -> list[str]:
    return _load_scenario(scenario).get("mock_logs", [])


def get_mock_traces(scenario: str) -> list[str]:
    return _load_scenario(scenario).get("mock_traces", [])


def get_mock_change_events(scenario: str) -> list[dict]:
    return _load_scenario(scenario).get("mock_change_events", [])


def get_mock_extra(scenario: str, key: str) -> dict | list:
    return _load_scenario(scenario).get(key, {})
