from dataclasses import dataclass, field
from datetime import datetime
from typing import TypedDict, Optional, Any
from .alert import AlertEvent


class DiagnosisState(TypedDict, total=False):
    alert: AlertEvent
    collected_data: dict
    trace_result: dict
    root_cause: str
    evidence_chain: list[str]
    confidence: float
    repair_hypothesis: str
    hypothesis_validated: bool
    validation_detail: str
    diagnosis_report: Optional["DiagnosisReport"]
    messages: list[Any]


@dataclass
class DiagnosisReport:
    alert_id: str
    timestamp: datetime
    root_cause: str
    evidence_chain: list[str]
    confidence: float
    repair_hypothesis: str
    hypothesis_validated: bool
    validation_detail: str
    diagnosis_duration_s: float
    phases_completed: list[str] = field(default_factory=list)
