from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class OperationPlan:
    action: str
    level: int               # 1-4 safety level
    description: str
    expected_impact: str
    rollback_plan: str
    verification: str


@dataclass
class DecisionProposal:
    report_id: str
    timestamp: datetime
    proposals: list[OperationPlan] = field(default_factory=list)
    requires_confirmation: bool = False
    reasoning: str = ""
