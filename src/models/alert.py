from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class AlertEvent:
    id: str
    timestamp: datetime
    source: str              # "prometheus" | "elk" | "kafka"
    severity: str            # "critical" | "warning" | "info"
    title: str
    resource: str
    metric: str
    current_value: float
    threshold: float
    labels: dict = field(default_factory=dict)
    raw_data: dict = field(default_factory=dict)
