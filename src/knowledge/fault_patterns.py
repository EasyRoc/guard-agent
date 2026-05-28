# src/knowledge/fault_patterns.py
import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class FaultPattern:
    id: str
    name: str
    symptoms: list[str] = field(default_factory=list)
    typical_traces: list[str] = field(default_factory=list)
    common_causes: list[str] = field(default_factory=list)
    diagnostic_queries: list[str] = field(default_factory=list)
    repair_strategies: list[str] = field(default_factory=list)
    level: int = 1  # 1=common, 2=occasional, 3=rare


def load_fault_patterns() -> list[FaultPattern]:
    path = Path(__file__).parent / "fault_patterns.json"
    try:
        with open(path) as f:
            data = json.load(f)
        return [FaultPattern(**item) for item in data]
    except (FileNotFoundError, json.JSONDecodeError) as e:
        raise RuntimeError(f"Failed to load fault patterns from {path}: {e}") from e


def match_patterns(
    description: str,
    patterns: list[FaultPattern],
    min_hits: int = 2,
) -> list[FaultPattern]:
    """Simple keyword matching. Checks if pattern keywords (name, symptoms, causes)
    appear as substrings of the description. Works for both Chinese and English."""
    desc_lower = description.lower()
    matches = []
    for p in patterns:
        keyword_list = (
            [p.name.lower()]
            + [s.lower() for s in p.symptoms]
            + [c.lower() for c in p.common_causes]
        )
        hits = sum(1 for kw in keyword_list if kw in desc_lower)
        if hits >= min_hits:
            matches.append(p)
    return sorted(matches, key=lambda p: p.level)
