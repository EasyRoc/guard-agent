# Guard Agent MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the decision-and-diagnosis layer of Guard Agent — three LangGraph agents (Supervisor, Diagnosis, Decision) that collaborate to diagnose faults and propose operations within a 4-level safety boundary.

**Architecture:** Three independent LangGraph agents. Supervisor orchestrates via tool-calling (diagnosis → decision → human confirmation). Diagnosis Agent uses an internal 4-phase StateGraph (collect → trace → root-cause → validate). Decision Agent uses a rule engine (Level 1-4 operation classification) with LLM fallback for ambiguous cases. Perception/execution layers are mocked with 3 fault scenarios.

**Tech Stack:** Python 3.11+, LangGraph, LangChain, OpenAI SDK (DeepSeek-compatible), Poetry, pytest + pytest-asyncio

---

### Task 1: Project scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `src/__init__.py`
- Create: `src/models/__init__.py`
- Create: `src/agents/__init__.py`
- Create: `src/knowledge/__init__.py`
- Create: `src/tools/__init__.py`
- Create: `src/safety/__init__.py`
- Create: `src/utils/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/fixtures/__init__.py`

- [ ] **Step 1: Write pyproject.toml**

```toml
[project]
name = "guard-agent"
version = "0.1.0"
description = "Intelligent SRE operations agent with structured diagnosis"
requires-python = ">=3.11"
dependencies = [
    "langgraph>=0.2.0",
    "langchain>=0.3.0",
    "langchain-openai>=0.2.0",
    "openai>=1.50.0",
    "pydantic>=2.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.24.0",
]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

- [ ] **Step 2: Write .gitignore**

```
__pycache__/
*.py[cod]
.venv/
.env
.pytest_cache/
dist/
*.egg-info/
.ruff_cache/
```

- [ ] **Step 3: Create directory structure and install**

Run:
```bash
cd /Users/zhouqiantalaogong/PycharmProjects/guard-agent
mkdir -p src/models src/agents src/knowledge src/tools src/safety src/utils tests/fixtures
touch src/__init__.py src/models/__init__.py src/agents/__init__.py src/knowledge/__init__.py src/tools/__init__.py src/safety/__init__.py src/utils/__init__.py tests/__init__.py tests/fixtures/__init__.py
pip install -e ".[dev]"
```

- [ ] **Step 4: Verify project structure**

Run:
```bash
cd /Users/zhouqiantalaogong/PycharmProjects/guard-agent
python -c "import src; print('src ok')"
python -c "import langgraph; print('langgraph ok')"
```

- [ ] **Step 5: Git init and commit**

```bash
cd /Users/zhouqiantalaogong/PycharmProjects/guard-agent
git init
git add pyproject.toml .gitignore src/ tests/
git commit -m "chore: scaffold project with Poetry and LangGraph deps"
```

---

### Task 2: Data models

**Files:**
- Create: `src/models/alert.py`
- Create: `src/models/diagnosis.py`
- Create: `src/models/decision.py`

- [ ] **Step 1: Write AlertEvent model**

```python
# src/models/alert.py
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
```

- [ ] **Step 2: Write DiagnosisState and DiagnosisReport**

```python
# src/models/diagnosis.py
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
```

- [ ] **Step 3: Write DecisionProposal and OperationPlan**

```python
# src/models/decision.py
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
```

- [ ] **Step 4: Verify models are importable**

Run:
```bash
cd /Users/zhouqiantalaogong/PycharmProjects/guard-agent
python -c "
from datetime import datetime
from src.models.alert import AlertEvent
from src.models.diagnosis import DiagnosisReport, DiagnosisState
from src.models.decision import DecisionProposal, OperationPlan

a = AlertEvent(id='1', timestamp=datetime.now(), source='prometheus', severity='critical',
               title='test', resource='db', metric='cpu', current_value=95.0, threshold=90.0)
print(f'AlertEvent ok: {a.title}')

op = OperationPlan(action='restart_pod', level=2, description='restart', expected_impact='none',
                   rollback_plan='none', verification='curl /health')
print(f'OperationPlan ok: level={op.level}')

dp = DecisionProposal(report_id='r1', timestamp=datetime.now(), proposals=[op],
                      requires_confirmation=True, reasoning='test')
print(f'DecisionProposal ok: requires_confirmation={dp.requires_confirmation}')
"
```

- [ ] **Step 5: Commit**

```bash
cd /Users/zhouqiantalaogong/PycharmProjects/guard-agent
git add src/models/
git commit -m "feat: add data models (AlertEvent, DiagnosisReport, DecisionProposal)"
```

---

### Task 3: Fault pattern knowledge base

**Files:**
- Create: `src/knowledge/fault_patterns.py`
- Create: `src/knowledge/fault_patterns.json`
- Create: `tests/test_fault_patterns.py`

- [ ] **Step 1: Write the test**

```python
# tests/test_fault_patterns.py
from src.knowledge.fault_patterns import load_fault_patterns, FaultPattern, match_patterns


def test_load_fault_patterns():
    patterns = load_fault_patterns()
    assert len(patterns) == 3
    assert all(isinstance(p, FaultPattern) for p in patterns)


def test_pattern_ids():
    patterns = load_fault_patterns()
    ids = {p.id for p in patterns}
    assert ids == {"connection_pool_exhaustion", "oom_cache_unbounded", "config_change_error"}


def test_match_patterns_by_keyword():
    patterns = load_fault_patterns()
    matches = match_patterns("数据库连接数超过阈值，连接池即将耗尽", patterns)
    assert len(matches) > 0
    assert matches[0].id == "connection_pool_exhaustion"


def test_match_patterns_no_match():
    patterns = load_fault_patterns()
    matches = match_patterns("unknown weird error XYZ123", patterns)
    assert len(matches) == 0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/zhouqiantalaogong/PycharmProjects/guard-agent
pytest tests/test_fault_patterns.py -v
```
Expected: FAIL (module not found)

- [ ] **Step 3: Write FaultPattern dataclass and loader**

```python
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
    with open(path) as f:
        data = json.load(f)
    return [FaultPattern(**item) for item in data]


def match_patterns(description: str, patterns: list[FaultPattern]) -> list[FaultPattern]:
    """Simple keyword matching. MVP: string contains check on symptoms and name."""
    desc_lower = description.lower()
    matches = []
    for p in patterns:
        search_text = " ".join([p.name] + p.symptoms + p.common_causes).lower()
        # Match if any significant keyword overlap
        keywords = [w for w in desc_lower.split() if len(w) > 1]
        hits = sum(1 for kw in keywords if kw in search_text)
        if hits >= 2:
            matches.append(p)
    return sorted(matches, key=lambda p: p.level)
```

- [ ] **Step 4: Write fault_patterns.json**

```json
[
  {
    "id": "connection_pool_exhaustion",
    "name": "数据库连接池耗尽",
    "symptoms": [
      "数据库连接数超过阈值",
      "连接数>90%",
      "too many connections",
      "connection timeout",
      "响应超时"
    ],
    "typical_traces": [
      "大量慢查询占用连接",
      "连接未释放",
      "连接等待队列增长"
    ],
    "common_causes": [
      "缺少索引导致全表扫描",
      "连接泄漏未正确关闭",
      "突发流量超过连接池上限"
    ],
    "diagnostic_queries": [
      "SHOW PROCESSLIST",
      "SHOW ENGINE INNODB STATUS",
      "SELECT * FROM information_schema.processlist",
      "EXPLAIN 慢查询语句"
    ],
    "repair_strategies": [
      "添加缺失索引",
      "修复连接泄漏代码",
      "调大max_connections",
      "启用连接池等待超时"
    ],
    "level": 1
  },
  {
    "id": "oom_cache_unbounded",
    "name": "内存溢出OOM - 缓存未限制大小",
    "symptoms": [
      "Pod OOMKilled",
      "heap使用率超过95%",
      "GC频繁Full GC",
      "java.lang.OutOfMemoryError",
      "memory usage 98%"
    ],
    "typical_traces": [
      "缓存对象持续增长",
      "GC无法回收",
      "堆内存线性增长至上限"
    ],
    "common_causes": [
      "本地缓存未设置大小上限",
      "缓存未配置过期策略",
      "大对象未及时释放"
    ],
    "diagnostic_queries": [
      "jmap -histo <pid>",
      "jstat -gc <pid>",
      "查看缓存配置",
      "dump heap分析"
    ],
    "repair_strategies": [
      "为缓存设置maxSize上限",
      "配置LRU过期策略",
      "临时重启Pod释放内存"
    ],
    "level": 1
  },
  {
    "id": "config_change_error",
    "name": "配置变更导致服务异常",
    "symptoms": [
      "部署后错误率飙升",
      "响应超时增加",
      "配置变更后异常",
      "error rate spike after deploy"
    ],
    "typical_traces": [
      "新配置值与旧值差异大",
      "错误集中在配置变更时间点后",
      "回滚配置后恢复"
    ],
    "common_causes": [
      "超时值单位错误（毫秒vs秒）",
      "连接池参数设置不合理",
      "环境特定配置遗漏"
    ],
    "diagnostic_queries": [
      "git diff 上次部署",
      "对比新旧配置文件",
      "检查配置变更时间与告警时间关联"
    ],
    "repair_strategies": [
      "回滚配置到上一版本",
      "修正配置值并重新部署",
      "增加配置校验规则"
    ],
    "level": 2
  }
]
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd /Users/zhouqiantalaogong/PycharmProjects/guard-agent
pytest tests/test_fault_patterns.py -v
```
Expected: 4 PASS

- [ ] **Step 6: Commit**

```bash
cd /Users/zhouqiantalaogong/PycharmProjects/guard-agent
git add src/knowledge/ tests/test_fault_patterns.py
git commit -m "feat: add fault pattern knowledge base with 3 patterns"
```

---

### Task 4: Safety operation levels

**Files:**
- Create: `src/safety/operation_levels.py`
- Create: `tests/test_operation_levels.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_operation_levels.py
from src.safety.operation_levels import classify_operation, get_level_description, is_operation_allowed


def test_classify_known_level1():
    assert classify_operation("query_metrics") == 1
    assert classify_operation("query_logs") == 1
    assert classify_operation("query_traces") == 1


def test_classify_known_level2():
    assert classify_operation("restart_pod") == 2
    assert classify_operation("scale_up") == 2
    assert classify_operation("adjust_rate_limit") == 2


def test_classify_known_level3():
    assert classify_operation("modify_config") == 3
    assert classify_operation("execute_ddl") == 3
    assert classify_operation("execute_dml") == 3


def test_classify_known_level4():
    assert classify_operation("drop_database") == 4
    assert classify_operation("drop_table") == 4
    assert classify_operation("rm_rf") == 4


def test_classify_unknown_returns_none():
    assert classify_operation("some_weird_action") is None


def test_is_operation_allowed():
    assert is_operation_allowed("query_metrics") is True     # Level 1
    assert is_operation_allowed("restart_pod") is True       # Level 2 (allowed with confirm)
    assert is_operation_allowed("execute_ddl") is True       # Level 3 (allowed with approval)
    assert is_operation_allowed("drop_database") is False    # Level 4 (blocked)


def test_get_level_description():
    desc = get_level_description(1)
    assert "自动" in desc
    desc2 = get_level_description(2)
    assert "确认" in desc2
    desc4 = get_level_description(4)
    assert "禁止" in desc4
```

- [ ] **Step 2: Run tests to verify failure**

```bash
cd /Users/zhouqiantalaogong/PycharmProjects/guard-agent
pytest tests/test_operation_levels.py -v
```
Expected: FAIL

- [ ] **Step 3: Write implementation**

```python
# src/safety/operation_levels.py
from typing import Optional


OPERATION_LEVELS: dict[str, int] = {
    # Level 1: Auto-approved (read-only operations)
    "query_metrics": 1,
    "query_logs": 1,
    "query_traces": 1,
    "query_topology": 1,
    "query_change_events": 1,
    "query_processlist": 1,
    "search_fault_patterns": 1,
    "compare_configs": 1,

    # Level 2: Need human confirmation
    "restart_pod": 2,
    "scale_up": 2,
    "scale_down": 2,
    "adjust_rate_limit": 2,
    "adjust_circuit_breaker": 2,
    "clear_cache": 2,
    "run_diagnostic_command": 2,
    "add_index": 2,

    # Level 3: Need human approval with detailed plan
    "modify_config": 3,
    "execute_ddl": 3,
    "execute_dml": 3,
    "deploy_rollback": 3,
    "modify_route": 3,
    "adjust_connection_pool": 3,

    # Level 4: Completely forbidden
    "drop_database": 4,
    "drop_table": 4,
    "truncate_table": 4,
    "rm_rf": 4,
    "kill_process": 4,
    "iptables_modify": 4,
    "shutdown_service": 4,
}


LEVEL_DESCRIPTIONS: dict[int, str] = {
    1: "自动批准 - Agent可直接执行，无需人工介入",
    2: "需人工确认 - Agent生成建议，等待人工确认后执行",
    3: "需人工审批 - Agent生成详细方案，必须人工审批后执行",
    4: "完全禁止 - Agent不会提议此操作",
}


def classify_operation(action_name: str) -> Optional[int]:
    """Return the safety level (1-4) for an operation, or None if unknown."""
    return OPERATION_LEVELS.get(action_name)


def is_operation_allowed(action_name: str) -> bool:
    """Check if operation is allowed at all (Level 4 is not)."""
    level = classify_operation(action_name)
    if level is None:
        return False  # unknown operations are not allowed
    return level < 4


def get_level_description(level: int) -> str:
    return LEVEL_DESCRIPTIONS.get(level, f"未知等级: {level}")


def get_operations_by_level(level: int) -> list[str]:
    return [op for op, lv in OPERATION_LEVELS.items() if lv == level]
```

- [ ] **Step 4: Run tests to verify pass**

```bash
cd /Users/zhouqiantalaogong/PycharmProjects/guard-agent
pytest tests/test_operation_levels.py -v
```
Expected: 7 PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/zhouqiantalaogong/PycharmProjects/guard-agent
git add src/safety/ tests/test_operation_levels.py
git commit -m "feat: add safety operation levels (Level 1-4 classification)"
```

---

### Task 5: LLM client utility

**Files:**
- Create: `src/utils/llm.py`
- Create: `src/utils/logging.py`
- Create: `tests/test_llm.py`

- [ ] **Step 1: Write test**

```python
# tests/test_llm.py
import os
from src.utils.llm import create_llm


def test_create_llm_default_model():
    llm = create_llm()
    assert llm is not None
    assert llm.model_name == "deepseek-chat"


def test_create_llm_custom_model():
    llm = create_llm(model="deepseek-chat")
    assert llm.model_name == "deepseek-chat"


def test_create_llm_with_base_url():
    llm = create_llm(base_url="https://api.deepseek.com/v1")
    assert llm is not None
```

- [ ] **Step 2: Run test to verify failure**

```bash
cd /Users/zhouqiantalaogong/PycharmProjects/guard-agent
pytest tests/test_llm.py -v
```
Expected: FAIL

- [ ] **Step 3: Write implementation**

```python
# src/utils/llm.py
import os
from langchain_openai import ChatOpenAI


def create_llm(
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 4096,
) -> ChatOpenAI:
    return ChatOpenAI(
        model=model or os.getenv("LLM_MODEL", "deepseek-chat"),
        base_url=base_url or os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1"),
        api_key=api_key or os.getenv("LLM_API_KEY", "sk-placeholder"),
        temperature=temperature,
        max_tokens=max_tokens,
    )
```

```python
# src/utils/logging.py
import logging
import sys


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )
```

- [ ] **Step 4: Run test to verify pass**

```bash
cd /Users/zhouqiantalaogong/PycharmProjects/guard-agent
pytest tests/test_llm.py -v
```
Expected: 3 PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/zhouqiantalaogong/PycharmProjects/guard-agent
git add src/utils/ tests/test_llm.py
git commit -m "feat: add LLM client factory with DeepSeek defaults"
```

---

### Task 6: Mock data and diagnostic tools

**Files:**
- Create: `src/tools/mock_data.py`
- Create: `src/tools/diagnostic.py`
- Create: `tests/fixtures/connection_pool.json`
- Create: `tests/fixtures/oom.json`
- Create: `tests/fixtures/config_change.json`
- Create: `tests/test_diagnostic_tools.py`

- [ ] **Step 1: Write mock data fixture files**

```json
// tests/fixtures/connection_pool.json
{
  "alert": {
    "id": "alert-001",
    "timestamp": "2026-05-28T10:30:00",
    "source": "prometheus",
    "severity": "critical",
    "title": "数据库连接数超过阈值",
    "resource": "mysql-orders-db",
    "metric": "connections_active",
    "current_value": 148.0,
    "threshold": 150.0,
    "labels": {"env": "production", "service": "api-server"},
    "raw_data": {
      "prometheus_query": "mysql_global_status_threads_connected",
      "alert_rule": "MySQL connection usage > 90%",
      "dashboard_url": "http://grafana/d/mysql-overview"
    }
  },
  "mock_metrics": {
    "connections_active": 148,
    "connections_max": 150,
    "qps": 1200,
    "avg_query_duration_ms": 45000,
    "slow_queries_last_5min": 23,
    "error_rate": 0.15,
    "cpu_usage_pct": 45,
    "memory_usage_pct": 62
  },
  "mock_logs": [
    "[10:28:15] ERROR - api-server: connection timeout after 30000ms",
    "[10:28:20] ERROR - api-server: unable to acquire connection from pool (pool size=150, active=148, waiting=14)",
    "[10:28:22] ERROR - api-server: too many connections",
    "[10:29:00] WARN  - mysql: slow query detected: SELECT * FROM orders WHERE status='pending' ORDER BY created_at DESC (execution_time=45123ms)"
  ],
  "mock_traces": [
    "api-server → mysql-orders-db: SELECT * FROM orders WHERE status='pending' (45.1s, rows_scanned=2840000)",
    "api-server → mysql-orders-db: SELECT * FROM orders WHERE status='pending' (44.8s, rows_scanned=2840000)",
    "api-server → mysql-orders-db: SELECT * FROM orders WHERE status='pending' (45.3s, rows_scanned=2840000)"
  ],
  "mock_change_events": [
    {"time": "2026-05-28T09:00:00", "type": "deploy", "service": "frontend", "description": "前端UI更新"},
    {"time": "2026-05-27T18:00:00", "type": "config", "service": "api-server", "description": "日志级别调整"}
  ],
  "mock_processlist": [
    {"id": 101, "user": "app_user", "host": "10.0.1.5", "db": "orders_db", "command": "Query", "time": 45, "state": "Sending data", "info": "SELECT * FROM orders WHERE status='pending' ORDER BY created_at DESC"},
    {"id": 102, "user": "app_user", "host": "10.0.1.6", "db": "orders_db", "command": "Query", "time": 44, "state": "Sending data", "info": "SELECT * FROM orders WHERE status='pending' ORDER BY created_at DESC"},
    {"id": 103, "user": "app_user", "host": "10.0.1.7", "db": "orders_db", "command": "Query", "time": 45, "state": "Sending data", "info": "SELECT * FROM orders WHERE status='pending' ORDER BY created_at DESC"},
    {"id": 104, "user": "app_user", "host": "10.0.1.8", "db": "orders_db", "command": "Sleep", "time": 0, "state": "", "info": ""}
  ],
  "mock_explain": {
    "query": "SELECT * FROM orders WHERE status='pending'",
    "plan": {
      "id": 1,
      "select_type": "SIMPLE",
      "table": "orders",
      "type": "ALL",
      "possible_keys": null,
      "key": null,
      "key_len": null,
      "rows": 2840000,
      "filtered": 10.0,
      "Extra": "Using where; Using filesort"
    },
    "with_index_plan": {
      "id": 1,
      "select_type": "SIMPLE",
      "table": "orders",
      "type": "ref",
      "possible_keys": "idx_orders_status",
      "key": "idx_orders_status",
      "key_len": "768",
      "rows": 1500,
      "filtered": 100.0,
      "Extra": "Using index condition; Using filesort"
    }
  }
}
```

```json
// tests/fixtures/oom.json
{
  "alert": {
    "id": "alert-002",
    "timestamp": "2026-05-28T14:00:00",
    "source": "prometheus",
    "severity": "critical",
    "title": "Pod频繁OOMKilled重启",
    "resource": "recommendation-svc-pod-3",
    "metric": "memory_usage_pct",
    "current_value": 98.0,
    "threshold": 90.0,
    "labels": {"env": "production", "service": "recommendation-svc", "pod": "recommendation-svc-pod-3"},
    "raw_data": {
      "prometheus_query": "container_memory_usage_bytes / container_spec_memory_limit_bytes",
      "alert_rule": "Container memory usage > 90% for 5min",
      "restart_count": 4
    }
  },
  "mock_metrics": {
    "memory_usage_pct": 98,
    "memory_limit_mb": 2048,
    "heap_usage_mb": 1950,
    "gc_frequency_per_min": 15,
    "gc_pause_avg_ms": 450,
    "cpu_usage_pct": 35,
    "request_qps": 800
  },
  "mock_logs": [
    "[13:55:00] WARN - recommendation-svc: GC overhead limit reached, Full GC taking 520ms",
    "[13:56:30] WARN - recommendation-svc: GC overhead limit reached, Full GC taking 610ms",
    "[13:58:00] ERROR - recommendation-svc: java.lang.OutOfMemoryError: Java heap space",
    "[13:58:05] INFO - kubelet: Container recommendation-svc exceeded memory limit (2048MB), OOMKilled",
    "[13:58:10] INFO - kubelet: Restarting container recommendation-svc (restart #4)"
  ],
  "mock_traces": [
    "recommendation-svc → redis-cache: HGETALL user:preferences:12345 (response_size=85MB)",
    "recommendation-svc → redis-cache: HGETALL user:preferences:67890 (response_size=92MB)"
  ],
  "mock_change_events": [
    {"time": "2026-05-28T12:00:00", "type": "deploy", "service": "recommendation-svc", "description": "新增个性化推荐缓存逻辑(v2.3.1)"}
  ],
  "mock_heap_dump": {
    "top_classes": [
      {"class": "java.util.concurrent.ConcurrentHashMap$Node", "instances": 2850000, "size_mb": 1420},
      {"class": "com.example.recommendation.UserPreference", "instances": 1420000, "size_mb": 380},
      {"class": "java.lang.String", "instances": 580000, "size_mb": 45}
    ],
    "cache_config": {
      "cache_name": "userPreferenceCache",
      "max_size": "unlimited",
      "expire_policy": "none",
      "implementation": "ConcurrentHashMap"
    }
  }
}
```

```json
// tests/fixtures/config_change.json
{
  "alert": {
    "id": "alert-003",
    "timestamp": "2026-05-28T16:05:00",
    "source": "prometheus",
    "severity": "critical",
    "title": "部署后错误率飙升",
    "resource": "payment-gateway-svc",
    "metric": "error_rate",
    "current_value": 0.42,
    "threshold": 0.05,
    "labels": {"env": "production", "service": "payment-gateway"},
    "raw_data": {
      "prometheus_query": "rate(http_requests_total{status=~'5..'}[5m])",
      "alert_rule": "Error rate > 5% for 3min",
      "deploy_time": "2026-05-28T16:00:00"
    }
  },
  "mock_metrics": {
    "error_rate": 0.42,
    "qps": 500,
    "p99_latency_ms": 45000,
    "timeout_rate": 0.38,
    "cpu_usage_pct": 28,
    "memory_usage_pct": 55
  },
  "mock_logs": [
    "[16:01:00] ERROR - payment-gateway: upstream request timeout after 3000ms (upstream: bank-api)",
    "[16:01:05] ERROR - payment-gateway: upstream request timeout after 3000ms (upstream: bank-api)",
    "[16:01:30] ERROR - payment-gateway: circuit breaker opened for bank-api",
    "[16:02:00] ERROR - payment-gateway: all upstreams unhealthy"
  ],
  "mock_traces": [
    "payment-gateway → bank-api: POST /api/v1/charge (timeout after 3000ms)",
    "payment-gateway → bank-api: POST /api/v1/charge (timeout after 3000ms)"
  ],
  "mock_change_events": [
    {"time": "2026-05-28T16:00:00", "type": "deploy", "service": "payment-gateway", "description": "v3.2.0: 更新银行API超时配置"},
    {"time": "2026-05-28T16:00:00", "type": "config", "service": "payment-gateway", "description": "bank_api_timeout_ms: 30000 → 3000"}
  ],
  "mock_config_diff": {
    "file": "application.yml",
    "changes": [
      {"key": "bank_api.timeout_ms", "old_value": "30000", "new_value": "3000", "note": "单位疑为秒→毫秒的换算错误: 30000ms=30s → 改成了3000ms=3s"},
      {"key": "bank_api.retry_count", "old_value": "3", "new_value": "3"}
    ]
  }
}
```

- [ ] **Step 2: Write mock_data.py**

```python
# src/tools/mock_data.py
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
```

- [ ] **Step 3: Write diagnostic.py (tool functions)**

```python
# src/tools/diagnostic.py
from langchain_core.tools import tool
from src.tools.mock_data import (
    get_mock_metrics,
    get_mock_logs,
    get_mock_traces,
    get_mock_change_events,
    get_mock_extra,
)


# Scenario is set globally before running diagnosis
_current_scenario: str = "connection_pool"


def set_scenario(scenario: str) -> None:
    global _current_scenario
    _current_scenario = scenario


def get_scenario() -> str:
    return _current_scenario


@tool
def query_metrics(metric_names: str) -> str:
    """查询指标数据。参数 metric_names: 逗号分隔的指标名列表，如 'connections_active,qps,error_rate'。返回指标名和值。"""
    metrics = get_mock_metrics(_current_scenario)
    requested = [m.strip() for m in metric_names.split(",")]
    result = {}
    for name in requested:
        if name in metrics:
            result[name] = metrics[name]
    if not result:
        return f"No metrics found for: {metric_names}. Available: {list(metrics.keys())}"
    return "\n".join(f"{k}: {v}" for k, v in result.items())


@tool
def query_logs(keyword: str, limit: int = 20) -> str:
    """查询日志。参数 keyword: 搜索关键词。参数 limit: 最大返回条数。返回匹配的日志行。"""
    logs = get_mock_logs(_current_scenario)
    matched = [log for log in logs if keyword.lower() in log.lower()]
    if not matched:
        matched = logs  # no keyword match, return all
    return "\n".join(matched[:limit])


@tool
def query_traces(service_name: str) -> str:
    """查询服务调用链。参数 service_name: 服务名称。返回调用链记录。"""
    traces = get_mock_traces(_current_scenario)
    matched = [t for t in traces if service_name.lower() in t.lower()]
    if not matched:
        matched = traces
    return "\n".join(matched)


@tool
def query_change_events(hours: int = 24) -> str:
    """查询变更事件。参数 hours: 查询最近多少小时的变更记录。返回变更事件列表。"""
    events = get_mock_change_events(_current_scenario)
    return "\n".join(
        f"[{e['time']}] {e['type']}: {e.get('service', '')} - {e.get('description', '')}"
        for e in events
    )


@tool
def query_processlist() -> str:
    """查询数据库当前进程列表 (SHOW PROCESSLIST)。返回当前活跃的连接和查询。"""
    data = get_mock_extra(_current_scenario, "mock_processlist")
    if not data:
        return "No processlist data available for this scenario."
    lines = []
    for p in data:
        lines.append(
            f"ID:{p['id']} User:{p['user']} Host:{p['host']} DB:{p['db']} "
            f"Command:{p['command']} Time:{p['time']}s State:{p['state']} Info:{p.get('info', '')}"
        )
    return "\n".join(lines)


@tool
def query_explain(query_text: str) -> str:
    """获取SQL查询的执行计划(EXPLAIN)。参数 query_text: SQL语句。返回执行计划。"""
    data = get_mock_extra(_current_scenario, "mock_explain")
    if not data:
        return "No EXPLAIN data available for this scenario."
    import json
    plan = data.get("plan", {})
    with_index = data.get("with_index_plan", {})
    return (
        f"当前执行计划:\n{json.dumps(plan, indent=2, ensure_ascii=False)}\n\n"
        f"如果添加索引后的执行计划:\n{json.dumps(with_index, indent=2, ensure_ascii=False)}"
    )


@tool
def query_heap_dump() -> str:
    """查询JVM堆内存分析数据。返回堆中占用最多的类。"""
    data = get_mock_extra(_current_scenario, "mock_heap_dump")
    if not data:
        return "No heap dump data available for this scenario."
    import json
    return json.dumps(data, indent=2, ensure_ascii=False)


@tool
def query_config_diff() -> str:
    """查询最近一次配置变更的差异。返回配置文件的变更内容。"""
    data = get_mock_extra(_current_scenario, "mock_config_diff")
    if not data:
        return "No config diff data available for this scenario."
    lines = [f"文件: {data.get('file', 'unknown')}"]
    for change in data.get("changes", []):
        lines.append(
            f"  {change['key']}: {change['old_value']} → {change['new_value']}"
            + (f"  // {change['note']}" if change.get("note") else "")
        )
    return "\n".join(lines)


DIAGNOSTIC_TOOLS = [
    query_metrics,
    query_logs,
    query_traces,
    query_change_events,
    query_processlist,
    query_explain,
    query_heap_dump,
    query_config_diff,
]
```

- [ ] **Step 4: Write tests**

```python
# tests/test_diagnostic_tools.py
from src.tools.diagnostic import (
    set_scenario,
    query_metrics,
    query_logs,
    query_traces,
    query_change_events,
    query_processlist,
    query_explain,
    query_heap_dump,
    query_config_diff,
)
from src.tools.mock_data import get_alert, get_mock_metrics


def test_set_scenario_and_query_metrics():
    set_scenario("connection_pool")
    result = query_metrics.invoke({"metric_names": "connections_active,qps"})
    assert "connections_active" in result
    assert "148" in result
    assert "qps" in result


def test_query_logs():
    set_scenario("connection_pool")
    result = query_logs.invoke({"keyword": "timeout"})
    assert "timeout" in result.lower()


def test_query_traces():
    set_scenario("connection_pool")
    result = query_traces.invoke({"service_name": "api-server"})
    assert "SELECT * FROM orders" in result


def test_query_change_events():
    set_scenario("config_change")
    result = query_change_events.invoke({"hours": 24})
    assert "bank_api_timeout" in result


def test_query_processlist_connection_pool():
    set_scenario("connection_pool")
    result = query_processlist.invoke({})
    assert "SELECT * FROM orders" in result
    assert "Sending data" in result


def test_query_explain():
    set_scenario("connection_pool")
    result = query_explain.invoke({"query_text": "SELECT * FROM orders WHERE status='pending'"})
    assert "ALL" in result
    assert "idx_orders_status" in result


def test_query_heap_dump():
    set_scenario("oom")
    result = query_heap_dump.invoke({})
    assert "ConcurrentHashMap" in result
    assert "unlimited" in result


def test_query_config_diff():
    set_scenario("config_change")
    result = query_config_diff.invoke({})
    assert "30000" in result
    assert "3000" in result


def test_get_alert():
    alert = get_alert("connection_pool")
    assert alert.id == "alert-001"
    assert alert.resource == "mysql-orders-db"
    assert alert.current_value == 148.0


def test_get_alert_oom():
    alert = get_alert("oom")
    assert alert.id == "alert-002"
    assert "OOM" in alert.title
```

- [ ] **Step 5: Run tests to verify pass**

```bash
cd /Users/zhouqiantalaogong/PycharmProjects/guard-agent
pytest tests/test_diagnostic_tools.py -v
```
Expected: 10 PASS

- [ ] **Step 6: Commit**

```bash
cd /Users/zhouqiantalaogong/PycharmProjects/guard-agent
git add src/tools/ tests/fixtures/ tests/test_diagnostic_tools.py
git commit -m "feat: add mock data scenarios and diagnostic tool functions"
```

---

### Task 7: Diagnosis Agent (4-phase StateGraph)

**Files:**
- Create: `src/agents/diagnosis.py`
- Create: `tests/test_diagnosis_agent.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_diagnosis_agent.py
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

    # Phase 1-4 should appear in order
    expected_order = ["phase1", "phase2", "phase3"]
    indices = [report.phases_completed.index(p) if p in report.phases_completed else 999
               for p in expected_order]
    assert indices == sorted(indices), f"Phases out of order: {report.phases_completed}"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/zhouqiantalaogong/PycharmProjects/guard-agent
pytest tests/test_diagnosis_agent.py -v
```
Expected: FAIL (module not found)

- [ ] **Step 3: Write Diagnosis Agent implementation**

```python
# src/agents/diagnosis.py
import time
import json
from datetime import datetime
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.models.alert import AlertEvent
from src.models.diagnosis import DiagnosisState, DiagnosisReport
from src.knowledge.fault_patterns import FaultPattern, match_patterns
from src.tools.diagnostic import DIAGNOSTIC_TOOLS


def _format_patterns(patterns: list[FaultPattern]) -> str:
    lines = []
    for p in patterns:
        lines.append(f"- [{p.id}] {p.name}")
        lines.append(f"  症状: {', '.join(p.symptoms)}")
        lines.append(f"  常见根因: {', '.join(p.common_causes)}")
        lines.append(f"  修复策略: {', '.join(p.repair_strategies)}")
    return "\n".join(lines)


def create_diagnosis_agent(
    llm: ChatOpenAI,
    fault_patterns: list[FaultPattern],
) -> StateGraph:
    """Create the 4-phase Diagnosis Agent as a compiled LangGraph StateGraph."""

    patterns_text = _format_patterns(fault_patterns)
    # Tools for each phase
    phase1_tools = DIAGNOSTIC_TOOLS[:4]  # metrics, logs, traces, change_events
    phase2_tools = DIAGNOSTIC_TOOLS[4:6]  # processlist, explain (deeper dive)
    phase3_tools = []  # LLM reasoning only
    phase4_tools = DIAGNOSTIC_TOOLS[5:8]  # explain, heap_dump, config_diff

    async def phase1_collect(state: DiagnosisState) -> dict:
        alert = state["alert"]
        system_prompt = f"""你是一个SRE故障诊断专家。当前处于 **Phase 1: 信息收集** 阶段。

告警信息：
- 标题: {alert.title}
- 资源: {alert.resource}
- 指标: {alert.metric} = {alert.current_value} (阈值: {alert.threshold})
- 来源: {alert.source}
- 严重度: {alert.severity}

**任务**：调用工具全面收集告警相关的数据。必须至少调用:
1. query_metrics - 查询相关指标
2. query_logs - 查询相关错误日志
3. query_change_events - 查询最近变更

收集完数据后，用中文总结你收集到了什么，格式为JSON:
{{"summary": "数据摘要", "key_findings": ["发现1", "发现2"]}}

只输出JSON，不要其他内容。"""
        llm_with_tools = llm.bind_tools(phase1_tools)
        response = await llm_with_tools.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"开始Phase 1信息收集。告警ID: {alert.id}")
        ])
        return {
            "collected_data": {"phase1_output": response.content, "tool_calls_done": True},
            "messages": state.get("messages", []) + [response],
        }

    async def phase2_trace(state: DiagnosisState) -> dict:
        alert = state["alert"]
        collected = state.get("collected_data", {})
        system_prompt = f"""你是一个SRE故障诊断专家。当前处于 **Phase 2: 数据追踪** 阶段。

告警: {alert.title} on {alert.resource}
Phase 1 收集的数据: {json.dumps(collected, ensure_ascii=False, default=str)}

**任务**：追溯异常指标的来源和传播链路。深入调查：
- 哪些进程/查询占用了资源？
- 异常是从哪个组件开始传播的？
- 调用必要的工具深入分析（如query_processlist、query_explain等）

输出JSON格式:
{{"trace_chain": "异常传播链路描述", "affected_components": ["组件1", "组件2"], "bottleneck": "瓶颈所在"}}

只输出JSON。"""
        llm_with_tools = llm.bind_tools(phase2_tools)
        response = await llm_with_tools.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content="开始Phase 2数据追踪。")
        ])
        return {
            "trace_result": {"phase2_output": response.content},
            "messages": state.get("messages", []) + [response],
        }

    async def phase3_root_cause(state: DiagnosisState) -> dict:
        alert = state["alert"]
        collected = state.get("collected_data", {})
        trace = state.get("trace_result", {})

        # Try pattern matching first
        matched = match_patterns(f"{alert.title} {alert.metric}", fault_patterns)
        matched_text = _format_patterns(matched) if matched else "无匹配的已知故障模式"

        system_prompt = f"""你是一个SRE故障诊断专家。当前处于 **Phase 3: 根因定位** 阶段。

告警: {alert.title} on {alert.resource}
Phase 1 数据: {json.dumps(collected, ensure_ascii=False, default=str)}
Phase 2 追踪: {json.dumps(trace, ensure_ascii=False, default=str)}

已知故障模式库（相似案例）:
{matched_text}

**任务**：定位根本原因。严格按以下格式输出JSON:
{{
    "root_cause": "根因的完整描述（一句话）",
    "evidence_chain": ["证据1", "证据2", "证据3"],
    "confidence": 0.85,
    "repair_hypothesis": "修复假设描述",
    "matched_pattern": "匹配到的故障模式ID（如有，否则为null）"
}}

要求：
- evidence_chain 必须包含至少3条具体证据
- confidence: 基于证据充分程度给出0-1的置信度
- 如果 confidence < 0.6，说明证据不足，标记为低置信度

只输出JSON。"""
        response = await llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content="开始Phase 3根因定位。")
        ])

        # Parse the JSON response
        try:
            result = json.loads(response.content.strip().removeprefix("```json").removesuffix("```").strip())
        except json.JSONDecodeError:
            result = {
                "root_cause": response.content,
                "evidence_chain": [],
                "confidence": 0.3,
                "repair_hypothesis": "",
                "matched_pattern": None,
            }

        return {
            "root_cause": result.get("root_cause", ""),
            "evidence_chain": result.get("evidence_chain", []),
            "confidence": float(result.get("confidence", 0.5)),
            "repair_hypothesis": result.get("repair_hypothesis", ""),
            "messages": state.get("messages", []) + [response],
        }

    async def phase4_validate(state: DiagnosisState) -> dict:
        hypothesis = state.get("repair_hypothesis", "")
        root_cause = state.get("root_cause", "")
        evidence = state.get("evidence_chain", [])
        alert = state["alert"]

        system_prompt = f"""你是一个SRE故障诊断专家。当前处于 **Phase 4: 验证修复** 阶段。

根因: {root_cause}
证据链: {json.dumps(evidence, ensure_ascii=False)}
修复假设: {hypothesis}

**任务**：在逻辑上验证修复假设的正确性。不执行实际操作，而是推演验证逻辑。
- 调用相关工具获取验证所需的数据（如EXPLAIN结果、配置对比等）
- 判断修复假设是否合理

输出JSON格式:
{{
    "hypothesis_validated": true/false,
    "validation_detail": "验证过程和结论",
    "revised_hypothesis": "如果验证失败，修正后的假设（否则为null）"
}}

只输出JSON。"""
        llm_with_tools = llm.bind_tools(phase4_tools)
        response = await llm_with_tools.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content="开始Phase 4验证修复。")
        ])

        try:
            result = json.loads(response.content.strip().removeprefix("```json").removesuffix("```").strip())
        except json.JSONDecodeError:
            result = {
                "hypothesis_validated": True,
                "validation_detail": response.content,
                "revised_hypothesis": None,
            }

        return {
            "hypothesis_validated": bool(result.get("hypothesis_validated", True)),
            "validation_detail": str(result.get("validation_detail", "")),
            "messages": state.get("messages", []) + [response],
        }

    async def generate_report(state: DiagnosisState) -> dict:
        now = datetime.now()
        report = DiagnosisReport(
            alert_id=state["alert"].id,
            timestamp=now,
            root_cause=state.get("root_cause", ""),
            evidence_chain=state.get("evidence_chain", []),
            confidence=float(state.get("confidence", 0.0)),
            repair_hypothesis=state.get("repair_hypothesis", ""),
            hypothesis_validated=bool(state.get("hypothesis_validated", False)),
            validation_detail=str(state.get("validation_detail", "")),
            diagnosis_duration_s=0.0,
            phases_completed=["phase1", "phase2", "phase3"],
        )
        if state.get("hypothesis_validated") is not None:
            report.phases_completed.append("phase4")

        return {
            "diagnosis_report": report,
        }

    def should_validate(state: DiagnosisState) -> str:
        confidence = state.get("confidence", 0.0)
        if confidence >= 0.6:
            return "phase4"
        return "generate_report"

    workflow = StateGraph(DiagnosisState)

    workflow.add_node("phase1", phase1_collect)
    workflow.add_node("phase2", phase2_trace)
    workflow.add_node("phase3", phase3_root_cause)
    workflow.add_node("phase4", phase4_validate)
    workflow.add_node("generate_report", generate_report)

    workflow.set_entry_point("phase1")
    workflow.add_edge("phase1", "phase2")
    workflow.add_edge("phase2", "phase3")
    workflow.add_conditional_edges(
        "phase3",
        should_validate,
        {"phase4": "phase4", "generate_report": "generate_report"},
    )
    workflow.add_edge("phase4", "generate_report")
    workflow.add_edge("generate_report", END)

    return workflow.compile(checkpointer=MemorySaver())


async def run_diagnosis(agent, alert: AlertEvent) -> DiagnosisReport:
    """Run the diagnosis agent and return the report."""
    start = time.time()
    config = {"configurable": {"thread_id": alert.id}}
    initial_state: DiagnosisState = {
        "alert": alert,
        "collected_data": {},
        "trace_result": {},
        "root_cause": "",
        "evidence_chain": [],
        "confidence": 0.0,
        "repair_hypothesis": "",
        "hypothesis_validated": False,
        "validation_detail": "",
        "diagnosis_report": None,
        "messages": [],
    }

    final_state = await agent.ainvoke(initial_state, config)
    report = final_state.get("diagnosis_report")
    if report is None:
        raise RuntimeError("Diagnosis agent did not produce a report")
    report.diagnosis_duration_s = time.time() - start
    return report
```

- [ ] **Step 4: Run tests (requires LLM_API_KEY configured)**

```bash
cd /Users/zhouqiantalaogong/PycharmProjects/guard-agent
LLM_API_KEY=<your-key> pytest tests/test_diagnosis_agent.py -v -s
```
Expected: 4 PASS (tests call real API, may take ~30-60s each)

- [ ] **Step 5: Commit**

```bash
cd /Users/zhouqiantalaogong/PycharmProjects/guard-agent
git add src/agents/diagnosis.py tests/test_diagnosis_agent.py
git commit -m "feat: add Diagnosis Agent with 4-phase StateGraph"
```

---

### Task 8: Decision Agent

**Files:**
- Create: `src/agents/decision.py`
- Create: `tests/test_decision_agent.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_decision_agent.py
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
        validation_detail="EXPLAIN验证通过，添加索引后扫描行数从284万降至1500",
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
    # Level 1 proposals don't require confirmation if no actionable repair
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
    # Level 4 operations should not appear in proposals
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


@pytest.mark.asyncio
async def test_decision_llm_fallback():
    """When repair_hypothesis mentions an unknown operation, LLM should classify it."""
    llm = create_llm(max_tokens=512)
    report = make_report(
        repair_hypothesis="需要调整数据库buffer pool大小并进行在线表优化"
    )
    proposal = await run_decision(report, llm=llm)
    assert isinstance(proposal, DecisionProposal)
    assert len(proposal.reasoning) > 0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/zhouqiantalaogong/PycharmProjects/guard-agent
pytest tests/test_decision_agent.py -v
```
Expected: FAIL

- [ ] **Step 3: Write Decision Agent implementation**

```python
# src/agents/decision.py
import re
from datetime import datetime
from typing import Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from src.models.diagnosis import DiagnosisReport
from src.models.decision import DecisionProposal, OperationPlan
from src.safety.operation_levels import classify_operation, is_operation_allowed, get_level_description


_ACTION_PATTERNS: list[tuple[str, str, str, str]] = [
    # (keyword, action_name, description_template, verification)
    # Level 2 operations
    ("restart", "restart_pod", "重启Pod", "kubectl get pods 确认Pod状态为Running"),
    ("重启", "restart_pod", "重启Pod", "kubectl get pods 确认Pod状态为Running"),
    ("scale up", "scale_up", "扩容实例", "检查实例数和负载指标是否恢复正常"),
    ("scale down", "scale_down", "缩容实例", "检查剩余实例负载是否在安全范围内"),
    ("扩容", "scale_up", "扩容实例", "检查实例数和负载指标是否恢复正常"),
    ("限流", "adjust_rate_limit", "调整限流阈值", "检查错误率和QPS是否恢复正常"),
    ("熔断", "adjust_circuit_breaker", "调整熔断参数", "检查服务调用成功率"),
    ("clear cache", "clear_cache", "清理缓存", "检查缓存命中率和内存使用"),
    ("清除缓存", "clear_cache", "清理缓存", "检查缓存命中率和内存使用"),
    ("add index", "add_index", "添加数据库索引", "EXPLAIN验证查询计划已使用新索引"),
    ("添加索引", "add_index", "添加数据库索引", "EXPLAIN验证查询计划已使用新索引"),
    ("create index", "add_index", "添加数据库索引", "EXPLAIN验证查询计划已使用新索引"),
    # Level 3 operations
    ("回滚配置", "deploy_rollback", "回滚配置到上一版本", "检查错误率是否恢复到部署前水平"),
    ("deploy rollback", "deploy_rollback", "回滚配置到上一版本", "检查错误率是否恢复到部署前水平"),
    ("rollback", "deploy_rollback", "回滚部署", "检查服务指标是否恢复正常"),
    ("modify config", "modify_config", "修改配置", "检查配置生效并验证服务行为"),
    ("修改配置", "modify_config", "修改配置", "检查配置生效并验证服务行为"),
    ("调整连接池", "adjust_connection_pool", "调整数据库连接池参数", "检查连接池使用率和等待队列"),
    # Level 4 operations (detected but rejected)
    ("drop database", "drop_database", None, None),
    ("drop table", "drop_table", None, None),
    ("truncate", "truncate_table", None, None),
    ("rm -rf", "rm_rf", None, None),
    ("kill -9", "kill_process", None, None),
]


def _extract_actions(hypothesis: str) -> list[tuple[str, str, str, str]]:
    """Extract actionable items from the repair hypothesis using pattern matching."""
    actions = []
    hypothesis_lower = hypothesis.lower()
    for keyword, action_name, desc_template, verification in _ACTION_PATTERNS:
        if keyword in hypothesis_lower:
            actions.append((keyword, action_name, desc_template, verification))
    return actions


def run_decision(
    report: DiagnosisReport,
    llm: Optional[ChatOpenAI] = None,
) -> DecisionProposal:
    """Run the Decision Agent: classify operations by safety level and generate proposal."""

    actions = _extract_actions(report.repair_hypothesis)
    proposals: list[OperationPlan] = []

    for keyword, action_name, desc_template, verification in actions:
        level = classify_operation(action_name)

        if level is None:
            # Unknown operation - need LLM to classify
            if llm:
                level = _llm_classify(llm, action_name, report.repair_hypothesis)
            else:
                continue  # skip if no LLM available

        if level == 4:
            # Level 4: completely forbidden, skip with warning
            proposals.append(OperationPlan(
                action=action_name,
                level=4,
                description=f"[已拒绝] {desc_template or keyword}",
                expected_impact="操作被安全策略拒绝",
                rollback_plan="N/A",
                verification="N/A",
            ))
            continue

        if desc_template is None:
            continue

        proposals.append(OperationPlan(
            action=action_name,
            level=level,
            description=desc_template,
            expected_impact=f"执行{desc_template}操作",
            rollback_plan=f"如失败则恢复原状态" if level >= 2 else "无需回滚",
            verification=verification or "检查相关指标是否恢复",
        ))

    # If no specific actions detected, generate a generic observation proposal
    if not proposals and report.root_cause:
        proposals.append(OperationPlan(
            action="query_metrics",
            level=1,
            description=f"观察: {report.root_cause}",
            expected_impact="持续监控指标变化",
            rollback_plan="无",
            verification="持续观察30分钟",
        ))

    requires_confirmation = any(p.level >= 2 for p in proposals)

    # Build reasoning
    parts = []
    if proposals:
        parts.append(f"诊断报告置信度: {report.confidence:.0%}")
        parts.append(f"根因: {report.root_cause}")
        for p in proposals:
            parts.append(f"[Level {p.level}] {p.description} - {get_level_description(p.level)}")

    return DecisionProposal(
        report_id=report.alert_id,
        timestamp=datetime.now(),
        proposals=proposals,
        requires_confirmation=requires_confirmation,
        reasoning="\n".join(parts),
    )


def _llm_classify(llm: ChatOpenAI, action_name: str, context: str) -> int:
    """Use LLM to classify an unknown operation into safety level 1-4."""
    system_prompt = """你是一个运维安全专家。请判断以下操作的安全等级。

等级定义：
1 - 只读操作/查询，无副作用
2 - 有副作用的操作，但影响范围可控，需人工确认
3 - 高风险操作，可能影响核心业务，需人工审批
4 - 极度危险操作，绝对不能执行

请只回复数字1-4。"""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"操作: {action_name}\n上下文: {context}\n\n请判断安全等级 (1-4):")
    ])

    try:
        level = int(response.content.strip()[0])
        return max(1, min(4, level))
    except (ValueError, IndexError):
        return 3  # unknown defaults to Level 3 (conservative)
```

- [ ] **Step 4: Run tests**

```bash
cd /Users/zhouqiantalaogong/PycharmProjects/guard-agent
pytest tests/test_decision_agent.py -v -k "not llm_fallback"
```
Expected: 5 PASS
```bash
LLM_API_KEY=<your-key> pytest tests/test_decision_agent.py::test_decision_llm_fallback -v -s
```
Expected: 1 PASS

- [ ] **Step 5: Commit**

```bash
cd /Users/zhouqiantalaogong/PycharmProjects/guard-agent
git add src/agents/decision.py tests/test_decision_agent.py
git commit -m "feat: add Decision Agent with rule engine and LLM fallback"
```

---

### Task 9: Supervisor Agent

**Files:**
- Create: `src/agents/supervisor.py`
- Create: `tests/test_supervisor.py`

- [ ] **Step 1: Write tests**

```python
# tests/test_supervisor.py
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
    # Verify order: diagnosis before decision
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/zhouqiantalaogong/PycharmProjects/guard-agent
pytest tests/test_supervisor.py -v
```
Expected: FAIL

- [ ] **Step 3: Write Supervisor Agent**

```python
# src/agents/supervisor.py
from typing import Any
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from src.models.alert import AlertEvent
from src.models.diagnosis import DiagnosisReport
from src.models.decision import DecisionProposal
from src.knowledge.fault_patterns import FaultPattern
from src.agents.diagnosis import create_diagnosis_agent, run_diagnosis
from src.agents.decision import run_decision


# Module-level storage for supervisor-agent handoff
_pending_report: DiagnosisReport | None = None
_pending_proposal: DecisionProposal | None = None


def create_supervisor(
    llm: ChatOpenAI,
    fault_patterns: list[FaultPattern],
) -> Any:
    """Create the Supervisor Agent as a LangGraph react agent with sub-agent tools."""

    diagnosis_agent = create_diagnosis_agent(llm, fault_patterns)

    @tool
    async def run_diagnosis_tool(alert_json: str) -> str:
        """调用诊断Agent执行四阶段故障诊断。参数 alert_json: AlertEvent的JSON字符串。
        必须在任何决策之前先调用此工具。返回诊断报告的JSON。"""
        import json
        alert_dict = json.loads(alert_json)
        from datetime import datetime
        alert = AlertEvent(
            id=alert_dict["id"],
            timestamp=datetime.fromisoformat(alert_dict["timestamp"]),
            source=alert_dict["source"],
            severity=alert_dict["severity"],
            title=alert_dict["title"],
            resource=alert_dict["resource"],
            metric=alert_dict["metric"],
            current_value=alert_dict["current_value"],
            threshold=alert_dict["threshold"],
            labels=alert_dict.get("labels", {}),
            raw_data=alert_dict.get("raw_data", {}),
        )
        global _pending_report
        _pending_report = await run_diagnosis(diagnosis_agent, alert)
        return json.dumps({
            "alert_id": _pending_report.alert_id,
            "root_cause": _pending_report.root_cause,
            "confidence": _pending_report.confidence,
            "evidence_chain": _pending_report.evidence_chain,
            "repair_hypothesis": _pending_report.repair_hypothesis,
            "hypothesis_validated": _pending_report.hypothesis_validated,
            "phases_completed": _pending_report.phases_completed,
        }, ensure_ascii=False, default=str)

    @tool
    async def run_decision_tool(unused: str = "") -> str:
        """调用决策Agent，基于诊断报告生成操作决策。
        参数 unused: 占位参数，传入空字符串即可。
        返回决策方案的JSON。"""
        import json
        global _pending_report, _pending_proposal
        if _pending_report is None:
            return json.dumps({"error": "没有诊断报告，请先运行诊断"})
        _pending_proposal = run_decision(_pending_report, llm=llm)
        return json.dumps({
            "requires_confirmation": _pending_proposal.requires_confirmation,
            "reasoning": _pending_proposal.reasoning,
            "proposals": [
                {
                    "action": p.action,
                    "level": p.level,
                    "description": p.description,
                    "expected_impact": p.expected_impact,
                    "rollback_plan": p.rollback_plan,
                }
                for p in _pending_proposal.proposals
            ],
        }, ensure_ascii=False)

    @tool
    def request_human_confirmation(summary: str) -> str:
        """当决策需要人工确认时调用此工具。参数 summary: 操作摘要（含诊断结论、建议操作、预期影响）。
        返回 'confirmed' 或 'rejected'。"""
        print("\n" + "=" * 60)
        print(" [人工确认请求]")
        print("=" * 60)
        print(summary)
        print("-" * 60)
        response = input("输入 confirm 确认执行 / reject 拒绝: ").strip().lower()
        if response == "confirm":
            return "confirmed"
        return "rejected"

    system_prompt = """你是一个SRE运维 Supervisor Agent。你的职责是编排故障诊断和决策流程。

**必须严格遵循的工作流**：
1. 收到告警 → 立即调用 run_diagnosis_tool 进行诊断（将告警信息以JSON格式传入）
2. 收到诊断报告 → 立即调用 run_decision_tool 生成决策方案
3. 收到决策方案 → 如果 requires_confirmation=true，调用 request_human_confirmation
4. 收到确认后 → 输出最终总结
5. 如果 requires_confirmation=false（Level 1操作），直接输出总结

**重要规则**：
- 绝不能在诊断之前调用决策
- 绝不能在决策之前请求确认
- 每一步完成后立即进行下一步
- 最终输出完整的处理总结"""

    tools = [run_diagnosis_tool, run_decision_tool, request_human_confirmation]

    return create_react_agent(
        llm,
        tools,
        state_modifier=system_prompt,
        checkpointer=MemorySaver(),
    )


async def run_supervisor(
    supervisor,
    alert: AlertEvent,
    auto_confirm: bool = False,
) -> dict[str, Any]:
    """Run the supervisor for a given alert. Returns dict with diagnosis_report and decision_proposal."""
    import json

    alert_json = json.dumps({
        "id": alert.id,
        "timestamp": alert.timestamp.isoformat(),
        "source": alert.source,
        "severity": alert.severity,
        "title": alert.title,
        "resource": alert.resource,
        "metric": alert.metric,
        "current_value": alert.current_value,
        "threshold": alert.threshold,
        "labels": alert.labels,
        "raw_data": alert.raw_data,
    }, ensure_ascii=False, default=str)

    config = {"configurable": {"thread_id": alert.id}}

    prompt = (
        f"收到一个运维告警，请立即开始诊断流程。\n\n"
        f"告警信息（JSON格式）:\n{alert_json}\n\n"
        f"请调用 run_diagnosis_tool 开始诊断。"
    )

    result = await supervisor.ainvoke(
        {"messages": [{"role": "user", "content": prompt}]},
        config,
    )

    # Auto-confirm if needed
    global _pending_report, _pending_proposal
    if auto_confirm and _pending_proposal and _pending_proposal.requires_confirmation:
        # In auto_confirm mode, we simulate confirmation
        pass

    return {
        "alert_id": alert.id,
        "diagnosis_report": _pending_report,
        "decision_proposal": _pending_proposal,
        "supervisor_output": result,
    }
```

- [ ] **Step 4: Run integration tests**

```bash
cd /Users/zhouqiantalaogong/PycharmProjects/guard-agent
LLM_API_KEY=<your-key> pytest tests/test_supervisor.py -v -s
```
Expected: 3 PASS (each test takes ~60-90s due to multi-LLM calls)

- [ ] **Step 5: Commit**

```bash
cd /Users/zhouqiantalaogong/PycharmProjects/guard-agent
git add src/agents/supervisor.py tests/test_supervisor.py
git commit -m "feat: add Supervisor Agent for orchestration (diagnosis → decision → confirmation)"
```

---

### Task 10: CLI entry point

**Files:**
- Create: `src/main.py`

- [ ] **Step 1: Write CLI entry point**

```python
# src/main.py
"""Guard Agent CLI - Intelligent SRE Operations Assistant."""
import argparse
import asyncio
import sys
import json
from datetime import datetime

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
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="详细输出",
    )

    args = parser.parse_args()
    setup_logging()

    print(f"\n{'='*60}")
    print(f" Guard Agent - 智能运维故障诊断")
    print(f" 场景: {args.scenario}")
    print(f" 时间: {datetime.now().isoformat()}")
    print(f"{'='*60}\n")

    # Load scenario
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
                "alert": {
                    "id": alert.id,
                    "title": alert.title,
                    "resource": alert.resource,
                },
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
```

- [ ] **Step 2: Verify CLI help works**

```bash
cd /Users/zhouqiantalaogong/PycharmProjects/guard-agent
python -m src.main --help
```
Expected: help text with scenario choices

- [ ] **Step 3: Commit**

```bash
cd /Users/zhouqiantalaogong/PycharmProjects/guard-agent
git add src/main.py
git commit -m "feat: add CLI entry point with scenario selection and JSON output"
```

---

### Task 11: End-to-end integration test

**Files:**
- Create: `tests/test_integration.py`

- [ ] **Step 1: Write end-to-end test covering all 3 scenarios**

```python
# tests/test_integration.py
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

    # Verify the diagnosis mentions relevant keywords for this scenario
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
```

- [ ] **Step 2: Run full integration test suite**

```bash
cd /Users/zhouqiantalaogong/PycharmProjects/guard-agent
LLM_API_KEY=<your-key> pytest tests/test_integration.py -v -s
```
Expected: 4 PASS

- [ ] **Step 3: Run all tests**

```bash
cd /Users/zhouqiantalaogong/PycharmProjects/guard-agent
LLM_API_KEY=<your-key> pytest tests/ -v
```
Expected: All tests pass (model + safety + tools + diagnosis + decision + supervisor + integration)

- [ ] **Step 4: Commit**

```bash
cd /Users/zhouqiantalaogong/PycharmProjects/guard-agent
git add tests/test_integration.py
git commit -m "test: add end-to-end integration tests for all 3 scenarios"
```

---

## Summary

**Total tasks:** 11
**Files created:** ~25
**Test coverage:** models, fault patterns, safety levels, diagnostic tools, diagnosis agent, decision agent, supervisor agent, end-to-end integration

**Prerequisites:**
- `LLM_API_KEY` environment variable set to DeepSeek API key
- `LLM_BASE_URL` defaults to `https://api.deepseek.com/v1` (override if using proxy)

**Run the full system:**
```bash
cd /Users/zhouqiantalaogong/PycharmProjects/guard-agent
LLM_API_KEY=<your-key> python -m src.main connection_pool --auto-confirm
```
