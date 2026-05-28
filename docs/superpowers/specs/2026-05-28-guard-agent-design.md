# Guard Agent 设计说明书

## 1. 概述

Guard Agent 是一个遵循 SRE 最佳实践的智能化运维助手。核心能力是**结构化故障诊断**：接收异常信号，通过四阶段诊断流程（信息收集 → 数据追踪 → 根因定位 → 验证修复）定位根因，在四级安全边界框架下生成操作建议，需要时等待人工确认。

MVP 聚焦**决策与诊断层**（大脑），感知层和执行层用 Mock 模拟，用 2-3 个典型故障场景验证诊断推理质量。

## 2. 技术选型

| 层面 | 选型 | 理由 |
|---|---|---|
| 语言 | Python 3.11+ | LLM 生态最完善，团队主流语言 |
| Agent 框架 | LangGraph | StateGraph 建模诊断流程，原生 human-in-the-loop，checkpointer 做审计 |
| LLM | DeepSeek v4 | 兼容 OpenAI API，推理能力强 |
| LLM 接入 | OpenAI SDK（兼容模式） | DeepSeek 兼容 OpenAI API 格式 |
| 包管理 | Poetry | 依赖锁定，企业级可复现 |
| 测试 | pytest + pytest-asyncio | 异步 Agent 调用的标准测试栈 |

## 3. 架构总览

### 3.1 分层与 Agent 映射

```
┌─────────────────────────────────────────────────┐
│               用户协作层                          │
│  HumanConfirmation  │  AuditLogger  │  Reporter  │
└──────────────────────┬──────────────────────────┘
                       │
┌──────────────────────┴──────────────────────────┐
│             决策与诊断层 (MVP 范围)               │
│                                                  │
│  ┌──────────────┐  ┌──────────────┐              │
│  │  Supervisor   │  │   Decision    │              │
│  │    Agent      │──│    Agent      │              │
│  │  (编排调度)    │  │ (安全边界+决策)│              │
│  └──────┬───────┘  └──────────────┘              │
│         │                                        │
│  ┌──────┴───────┐                                │
│  │  Diagnosis    │                               │
│  │    Agent      │                               │
│  │ (四阶段诊断)   │                               │
│  └──────────────┘                                │
└──────────────────────┬──────────────────────────┘
                       │
┌──────────────────────┴──────────────────────────┐
│             感知与执行层 (Mock)                    │
│  MockMetrics  │  MockLogs  │  MockChangeEvents   │
└─────────────────────────────────────────────────┘
```

### 3.2 三个 Agent 的职责与边界

**Supervisor Agent**
- 职责：编排调度，管理每个故障处理任务的生命周期
- 输入：来自感知模块的标准化 `AlertEvent`
- 核心流程：接收告警 → 派发诊断 → 收到诊断报告 → 派发决策 → 收到决策方案 →（后续）派发执行 → 收到验证结果 → 结束/重试
- 实现：LangGraph 的 supervisor pattern，将 Diagnosis Agent 和 Decision Agent 作为 tool 调用
- 状态管理：通过 LangGraph checkpointer 持久化每个故障处理会话的状态

**Diagnosis Agent**
- 职责：执行四阶段结构化诊断，输出诊断报告
- 输入：`AlertEvent` + Supervisor 传递的上下文
- 内部流程：Phase 1→2→3→4 的 StateGraph（见 4.1 节）
- 输出：`DiagnosisReport`（根因链、证据、置信度、修复假设及验证结果）
- 独立性：不关心"谁来执行修复"，只输出诊断结论

**Decision Agent**
- 职责：接收诊断报告，判定操作的安全等级，生成决策方案
- 输入：`DiagnosisReport`
- 核心逻辑：规则引擎优先（Level 1-4 硬编码映射表 + 安全边界规则）→ LLM 兜底（多异常并发、规则未覆盖场景）
- 输出：`DecisionProposal`（方案列表、每个方案的安全等级、预期影响、回滚策略）
- 独立性：不关心"怎么诊断出来的"，只看诊断报告的结论和置信度

## 4. 核心模块详细设计

### 4.1 Diagnosis Agent 内部状态机

Diagnosis Agent 内部使用 LangGraph StateGraph 强制四阶段流程。状态定义：

```python
class DiagnosisState(TypedDict):
    alert: AlertEvent           # 输入：异常事件
    # Phase 1 产出
    collected_data: dict        # 聚合的指标/日志/事件/变更记录
    # Phase 2 产出
    trace_result: dict          # 异常链路追踪结果
    # Phase 3 产出
    root_cause: str             # 根因描述
    evidence_chain: list[str]   # 证据链
    confidence: float           # 置信度 0-1
    repair_hypothesis: str      # 修复假设
    # Phase 4 产出
    hypothesis_validated: bool  # 假设是否通过验证
    validation_detail: str      # 验证过程描述
    # 最终产出
    diagnosis_report: DiagnosisReport
```

**Phase 1: 信息收集**
- LLM 分析 alert 内容，决定需要收集哪些数据
- 调用 tools：`query_metrics()`, `query_logs()`, `query_change_events()`, `query_traces()`
- 输出：时间窗口内的完整数据快照
- MVP：所有 tool 返回预定义的 mock 数据（对应故障场景）

**Phase 2: 数据追踪**
- LLM 分析 Phase 1 收集的数据，追溯异常指标的来源
- 例："连接池耗尽 → 哪些服务占用连接？→ 这些服务在执行什么操作？"
- 输出：异常传播链路

**Phase 3: 根因定位**
- LLM 结合 Phase 1+2 的数据 + 故障模式库（见 4.3），对比变更前后差异
- 识别匹配的故障模式，或推理新的根因链
- 提出修复假设
- 输出：`root_cause`, `evidence_chain`, `confidence`, `repair_hypothesis`
- 硬约束：confidence < 0.6 时，标记为"低置信度"，不进入 Phase 4，直接输出报告并要求人工介入

**Phase 4: 验证修复**
- 在逻辑上验证 repair_hypothesis（不执行实际操作）
- 例："假设缺少索引导致慢查询" → 验证"添加索引后 EXPLAIN 结果是否改善"（通过查询 schema 和执行计划来推演，不真正执行 DDL）
- 通过后才输出 diagnosis_report
- 不通过则回到 Phase 3 重新推理

### 4.2 Decision Agent 的安全边界

Decision Agent 的核心是**规则引擎优先，LLM 兜底**。

**规则引擎：四级操作分级表**

```python
OPERATION_LEVELS = {
    # Level 1: 自动批准（Agent 可直接执行）
    "query_metrics": 1,
    "query_logs": 1,
    "query_traces": 1,
    "query_topology": 1,

    # Level 2: 需人工确认
    "restart_pod": 2,
    "scale_up": 2,
    "scale_down": 2,
    "adjust_rate_limit": 2,
    "adjust_circuit_breaker": 2,
    "clear_cache": 2,
    "run_diagnostic_command": 2,

    # Level 3: 需人工审批（生成详细方案）
    "modify_config": 3,
    "execute_ddl": 3,
    "execute_dml": 3,
    "deploy_rollback": 3,
    "modify_route": 3,

    # Level 4: 完全禁止
    "drop_database": 4,
    "drop_table": 4,
    "truncate_table": 4,
    "rm_rf": 4,
    "kill_process": 4,
    "iptables_modify": 4,
}
```

**决策流程**：
1. 收到 DiagnosisReport → 提取 repair_hypothesis 中建议的操作
2. 规则引擎查表，判定每个操作的安全等级
3. Level 1 → 直接生成 "自动执行" 决策
4. Level 2/3 → 生成 DecisionProposal（含问题摘要、诊断结论、建议操作、预期影响、回滚方案），标记 `requires_confirmation=True`
5. Level 4 → 直接拒绝，记录告警
6. 多异常并发或规则未覆盖 → 调用 LLM 做综合判断和优先级排序

### 4.3 故障模式库

诊断Agent 的 Phase 3 依赖一个可检索的故障模式库来做模式匹配。

```python
@dataclass
class FaultPattern:
    id: str
    name: str                    # "数据库连接池耗尽"
    symptoms: list[str]          # ["连接数>90%", "响应超时", "错误日志: too many connections"]
    typical_traces: list[str]    # ["大量慢查询占用连接", "连接未释放"]
    common_causes: list[str]     # ["缺少索引导致全表扫描", "连接泄漏", "突发流量"]
    diagnostic_queries: list[str] # ["SHOW PROCESSLIST", "EXPLAIN 慢查询", "检查连接池配置"]
    repair_strategies: list[str] # ["添加索引", "修复连接泄漏", "调大连接池"]
    level: int                   # 1=常见 2=偶发 3=罕见
```

MVP 预置 2-3 个故障模式：
1. 数据库连接池耗尽（慢查询导致）
2. 内存溢出 OOM（缓存未限制大小导致）
3. 配置变更导致服务异常

故障模式存储在 JSON/YAML 文件中，诊断Agent 通过 semantic search（embedding → 向量匹配）检索最相关的历史案例。

### 4.4 Supervisor Agent 编排逻辑

Supervisor 使用 LangGraph 的 `create_react_agent` 模式，将 Diagnosis Agent 和 Decision Agent 作为两个 tool 暴露给 LLM：

```python
tools = [
    run_diagnosis,    # 调用 Diagnosis Agent
    run_decision,     # 调用 Decision Agent
    request_human_confirmation,  # 人工确认（MVP 用 CLI 输入模拟）
    generate_report,  # 生成最终报告
]
```

Supervisor 的 system prompt 定义了编排规则：
- 收到 Alert → 必须先调用 `run_diagnosis`
- 收到 DiagnosisReport → 必须调用 `run_decision`
- 收到 DecisionProposal → 如果 requires_confirmation=True，调用 `request_human_confirmation`
- 收到确认 →（后续）调用执行模块
- 记录所有步骤到 AuditLogger

## 5. 数据模型

### 5.1 AlertEvent（输入）
```python
@dataclass
class AlertEvent:
    id: str
    timestamp: datetime
    source: str              # "prometheus" | "elk" | "kafka"
    severity: str            # "critical" | "warning" | "info"
    title: str               # "数据库连接数超过阈值"
    resource: str            # "mysql-orders-db"
    metric: str              # "connections_active"
    current_value: float
    threshold: float
    labels: dict
    raw_data: dict           # 原始告警 payload
```

### 5.2 DiagnosisReport（诊断Agent 输出）
```python
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
    phases_completed: list[str]
```

### 5.3 DecisionProposal（决策Agent 输出）
```python
@dataclass
class DecisionProposal:
    report_id: str
    timestamp: datetime
    proposals: list[OperationPlan]
    requires_confirmation: bool
    reasoning: str

@dataclass
class OperationPlan:
    action: str              # 操作名称
    level: int               # 安全等级 1-4
    description: str         # 操作描述
    expected_impact: str     # 预期影响
    rollback_plan: str       # 回滚方案
    verification: str        # 验证方法
```

## 6. 项目目录结构

```
guard-agent/
├── src/
│   ├── __init__.py
│   ├── main.py                    # 入口：CLI 触发诊断流程
│   ├── models/
│   │   ├── __init__.py
│   │   ├── alert.py               # AlertEvent
│   │   ├── diagnosis.py           # DiagnosisReport, DiagnosisState
│   │   └── decision.py            # DecisionProposal, OperationPlan
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── supervisor.py          # Supervisor Agent (LangGraph)
│   │   ├── diagnosis.py           # Diagnosis Agent (四阶段 StateGraph)
│   │   └── decision.py            # Decision Agent (规则引擎 + LLM)
│   ├── knowledge/
│   │   ├── __init__.py
│   │   ├── fault_patterns.py      # FaultPattern 数据类 + 加载
│   │   └── fault_patterns.json    # 预置故障模式库
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── diagnostic.py          # 诊断相关工具（query_metrics 等，MVP mock）
│   │   └── mock_data.py           # Mock 数据提供器（2-3个故障场景）
│   ├── safety/
│   │   ├── __init__.py
│   │   └── operation_levels.py    # 四级操作分级表 + 规则引擎
│   └── utils/
│       ├── __init__.py
│       ├── llm.py                  # LLM 客户端工厂
│       └── logging.py             # 日志配置
├── tests/
│   ├── __init__.py
│   ├── test_diagnosis_agent.py    # 诊断Agent 单元测试
│   ├── test_decision_agent.py     # 决策Agent 单元测试
│   ├── test_supervisor.py         # Supervisor 集成测试
│   └── fixtures/                  # 测试用的 mock 数据
│       ├── connection_pool.json
│       ├── oom.json
│       └── config_change.json
├── pyproject.toml
├── README.md
└── .gitignore
```

## 7. MVP 故障场景

| 场景 | 故障描述 | 根因 | 验证内容 |
|---|---|---|---|
| 数据库连接池耗尽 | 连接数 148/150，大量超时 | orders 表缺少 status 字段索引，全表扫描 | Phase 4 验证 EXPLAIN 改善 |
| 内存溢出 OOM | Pod 频繁重启，heap 使用 98% | 缓存未设大小上限，持续增长至 OOM | Phase 4 验证缓存大小计算 |
| 配置变更异常 | 部署后错误率飙升 40% | 新配置中的超时值单位错误（ms → s）| Phase 4 对比新旧配置差异 |

## 8. 测试策略

- **Diagnosis Agent 单元测试**：每个 Phase 独立测试，验证 Phase 不跳过、输出格式正确、confidence 计算合理
- **Decision Agent 单元测试**：验证每个 Level 的操作判定正确、Level 4 操作被拒绝、规则覆盖范围完整
- **Supervisor 集成测试**：端到端测试完整诊断链路，验证 Agent 协作顺序正确
- **场景回归测试**：每个故障场景一条 golden path 测试，确保场景不会被"误修复"
- **不做的测试**：不 mock LLM 调用（测试中使用真实 API，但限制 max_tokens 降低成本）；不测试 LangGraph 框架本身

## 9. 不在 MVP 范围内的

- 感知模块（Prometheus/ELK/Kafka 接入）
- 执行模块的真实操作（限流/重启/DDL）
- 验证模块的真实操作后验证
- 钉钉/企微机器人对接
- 知识库持久化和语义检索（MVP 用 JSON 文件 + 简单关键词匹配）
- LangGraph checkpointer 持久化（MVP 用 MemorySaver）
- 多故障并发处理
