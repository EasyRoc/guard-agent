# Guard Agent

遵循 SRE 最佳实践的智能化运维助手——结构化故障诊断、四级安全边界、人机协作。

## 架构

```
决策与诊断层（大脑）
┌──────────────┐  ┌──────────────┐
│  Supervisor   │  │   Decision    │
│    Agent      │──│    Agent      │
│  (编排调度)    │  │ (安全边界+决策) │
└──────┬───────┘  └──────────────┘
       │
┌──────┴───────┐
│  Diagnosis    │
│    Agent      │
│ (四阶段诊断)   │
└──────────────┘
```

- **Diagnosis Agent**：四阶段 StateGraph（信息收集 → 数据追踪 → 根因定位 → 验证修复），置信度 < 0.6 自动跳过修复建议
- **Decision Agent**：规则引擎优先（Level 1-4 操作分级）+ LLM 兜底
- **Supervisor Agent**：LangGraph ReAct agent 编排 diagnosis → decision → confirmation

## 快速开始

```bash
# 安装
pip install -e ".[dev]"

# 设置 API Key
export LLM_API_KEY=<your-deepseek-key>
# 可选：自定义模型和 Base URL
export LLM_MODEL=deepseek-chat
export LLM_BASE_URL=https://api.deepseek.com/v1

# 运行诊断
python -m src.main connection_pool --auto-confirm

# 交互模式（需人工确认 Level 2/3 操作）
python -m src.main oom

# 输出 JSON 报告
python -m src.main config_change -o report.json
```

## CLI 用法

```
python -m src.main [场景] [选项]

场景:
  connection_pool   数据库连接池耗尽
  oom               内存溢出 OOM
  config_change     配置变更导致服务异常

选项:
  --auto-confirm    跳过人工确认
  -o, --output      输出 JSON 报告到文件
```

## 四级安全边界

| Level | 说明 | 示例操作 |
|---|---|---|
| 1 自动批准 | 只读操作，无副作用 | query_metrics, query_logs |
| 2 需确认 | 有副作用，影响可控 | restart_pod, add_index |
| 3 需审批 | 高风险，影响核心业务 | execute_ddl, deploy_rollback |
| 4 完全禁止 | Agent 不会提议 | drop_database, rm_rf |

## 运行测试

```bash
# 非 LLM 测试（不需要 API Key）
pytest tests/ -v -k "not diagnosis_agent and not decision_llm and not supervisor and not integration"

# 全部测试（需要 LLM_API_KEY）
pytest tests/ -v
```

## 项目结构

```
src/
├── main.py              # CLI 入口
├── models/              # 数据模型
│   ├── alert.py         # AlertEvent
│   ├── diagnosis.py     # DiagnosisState, DiagnosisReport
│   └── decision.py      # OperationPlan, DecisionProposal
├── agents/              # Agent 实现
│   ├── supervisor.py    # Supervisor Agent (编排)
│   ├── diagnosis.py     # Diagnosis Agent (四阶段 StateGraph)
│   └── decision.py      # Decision Agent (规则引擎 + LLM)
├── knowledge/           # 故障模式库
│   ├── fault_patterns.py
│   └── fault_patterns.json
├── safety/              # 安全边界
│   └── operation_levels.py
├── tools/               # 诊断工具
│   ├── diagnostic.py    # LangChain tool 函数
│   └── mock_data.py     # Mock 场景数据加载
└── utils/
    ├── llm.py           # LLM 客户端工厂
    └── logging.py       # 日志配置
```
