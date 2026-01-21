# Agentic Rubrics RL - 详细架构文档

## 系统架构流程图

```
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                              Phase 1: Offline Rubric Synthesis (数据准备阶段)                            │
│                                                                                                         │
│  ┌──────────────────────┐                                                                               │
│  │   Dataset D          │                                                                               │
│  │   {x_i, spec?, verification_data?}       │                                                                               │
│  │                      │                                                                               │
│  │  DataProto:          │                                                                               │
│  │  - prompt: str       │                                                                               │
│  │  - specification?    │                                                                               │
│  │  - verification_data? (I/O pairs)    │                                                                               │
│  └──────────┬───────────┘                                                                               │
│             │                                                                                           │
│             │  Input: (x_i, spec?, verification_data?)                                                                      │
│             v                                                                                           │
│  ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐  │
│  │                          Rubric Generator G_ψ (两阶段编译器)                                       │  │
│  │                                                                                                    │  │
│  │   Input Schema:                                                                                    │  │
│  │   ┌─────────────────────────────────────────────────────────────────────────────────────────┐     │  │
│  │   │ {                                                                                       │     │  │
│  │   │   "task_input": str,           # 任务描述 x_i                                           │     │  │
│  │   │   "specification": str | None, # 可选的规格说明                                          │     │  │
│  │   │   "verification_data": [...],    # unit-test I/O pairs (optional)│     │  │
│  │   │   "tool_schemas": [            # T_e^rub = T_e^pol ∪ T_e^ver                            │     │  │
│  │   │     {                                                                                   │     │  │
│  │   │       "name": str,                                                                      │     │  │
│  │   │       "description": str,                                                               │     │  │
│  │   │       "parameters": JSONSchema,                                                         │     │  │
│  │   │       "returns": JSONSchema     # 重要: 包含返回值schema                                  │     │  │
│  │   │     }                                                                                   │     │  │
│  │   │   ]                                                                                     │     │  │
│  │   │ }                                                                                       │     │  │
│  │   └─────────────────────────────────────────────────────────────────────────────────────────┘     │  │
│  │                                                                                                    │  │
│  │   ┌─────────────────────────┐      ┌─────────────────────────────────────┐                        │  │
│  │   │ Stage 1: Intent         │      │ Stage 2: Tool Compilation           │                        │  │
│  │   │ Extraction              │      │                                     │                        │  │
│  │   │                         │      │ Input: (x_i, d_i, T_e^rub, verification_data?)          │                        │  │
│  │   │ x_i ──────────────────> │ d_i  │                                     │                        │  │
│  │   │                         │ ───> │ Output:                             │                        │  │
│  │   │ Output: direction d_i   │      │   ExecutableRubric ρ_i              │                        │  │
│  │   │ {                       │      │   {                                 │                        │  │
│  │   │   task_intent: str,     │      │     id: str,                        │                        │  │
│  │   │   verification_dims: [] │      │     verification_checklist: V,      │                        │  │
│  │   │ }                       │      │     evidence_plans: Φ,              │                        │  │
│  │   │                         │      │     aggregation: Agg,               │                        │  │
│  │   └─────────────────────────┘      │     estimated_latency_ms: L_i       │                        │  │
│  │                                    │   }                                 │                        │  │
│  │                                    └─────────────────────────────────────┘                        │  │
│  └──────────────────────────────────────────────────────────────────────────────────────────────────┘  │
│             │                                                                                           │
│             │  Output: (ρ_i, L_i)                                                                       │
│             v                                                                                           │
│  ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐  │
│  │                              Enriched Dataset D' (按延迟分桶)                                      │  │
│  │                                                                                                    │  │
│  │   Storage Format (via extra_info):                                                                 │  │
│  │   ┌─────────────────────────────────────────────────────────────────────────────────────────┐     │  │
│  │   │ {                                                                                       │     │  │
│  │   │   "prompt": [...],                                                                      │     │  │
│  │   │   "extra_info": {                                                                       │     │  │
│  │   │     "rubric": ExecutableRubric.model_dump(),  # 完整的 ρ_i 序列化                        │     │  │
│  │   │     "latency_bucket": int (0-4)               # 预计算的延迟桶ID                         │     │  │
│  │   │   },                                                                                    │     │  │
│  │   │   "data_source": "agentic_rubric"                                                       │     │  │
│  │   │ }                                                                                       │     │  │
│  │   └─────────────────────────────────────────────────────────────────────────────────────────┘     │  │
│  │                                                                                                    │  │
│  │   Latency Buckets:                                                                                 │  │
│  │   ┌─────────┬─────────┬─────────┬─────────┬─────────┐                                             │  │
│  │   │ Bucket 0│ Bucket 1│ Bucket 2│ Bucket 3│ Bucket 4│                                             │  │
│  │   │ <1s     │ 1-5s    │ 5-15s   │ 15-30s  │ >30s    │                                             │  │
│  │   └─────────┴─────────┴─────────┴─────────┴─────────┘                                             │  │
│  └──────────────────────────────────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                                     │
                                                     │ D' = {(x_i, ρ_i, L_i)}
                                                     v
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                              Phase 2: Online RL Training (在线训练阶段)                                  │
│                                                                                                         │
│  ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐  │
│  │                              1. Policy Rollout (策略生成)                                          │  │
│  │                                                                                                    │  │
│  │   ┌────────────────────┐                          ┌────────────────────────────────────┐          │  │
│  │   │   Policy π_θ       │                          │   T_e^pol (Policy可见工具)          │          │  │
│  │   │                    │                          │                                    │          │  │
│  │   │   Input:           │      Tool Calls          │   Available Tools:                 │          │  │
│  │   │   - x (task)       │  ─────────────────────>  │   - sandbox_fusion (代码执行)      │          │  │
│  │   │   - T_e^pol        │      FunctionCall:       │   - web_search (搜索)              │          │  │
│  │   │                    │      {                   │   - file_read/write                │          │  │
│  │   │   State Machine:   │        name: str,        │   - ...                            │          │  │
│  │   │   PENDING →        │        arguments: JSON   │                                    │          │  │
│  │   │   GENERATING →     │      }                   │   Tool Response (ToolResponse):   │          │  │
│  │   │   PROCESSING_TOOLS │  <─────────────────────  │   {                                │          │  │
│  │   │   → ... →          │      ToolResponse:       │     text: str,                     │          │  │
│  │   │   TERMINATED       │      {                   │     image?: PIL.Image,             │          │  │
│  │   │                    │        text: str,        │     video?: Tensor                 │          │  │
│  │   │                    │        image?: Image,    │   }                                │          │  │
│  │   │                    │        video?: Tensor    │                                    │          │  │
│  │   │                    │      }                   │                                    │          │  │
│  │   └────────┬───────────┘                          └────────────────────────────────────┘          │  │
│  │            │                                                                                       │  │
│  │            │  Output: (y, τ) ~ π_θ(· | x; T_e^pol)                                                │  │
│  │            │                                                                                       │  │
│  │            │  ┌──────────────────────────────────────────────────────────────────────┐            │  │
│  │            │  │ Policy Output Schema:                                                │            │  │
│  │            │  │ {                                                                    │            │  │
│  │            │  │   "y": str,                    # 最终响应文本                         │            │  │
│  │            │  │   "τ": [                       # 交互轨迹 (Interaction Trace)         │            │  │
│  │            │  │     {                                                                │            │  │
│  │            │  │       "tool_name": str,        # t_k ∈ T_e^pol                       │            │  │
│  │            │  │       "arguments": dict,       # a_k                                 │            │  │
│  │            │  │       "observation": Any       # o_k (工具返回)                       │            │  │
│  │            │  │     }, ...                                                           │            │  │
│  │            │  │   ],                                                                 │            │  │
│  │            │  │   "response_ids": List[int],   # Token IDs                           │            │  │
│  │            │  │   "response_mask": List[int],  # 训练mask                            │            │  │
│  │            │  │   "metrics": dict              # 性能指标                             │            │  │
│  │            │  │ }                                                                    │            │  │
│  │            │  └──────────────────────────────────────────────────────────────────────┘            │  │
│  │            v                                                                                       │  │
│  └──────────────────────────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                                         │
│  ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐  │
│  │                              2. Rubric Execution (Rubric 执行)                                     │  │
│  │                                                                                                    │  │
│  │   ┌─────────────────────────────────────────────────────────────────────────────────────────────┐ │  │
│  │   │                          Rubric Executor R                                                   │ │  │
│  │   │                                                                                              │ │  │
│  │   │   Input:                                                                                     │ │  │
│  │   │   ┌────────────────────────────────────────────────────────────────────────────────────┐    │ │  │
│  │   │   │ {                                                                                  │    │ │  │
│  │   │   │   "task_input": str,         # x_i                                                 │    │ │  │
│  │   │   │   "policy_output": str,      # y                                                   │    │ │  │
│  │   │   │   "policy_trace": [          # τ = {(t_k, a_k, o_k)}                               │    │ │  │
│  │   │   │     {"tool_name", "arguments", "observation"}, ...                                 │    │ │  │
│  │   │   │   ],                                                                               │    │ │  │
│  │   │   │   "rubric": ExecutableRubric # ρ_i                                                 │    │ │  │
│  │   │   │ }                                                                                  │    │ │  │
│  │   │   └────────────────────────────────────────────────────────────────────────────────────┘    │ │  │
│  │   │                                                                                              │ │  │
│  │   │   ┌───────────────────────────────────────────────────────────────────────────────────────┐ │ │  │
│  │   │   │                    Evidence Collection (证据收集)                                       │ │ │  │
│  │   │   │                                                                                        │ │ │  │
│  │   │   │   for each plan in ρ.evidence_plans:                                                   │ │ │  │
│  │   │   │       ┌─────────────────────┐                    ┌─────────────────────────────────┐  │ │ │  │
│  │   │   │       │ EvidencePlan        │     Tool Call      │ T_e^rub (全部工具)               │  │ │ │  │
│  │   │   │       │                     │  ───────────────>  │                                 │  │ │ │  │
│  │   │   │       │ check_id: str       │  {                 │ T_e^pol (Policy工具)            │  │ │ │  │
│  │   │   │       │ tool_name: str      │    tool: str,      │   + sandbox_fusion              │  │ │ │  │
│  │   │   │       │ tool_arguments: {}  │    args: dict      │   + web_search                  │  │ │ │  │
│  │   │   │       │ argument_extractor  │  }                 │   + ...                         │  │ │ │  │
│  │   │   │       │ timeout_ms: int     │                    │                                 │  │ │ │  │
│  │   │   │       │                     │  <───────────────  │ T_e^ver (验证专用工具)           │  │ │ │  │
│  │   │   │       │ (Jinja2 template    │  Tool Output +     │   + test_runner                 │  │ │ │  │
│  │   │   │       │  extracts args      │  Provenance        │   + ground_truth_checker        │  │ │ │  │
│  │   │   │       │  from y/τ)          │                    │   + answer_key_lookup           │  │ │ │  │
│  │   │   │       └─────────────────────┘                    └─────────────────────────────────┘  │ │ │  │
│  │   │   │                                                                                        │ │ │  │
│  │   │   │   ┌──────────────────────────────────────────────────────────────────────────────────┐│ │ │  │
│  │   │   │   │ EvidenceRecord (证据记录) - 包含完整的审计追踪                                     ││ │ │  │
│  │   │   │   │ {                                                                                ││ │ │  │
│  │   │   │   │   "check_id": str,           # 关联的验证项ID                                    ││ │ │  │
│  │   │   │   │   "tool_name": str,          # 调用的工具名称                                    ││ │ │  │
│  │   │   │   │   "tool_input": dict,        # 实际传入的参数                                    ││ │ │  │
│  │   │   │   │   "tool_output": Any,        # 工具返回的原始输出                                ││ │ │  │
│  │   │   │   │   "success": bool,           # 执行是否成功                                      ││ │ │  │
│  │   │   │   │   "status": "ok"|"error"|"timeout",                                              ││ │ │  │
│  │   │   │   │   "error_message": str|null, # 错误信息                                          ││ │ │  │
│  │   │   │   │   // Provenance (来源追踪)                                                       ││ │ │  │
│  │   │   │   │   "start_time_ms": float,    # 开始时间戳                                        ││ │ │  │
│  │   │   │   │   "end_time_ms": float,      # 结束时间戳                                        ││ │ │  │
│  │   │   │   │   "latency_ms": float        # 实际执行延迟                                      ││ │ │  │
│  │   │   │   │ }                                                                                ││ │ │  │
│  │   │   │   └──────────────────────────────────────────────────────────────────────────────────┘│ │ │  │
│  │   │   │                                                                                        │ │ │  │
│  │   │   │   Output: E = {e_1, ..., e_J}  # 证据记录集合                                          │ │ │  │
│  │   │   └───────────────────────────────────────────────────────────────────────────────────────┘ │ │  │
│  │   │                                                                                              │ │  │
│  │   │   ┌───────────────────────────────────────────────────────────────────────────────────────┐ │ │  │
│  │   │   │                    Scoring & Aggregation (评分与聚合)                                   │ │ │  │
│  │   │   │                                                                                        │ │ │  │
│  │   │   │   for each check v_j in ρ.verification_checklist:                                      │ │ │  │
│  │   │   │                                                                                        │ │ │  │
│  │   │   │       ┌──────────────────────────────────────────────────────────────────────────┐    │ │ │  │
│  │   │   │       │ VerificationItem v_j                                                     │    │ │ │  │
│  │   │   │       │ {                                                                        │    │ │ │  │
│  │   │   │       │   "id": str,                  # 验证项唯一标识                            │    │ │ │  │
│  │   │   │       │   "description": str,         # 验证内容描述                              │    │ │ │  │
│  │   │   │       │   "category": str,            # "correctness"|"style"|"safety"           │    │ │ │  │
│  │   │   │       │   "weight": float,            # 聚合权重 (default: 1.0)                  │    │ │ │  │
│  │   │   │       │   "is_gate": bool             # 硬门控 (失败则总分为0)                    │    │ │ │  │
│  │   │   │       │ }                                                                        │    │ │ │  │
│  │   │   │       └──────────────────────────────────────────────────────────────────────────┘    │ │ │  │
│  │   │   │                                      │                                                 │ │ │  │
│  │   │   │       evidence_j = filter(E, check_id == v_j.id)                                       │ │ │  │
│  │   │   │       score_j = φ_j(y, evidence_j)   # 评分函数                                        │ │ │  │
│  │   │   │                                      │                                                 │ │ │  │
│  │   │   │                                      v                                                 │ │ │  │
│  │   │   │       ┌──────────────────────────────────────────────────────────────────────────┐    │ │ │  │
│  │   │   │       │ AggregationRule                                                          │    │ │ │  │
│  │   │   │       │ {                                                                        │    │ │ │  │
│  │   │   │       │   "method": "weighted_sum"|"min"|"product",                              │    │ │ │  │
│  │   │   │       │   "normalize": bool                                                      │    │ │ │  │
│  │   │   │       │ }                                                                        │    │ │ │  │
│  │   │   │       │                                                                          │    │ │ │  │
│  │   │   │       │ Gate Check: if any v_j.is_gate && score_j < 0.5 → R = 0                 │    │ │ │  │
│  │   │   │       │                                                                          │    │ │ │  │
│  │   │   │       │ Aggregation:                                                             │    │ │ │  │
│  │   │   │       │   weighted_sum: Σ(w_j * s_j) / Σ(w_j)                                   │    │ │ │  │
│  │   │   │       │   min:          min(s_j)                                                 │    │ │ │  │
│  │   │   │       │   product:      Π(s_j)                                                   │    │ │ │  │
│  │   │   │       └──────────────────────────────────────────────────────────────────────────┘    │ │ │  │
│  │   │   │                                                                                        │ │ │  │
│  │   │   └───────────────────────────────────────────────────────────────────────────────────────┘ │ │  │
│  │   │                                                                                              │ │  │
│  │   │   Output:                                                                                    │ │  │
│  │   │   ┌────────────────────────────────────────────────────────────────────────────────────┐    │ │  │
│  │   │   │ {                                                                                  │    │ │  │
│  │   │   │   "reward_score": float,              # R(x, y) ∈ [0, 1]                           │    │ │  │
│  │   │   │   "reward_extra_info": {                                                           │    │ │  │
│  │   │   │     "rubric_id": str,                                                              │    │ │  │
│  │   │   │     "check_scores": {check_id: score, ...},                                        │    │ │  │
│  │   │   │     "evidence_count": int,                                                         │    │ │  │
│  │   │   │     "evidence_success_rate": float,                                                │    │ │  │
│  │   │   │     "evidence_records": [EvidenceRecord, ...]  # 完整审计追踪                       │    │ │  │
│  │   │   │   }                                                                                │    │ │  │
│  │   │   │ }                                                                                  │    │ │  │
│  │   │   └────────────────────────────────────────────────────────────────────────────────────┘    │ │  │
│  │   └─────────────────────────────────────────────────────────────────────────────────────────────┘ │  │
│  └──────────────────────────────────────────────────────────────────────────────────────────────────┘  │
│                                                                                                         │
│  ┌──────────────────────────────────────────────────────────────────────────────────────────────────┐  │
│  │                              3. GRPO/PPO Optimization (策略优化)                                   │  │
│  │                                                                                                    │  │
│  │   Objective: J(θ) = E_{(x,ρ)~D', (y,τ)~π_θ}[R(x, y, τ)]                                          │  │
│  │                                                                                                    │  │
│  │   ┌─────────────────┐     rewards {r_i}     ┌─────────────────────────────────────────┐           │  │
│  │   │   Policy π_θ    │ <──────────────────── │ Collected from Rubric Executor          │           │  │
│  │   │                 │                       │                                         │           │  │
│  │   │   θ ← θ + α∇_θ J(θ)                    │ Per-sample rewards with detailed        │           │  │
│  │   │                 │                       │ breakdown for analysis                  │           │  │
│  │   │   Gradient via  │                       │                                         │           │  │
│  │   │   PPO/GRPO      │ ───────────────────>  │ Updated policy for next rollout         │           │  │
│  │   └─────────────────┘     updated π_θ       └─────────────────────────────────────────┘           │  │
│  │                                                                                                    │  │
│  └──────────────────────────────────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Environment Contract (环境契约)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Environment Contract                               │
│                                                                             │
│  1. Discovery (工具发现)                                                     │
│     ┌─────────────────────────────────────────────────────────────────┐    │
│     │ Tool Schema Format:                                              │    │
│     │ {                                                                │    │
│     │   "name": str,               # 工具唯一名称                       │    │
│     │   "description": str,        # 工具功能描述                       │    │
│     │   "parameters": {            # JSON Schema for input             │    │
│     │     "type": "object",                                            │    │
│     │     "properties": {...},                                         │    │
│     │     "required": [...]                                            │    │
│     │   },                                                             │    │
│     │   "returns": {               # JSON Schema for output (重要!)    │    │
│     │     "type": "object",                                            │    │
│     │     "properties": {...}                                          │    │
│     │   }                                                              │    │
│     │ }                                                                │    │
│     └─────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  2. Invocation (工具调用)                                                    │
│     - 支持无状态调用 (stateless execution)                                  │
│     - 支持会话调用 (session-based execution)                                │
│     - 来自 Policy 和 Rubric Executor 的调用使用相同接口                      │
│                                                                             │
│  3. Provenance (来源追踪)                                                    │
│     ┌─────────────────────────────────────────────────────────────────┐    │
│     │ Every tool execution returns:                                    │    │
│     │ {                                                                │    │
│     │   "output": Any,             # 工具的实际输出                     │    │
│     │   "provenance": {                                                │    │
│     │     "start_time_ms": float,  # 执行开始时间                       │    │
│     │     "end_time_ms": float,    # 执行结束时间                       │    │
│     │     "latency_ms": float,     # 执行延迟                           │    │
│     │     "status": str,           # "ok" | "error" | "timeout"        │    │
│     │     "error_message": str?    # 错误信息 (如有)                    │    │
│     │   }                                                              │    │
│     │ }                                                                │    │
│     └─────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Tool Separation (工具隔离)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Tool Separation Model                             │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                    T_e (All Environment Tools)                       │  │
│   │                                                                      │  │
│   │   ┌─────────────────────────┐   ┌─────────────────────────────────┐ │  │
│   │   │   T_e^pol               │   │   T_e^ver                       │ │  │
│   │   │   (Policy-visible)      │   │   (Verification-only)           │ │  │
│   │   │                         │   │                                 │ │  │
│   │   │   - sandbox_fusion      │   │   - test_runner (私有测试用例)  │ │  │
│   │   │   - web_search          │   │   - answer_key_lookup           │ │  │
│   │   │   - file_read           │   │   - ground_truth_checker        │ │  │
│   │   │   - file_write          │   │   - expert_simulator            │ │  │
│   │   │   - calculator          │   │   - adversarial_tester          │ │  │
│   │   │                         │   │                                 │ │  │
│   │   │   Config:               │   │   Config:                       │ │  │
│   │   │   policy_tools.json     │   │   verification_tools.json       │ │  │
│   │   └─────────────────────────┘   └─────────────────────────────────┘ │  │
│   │                                                                      │  │
│   │   ┌─────────────────────────────────────────────────────────────┐   │  │
│   │   │            T_e^rub = T_e^pol ∪ T_e^ver                      │   │  │
│   │   │            (Rubric Generator & Executor see all)             │   │  │
│   │   │                                                              │   │  │
│   │   │            Merged during offline synthesis:                  │   │  │
│   │   │            - Deduplication by tool name                      │   │  │
│   │   │            - Policy tools take precedence                    │   │  │
│   │   └─────────────────────────────────────────────────────────────┘   │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   Access Control:                                                           │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │ Component              │ Tool Access                                │  │
│   ├────────────────────────┼────────────────────────────────────────────┤  │
│   │ Policy π_θ             │ T_e^pol only                               │  │
│   │ Rubric Generator G_ψ   │ T_e^rub (schemas only, no execution)       │  │
│   │ Rubric Executor R      │ T_e^rub (full execution access)            │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 组件参考表

| 组件名称 | 描述 | 详细说明位置 | 代码实现路径 | 状态 |
|---------|------|-------------|-------------|------|
| **核心数据结构** |||||
| `ExecutableRubric` | 可执行的 Rubric 程序 ρ = ⟨V, Φ, Agg⟩ | [Plan.md#核心数据结构](Plan.md#核心数据结构) | [`verl/rubric/schemas.py`](verl/rubric/schemas.py) | 🔲 待实现 |
| `VerificationItem` | 验证项 v_j ∈ V | [Plan.md#核心数据结构](Plan.md#核心数据结构) | [`verl/rubric/schemas.py`](verl/rubric/schemas.py) | 🔲 待实现 |
| `EvidencePlan` | 证据收集计划 φ_j ∈ Φ | [Plan.md#核心数据结构](Plan.md#核心数据结构) | [`verl/rubric/schemas.py`](verl/rubric/schemas.py) | 🔲 待实现 |
| `EvidenceRecord` | 证据记录 e_j (含 Provenance) | [Plan.md#核心数据结构](Plan.md#核心数据结构) | [`verl/rubric/schemas.py`](verl/rubric/schemas.py) | 🔲 待实现 |
| `AggregationRule` | 聚合规则 Agg | [Plan.md#核心数据结构](Plan.md#核心数据结构) | [`verl/rubric/schemas.py`](verl/rubric/schemas.py) | 🔲 待实现 |
| **Rubric Generator** |||||
| `BaseRubricGenerator` | Rubric 生成器基类 | [Plan.md#1-rubric-generator](Plan.md#1-rubric-generator-离线两阶段) | [`verl/rubric/generator/base.py`](verl/rubric/generator/base.py) | 🔲 待实现 |
| `LLMRubricGenerator` | LLM 驱动的两阶段生成器 (GPT-4o) | [Plan.md#1-rubric-generator](Plan.md#1-rubric-generator-离线两阶段) | [`verl/rubric/generator/llm_generator.py`](verl/rubric/generator/llm_generator.py) | 🔲 待实现 |
| **Rubric Executor** |||||
| `BaseRubricExecutor` | Rubric 执行器基类 | [Plan.md#2-rubric-executor](Plan.md#2-rubric-executor-在线) | [`verl/rubric/executor/base.py`](verl/rubric/executor/base.py) | 🔲 待实现 |
| `DefaultRubricExecutor` | 默认 Rubric 执行器 | [Plan.md#2-rubric-executor](Plan.md#2-rubric-executor-在线) | [`verl/rubric/executor/default_executor.py`](verl/rubric/executor/default_executor.py) | 🔲 待实现 |
| **Reward Manager** |||||
| `RewardManagerBase` | 奖励管理器基类 (verl 现有) | - | [`verl/experimental/reward_loop/reward_manager/base.py`](verl/experimental/reward_loop/reward_manager/base.py) | ✅ 已存在 |
| `RubricRewardManager` | **⭐ Rubric 奖励管理器** | [Plan.md#3-rubric-reward-manager](Plan.md#3-rubric-reward-manager-关键集成点) | [`verl/experimental/reward_loop/reward_manager/rubric_manager.py`](verl/experimental/reward_loop/reward_manager/rubric_manager.py) | 🔲 待实现 |
| **采样器** |||||
| `LatencyBucketedSampler` | 延迟分桶采样器 | [Plan.md#4-latency-bucketed-sampler](Plan.md#4-latency-bucketed-sampler) | [`verl/experimental/dataset/latency_bucketed_sampler.py`](verl/experimental/dataset/latency_bucketed_sampler.py) | 🔲 待实现 |
| **Agent Loop (verl 现有)** |||||
| `AgentLoopBase` | Agent Loop 基类 | - | [`verl/experimental/agent_loop/agent_loop.py`](verl/experimental/agent_loop/agent_loop.py) | ✅ 已存在 |
| `ToolAgentLoop` | 工具调用 Agent Loop | - | [`verl/experimental/agent_loop/tool_agent_loop.py`](verl/experimental/agent_loop/tool_agent_loop.py) | ✅ 已存在 |
| `AgentData` | Agent 状态数据容器 | - | [`verl/experimental/agent_loop/tool_agent_loop.py`](verl/experimental/agent_loop/tool_agent_loop.py) | ✅ 已存在 |
| **工具系统 (verl 现有)** |||||
| `BaseTool` | 工具基类 | - | `verl/tools/base.py` | ✅ 已存在 |
| `ToolResponse` | 工具响应结构 | - | [`verl/tools/schemas.py`](verl/tools/schemas.py) | ✅ 已存在 |

---

## 数据流详解

### Phase 1: 离线 Rubric 合成

```
Input Dataset               Rubric Generator              Output Dataset
┌────────────┐              ┌─────────────┐              ┌────────────────────┐
│ x_i        │──────────────│ G_ψ         │──────────────│ x_i                │
│ spec?      │  task_input  │             │  ρ_i, L_i    │ ρ_i (rubric)       │
│            │  + spec      │ Stage 1:    │              │ L_i (latency_bucket)│
│            │  + T_e^rub + verification_data?   │  Intent     │              │                    │
│            │              │  Extraction │              │                    │
│            │              │             │              │                    │
│            │              │ Stage 2:    │              │                    │
│            │              │  Tool       │              │                    │
│            │              │  Compilation│              │                    │
└────────────┘              └─────────────┘              └────────────────────┘
```

### Phase 2: 在线 RL 训练

```
         ┌──────────────────────────────────────────────────────────────────┐
         │                                                                  │
         v                                                                  │
┌────────────────┐      ┌────────────────┐      ┌────────────────┐         │
│  Policy π_θ    │      │  T_e^pol       │      │  Rubric        │         │
│                │      │  (Tools)       │      │  Executor R    │         │
│  Input: x      │══════│                │      │                │         │
│                │ tool │  execute()     │      │  Input:        │         │
│  Output:       │ call │  ────────────> │      │  - x           │         │
│  - y (text)    │══════│  <────────────  │      │  - y           │         │
│  - τ (trace)   │ resp │  ToolResponse  │      │  - τ           │         │
└───────┬────────┘      └────────────────┘      │  - ρ           │         │
        │                                       │                │         │
        │ (y, τ)                                │  Uses T_e^rub: │         │
        │                                       │  ┌───────────┐ │         │
        └───────────────────────────────────────│──│ Evidence  │ │         │
                                                │  │ Collection│ │         │
                                                │  └─────┬─────┘ │         │
                                                │        │       │         │
                                                │        v       │         │
                                                │  ┌───────────┐ │         │
                                                │  │ Scoring   │ │         │
                                                │  │ & Agg     │ │         │
                                                │  └─────┬─────┘ │         │
                                                │        │       │         │
                                                └────────┼───────┘         │
                                                         │                 │
                                                         │ R(x,y)          │
                                                         │ + evidence      │
                                                         v                 │
                                                ┌────────────────┐         │
                                                │  GRPO/PPO      │         │
                                                │  Optimizer     │─────────┘
                                                │                │  θ update
                                                │  ∇_θ J(θ)      │
                                                └────────────────┘
```

---

## 文件结构

```
verl/
├── rubric/                                    # 🆕 Rubric 核心模块
│   ├── __init__.py                           # ✅ 已创建 (导出接口)
│   ├── schemas.py                            # 🔲 数据结构定义
│   ├── generator/
│   │   ├── __init__.py                       # 🔲 Generator 模块入口
│   │   ├── base.py                           # 🔲 BaseRubricGenerator
│   │   └── llm_generator.py                  # 🔲 LLMRubricGenerator
│   └── executor/
│       ├── __init__.py                       # 🔲 Executor 模块入口
│       ├── base.py                           # 🔲 BaseRubricExecutor
│       └── default_executor.py               # 🔲 DefaultRubricExecutor
│
├── experimental/
│   ├── dataset/
│   │   └── latency_bucketed_sampler.py       # 🔲 延迟分桶采样器
│   │
│   └── reward_loop/
│       └── reward_manager/
│           ├── base.py                        # ✅ RewardManagerBase
│           ├── registry.py                    # ✅ 注册表
│           └── rubric_manager.py              # 🔲 ⭐ RubricRewardManager
│
└── tools/                                     # ✅ 现有工具系统
    ├── base.py
    └── schemas.py

examples/agentic_rubric/                       # 🆕 示例
├── config/
│   ├── rubric_rl_grpo.yaml                   # 🔲 训练配置
│   └── tool_config/
│       ├── policy_tools.json                 # 🔲 Policy 可见工具
│       └── verification_tools.json           # 🔲 验证专用工具
└── data_preprocess/
    └── rubric_dataset.py                     # 🔲 离线合成脚本

tests/
├── rubric/
│   ├── test_schemas.py                       # 🔲 数据结构测试
│   └── test_executor.py                      # 🔲 执行器测试
└── experimental/
    └── reward_loop/
        └── test_rubric_manager.py            # 🔲 奖励管理器测试
```

---

## 关键接口定义

### ExecutableRubric 完整结构

```python
class ExecutableRubric(BaseModel):
    """
    可执行的 Rubric 程序
    ρ = ⟨V, Φ, Agg⟩

    参考: framework.tex Section 2.4 (Definition of an Executable Rubric)
    """
    id: str                                      # 唯一标识
    task_intent: str                             # 从 x 提取的任务意图
    verification_checklist: list[VerificationItem]  # V: 验证项列表
    evidence_plans: list[EvidencePlan]           # Φ: 证据收集计划
    aggregation: AggregationRule                 # Agg: 聚合规则
    estimated_latency_ms: int = 0                # L: 预估执行延迟
```

### Rubric Executor 接口

```python
class BaseRubricExecutor(ABC):
    """
    Rubric 执行器基类

    R(x, y, τ, ρ; T_e^rub) → (reward, evidence_records, details)

    参考: framework.tex Section 2.2 (The Rubric Executor)
    """

    @abstractmethod
    async def execute(
        self,
        task_input: str,           # x
        policy_output: str,        # y
        policy_trace: list[dict],  # τ = {(t_k, a_k, o_k)}
        rubric: ExecutableRubric,  # ρ
        tools: dict[str, Any],     # T_e^rub
    ) -> tuple[float, list[EvidenceRecord], dict]:
        """
        执行 Rubric 并返回奖励分数

        Returns:
            reward: float - 最终奖励 R(x, y) ∈ [0, 1]
            evidence_records: list[EvidenceRecord] - 证据记录 E
            details: dict - 详细评分信息
        """
        pass
```

### RubricRewardManager 接口

```python
@register("rubric")
class RubricRewardManager(RewardManagerBase):
    """
    Rubric 奖励管理器

    集成到 verl 的 Agent Loop 体系
    通过 reward_model.reward_manager: "rubric" 配置启用

    参考: Plan.md#3-rubric-reward-manager
    """

    async def run_single(self, data: DataProto) -> dict:
        """
        从 DataProto 提取数据，执行 Rubric，返回奖励

        Input (via DataProto):
            - extra_info["rubric"]: ExecutableRubric 序列化
            - batch["responses"]: Policy 输出 token IDs
            - non_tensor_batch["raw_prompt"]: 原始 prompt
            - non_tensor_batch["tool_extra_fields"]["tool_calls"]: Policy trace

        Returns:
            {
                "reward_score": float,
                "reward_extra_info": {
                    "rubric_id": str,
                    "check_scores": dict,
                    "evidence_count": int,
                    "evidence_success_rate": float
                }
            }
        """
        pass
```
