# Plan: CCPlus-1x-Optimized 数据集训练准备

## Context

基于已实现的 agentic rubric RL 框架（Plan.md），使用 `ccplus-1x-optimized` 竞赛编程数据集进行训练。数据集有 2,407 道题，包含 test_cases 和参考解法（`time_opt`/`memory_opt`）。

**核心需求**：
1. 测试用例通过性（gate 检查，不通过则所有 reward=0）
2. **时间效率**（独立信号）：runtime 运行模型代码和 `time_opt` 参考解法，以 `参考时间/模型时间` 作为 reward
3. **内存效率**（独立信号）：runtime 运行模型代码和 `memory_opt` 参考解法，以 `参考内存/模型内存` 作为 reward
4. 三个信号（correctness, time_ratio, memory_ratio）独立返回，correctness 是 gate
5. **训练过程中每个 sample 的详细信息记录到 SwanLab**（per-sample table logging）

---

## 已完成的改动

### 1. `verl/rubric/schemas.py` — 添加 `PERFORMANCE_RATIO` 枚举 ✅

在 `ScoringPrimitive` 枚举中添加:
```python
PERFORMANCE_RATIO = "performance_ratio"
```

### 2. `verl/rubric/executor/primitives.py` — 添加 PerformanceRatioPrimitive ✅

新增评分原语类 `PerformanceRatioPrimitive(BaseScoringPrimitive)`，比较模型代码和参考代码的单一运行时指标。

**核心逻辑**：
- 通过 evidence 中 `tool_input["role"]` 区分 `"model"` 和 `"reference"` 两组证据
- `scoring_config.metric_field`：tool_output 中的指标字段名（如 `"execution_time_ms"`）
- `scoring_config.direction`：`"lower_is_better"`（默认）或 `"higher_is_better"`
- 评分公式（lower_is_better）：`score = min(1.0, reference_value / model_value)`
- 评分公式（higher_is_better）：`score = min(1.0, model_value / reference_value)`
- 处理除零、缺失证据、类型转换等边界情况

注册到 `_SCORING_PRIMITIVES` 字典：
```python
ScoringPrimitive.PERFORMANCE_RATIO: PerformanceRatioPrimitive,
```

### 3. `examples/agentic_rubric/config/tool_config/verification_tools.json` — 扩展 test_runner 返回值 ✅

在 test_runner 的 `returns.properties` 中添加:
```json
"execution_time_ms": {"type": "number", "description": "Total execution time in milliseconds"},
"max_memory_mb": {"type": "number", "description": "Peak memory usage in megabytes"}
```

### 4. `verl/utils/tracking.py` — 添加 RubricSampleLogger ✅

新增 `RubricSampleLogger` dataclass，将每个 sample 的 rubric 执行细节记录为 SwanLab/WandB 表格。

**功能**：
- 从 `reward_extra_infos_dict` 提取 per-sample 数据
- 展平嵌套的 `check_scores` dict 为独立列（如 `check_correctness`, `check_time`, `check_memory`）
- 每个 training step 记录一个 table（batch_size 行）
- 表头：`step, rubric_id, <check_score_columns>, gate_passed, total_latency_ms, evidence_count, reward_score`
- 支持 SwanLab（`swanlab.echarts.Table`）和 WandB（`wandb.Table`）两种后端

### 5. `verl/trainer/ppo/ray_trainer.py` — 在训练循环中添加 per-sample 日志调用 ✅

- 导入 `RubricSampleLogger`
- 在 `__init__` 中初始化 `self.rubric_sample_logger = RubricSampleLogger()`
- 在 reward 计算后（`reward_extra_infos_dict` 更新到 batch 之后），添加条件日志调用：
```python
if reward_extra_infos_dict and "check_scores" in reward_extra_infos_dict:
    self.rubric_sample_logger.log(
        loggers=self.logger.logger,
        reward_extra_infos_dict=reward_extra_infos_dict,
        step=self.global_steps,
    )
```

### 6. `examples/agentic_rubric/data_preprocess/preprocess_ccplus.py` — 预处理脚本 ✅（新建）

将 ccplus parquet 转换为 verl 训练格式（无需 LLM 调用）。

**输入**: ccplus parquet（列: `title`, `description`, `time_limit`, `memory_limit`, `test_cases`, `time_opt`, `memory_opt`）
**输出**: verl 格式 parquet（列: `prompt`, `extra_info`, `data_source`）

#### Rubric 结构（每道题最多 3 个验证项 + 5 个证据计划）

```
verification_checklist:
  check_correctness (weight=0.0, gate=True, scoring=CODE_EXECUTION)
  check_time        (weight=1.0, gate=False, scoring=PERFORMANCE_RATIO, metric=execution_time_ms)
  check_memory      (weight=1.0, gate=False, scoring=PERFORMANCE_RATIO, metric=max_memory_mb)

evidence_plans:
  Plan 1: 运行模型代码验证正确性 (check_correctness, argument_extractor提取policy_output)
  Plan 2: 运行模型代码获取时间指标 (check_time, role=model, depends_on=[check_correctness])
  Plan 3: 运行 time_opt 参考代码获取时间基线 (check_time, role=reference, 静态code)
  Plan 4: 运行模型代码获取内存指标 (check_memory, role=model, depends_on=[check_correctness])
  Plan 5: 运行 memory_opt 参考代码获取内存基线 (check_memory, role=reference, 静态code)
```

**关键设计**:
- `check_correctness` 的 `weight=0.0` + `is_gate=True`：不参与加权聚合，但作为门控
- `depends_on=["check_correctness"]` 确保只有正确性通过后才进行性能基准测试
- Plans 2-5 在正确性通过后并行执行
- 若 `time_opt` 或 `memory_opt` 为空：跳过对应的效率检查
- `argument_extractor` 使用 Jinja2 `tojson` 过滤器确保代码字符串正确 JSON 转义
- 参考代码作为静态参数存入 `tool_arguments.code`（不走 argument_extractor）
- 测试用例字段名映射: ccplus `output` → test_runner `expected_output`
- Rubric ID: `rubric_ccplus_{sha256(title)[:12]}`
- 使用 `pyarrow.parquet.ParquetFile` 流式读取（大文件避免 OOM）
- 输出验证: `ExecutableRubric.model_validate()` round-trip

**Prompt 格式**:
```python
[
    {"role": "system", "content": "You are a competitive programmer. Write Python code..."},
    {"role": "user", "content": f"## {title}\n\n{description}\n\nTime Limit: {time_limit}ms\nMemory Limit: {memory_limit}MB"}
]
```

**参数**:
- `--input_path`: ccplus parquet 路径
- `--output_dir`: 输出目录（默认 `examples/agentic_rubric/data/ccplus/`）
- `--latency_factor`: 延迟估算因子（默认 0.3）
- `--train_ratio`: 训练集比例（默认 0.9）
- `--seed`: 随机种子（默认 42）

### 7. `examples/agentic_rubric/config/ccplus_rl_grpo.yaml` — 训练配置 ✅（新建）

基于 `rubric_rl_grpo.yaml`，关键修改:

| 配置项 | 新值 |
|--------|------|
| `data.train_files` | `examples/agentic_rubric/data/ccplus/train.parquet` |
| `data.max_prompt_length` | 4096（竞赛题目描述较长） |
| `trainer.output_dir` | `outputs/rubric_rl_grpo_ccplus` |
| `trainer.logger` | `["console", "swanlab"]` |
| `logging.swanlab.enabled` | `true` |
| `logging.swanlab.name` | `rubric_grpo_ccplus` |

### 8. `tests/rubric/test_preprocess_ccplus.py` — 单元测试 ✅（新建）

使用 `importlib.util` 加载（无需 ray），共 17 个测试：

| 测试名 | 验证内容 |
|--------|---------|
| `test_build_prompt` | chat 消息格式、title/limits 出现在 user content |
| `test_build_rubric_with_both_refs` | 有 time_opt 和 memory_opt → 3 个验证项、5 个证据计划 |
| `test_build_rubric_missing_time_opt` | 缺少 time_opt → 只有 correctness + memory (2项, 3计划) |
| `test_build_rubric_missing_both_opts` | 缺少两个 → 只有 correctness (1项, 1计划) |
| `test_performance_ratio_model_faster` | 模型更快 → score=1.0 (capped) |
| `test_performance_ratio_model_slower` | 模型慢2x → score=0.5 |
| `test_performance_ratio_equal` | 相等 → score=1.0 |
| `test_performance_ratio_missing_evidence` | 缺失证据 → score=0.0 |
| `test_performance_ratio_higher_is_better` | higher_is_better 方向正确 |
| `test_performance_ratio_memory` | 内存 ratio 计算正确 |
| `test_get_scoring_primitive_performance_ratio` | 注册表可查到新原语 |
| `test_rubric_roundtrip` | `model_dump()` + `model_validate()` 保持一致 |
| `test_field_name_mapping` | `output` → `expected_output` 映射 |
| `test_field_name_mapping_already_mapped` | 已有 `expected_output` 时直接使用 |
| `test_evidence_plan_dependencies` | 效率检查依赖正确性检查 |
| `test_evidence_plan_roles` | model/reference 角色分配、静态code vs 动态extractor |
| `test_rubric_id_deterministic` | 同title→同ID，不同title→不同ID |

---

## 奖励结构

```
# 三个独立信号，correctness 作为 gate

check_correctness (weight=0.0, gate=True):
  = passed_tests / total_tests  (partial credit via code_execution primitive)
  如果 < gate_threshold (0.5) → 所有信号归零

check_time (weight=1.0):
  = min(1.0, ref_execution_time / model_execution_time)
  参考: time_opt 列的代码

check_memory (weight=1.0):
  = min(1.0, ref_max_memory / model_max_memory)
  参考: memory_opt 列的代码

# 聚合 (weighted_sum, correctness weight=0 不参与)
# final_reward = (1.0 × time_score + 1.0 × memory_score) / 2.0
# 但如果 gate 失败: final_reward = 0.0

# reward_extra_info 中包含各项独立分数:
# {"check_correctness": 0.8, "check_time": 0.65, "check_memory": 0.9, ...}
```

- 模型比参考快/省内存 → 对应分项得满分 1.0
- 模型慢 2x → 分项得 0.5
- 模型代码不通过测试 → 所有分项归 0

## 使用方法

```bash
# Step 1: 预处理数据
python examples/agentic_rubric/data_preprocess/preprocess_ccplus.py \
    --input_path "C:\Users\14183\.cache\modelscope\hub\datasets\kurt123___ccplus-1x-optimized" \
    --output_dir examples/agentic_rubric/data/ccplus/ \
    --train_ratio 0.9

# Step 2: 训练
python -m verl.trainer.main_ppo \
    --config examples/agentic_rubric/config/ccplus_rl_grpo.yaml
```

## 测试验证

```bash
# 新测试 (17/17 通过)
python tests/rubric/test_preprocess_ccplus.py

# 原有测试 (31/31 通过，无回归)
python tests/rubric/run_standalone_tests.py
```

## 文件改动总览

| 文件 | 操作 | 说明 |
|------|------|------|
| `verl/rubric/schemas.py` | 修改 | +1 行枚举值 |
| `verl/rubric/executor/primitives.py` | 修改 | +60 行新 primitive 类 + 注册 |
| `examples/agentic_rubric/config/tool_config/verification_tools.json` | 修改 | +6 行返回字段 |
| `verl/utils/tracking.py` | 修改 | +80 行 RubricSampleLogger |
| `verl/trainer/ppo/ray_trainer.py` | 修改 | +8 行导入+初始化+日志调用 |
| `examples/agentic_rubric/data_preprocess/preprocess_ccplus.py` | 新建 | ~330 行预处理脚本 |
| `examples/agentic_rubric/config/ccplus_rl_grpo.yaml` | 新建 | ~95 行训练配置 |
| `tests/rubric/test_preprocess_ccplus.py` | 新建 | ~340 行, 17 个测试 |
