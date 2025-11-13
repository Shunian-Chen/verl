# Pipeline新功能汇总

本文档汇总了今天添加的所有新功能。

## 功能列表

### 1. ✅ 选择题支持

**文档**: [MULTIPLE_CHOICE_README.md](MULTIPLE_CHOICE_README.md)

**功能**:
- 生成4选项的选择题（A/B/C/D）
- 可配置选择题占比（默认20%）
- 保持完整的look-think-answer推理过程
- 自动解析和验证选项

**使用**:
```bash
python3 data_construction_gpt_pipeline.py \
  --source data.json \
  --output ./output \
  --multiple-choice-ratio 0.2  # 20%选择题
```

**特点**:
- 干扰项合理，测试真实理解
- 推理过程分析各选项
- 明确指出正确答案及原因

---

### 2. ✅ 失败样本保存

**文档**: [FAILED_EXAMPLES_FEATURE.md](FAILED_EXAMPLES_FEATURE.md)

**功能**:
- 自动保存验证失败的样本到单独文件
- 包含完整的失败原因和validation结果
- 便于质量分析和对比

**输出**:
```
output/
├── generated_examples.jsonl      # 成功样本
├── failed_examples.jsonl         # 失败样本 ⭐新增
└── pipeline_report.json
```

**分析工具**:
```bash
python3 analyze_failures.py ./output
```

**用途**:
- 识别常见质量问题
- 对比成功vs失败特征
- 迭代优化prompt和参数

---

### 3. ✅ Tag结构严格验证

**文档**: [TAG_STRUCTURE_VALIDATION.md](TAG_STRUCTURE_VALIDATION.md)

**功能**:
- 在GPT validation前先进行本地tag结构验证
- 严格检查look/think交替、answer结尾、无杂文等
- 结构不合格直接失败，不调用GPT API

**验证规则**:
1. ✅ `<look>` / `<think>` 严格交替
2. ✅ 可从任一标签开始
3. ✅ 以且仅以一个 `<answer>` 结尾
4. ✅ 不允许tag外的非空文本
5. ✅ 所有tag内容非空

**优势**:
- 💰 节省成本（减少validation API调用）
- ⚡ 快速反馈（本地验证）
- 🔍 明确诊断（精确错误类型）

**测试**:
```bash
python3 test_tag_validation.py
# 输出: 17/17 tests passed ✓
```

---

## 综合使用示例

### 完整命令

```bash
python3 data_construction_gpt_pipeline.py \
  --source /data_ali/shunian/data/iceberg/scripts/data_clean.json \
  --output ./output \
  --sample 100 \
  --examples-per-item 3 \
  --sampling-strategy cluster \
  --multiple-choice-ratio 0.2 \
  --max-concurrent 10 \
  --seed 42
```

### 工作流程

```
1. 数据采样 (cluster采样，最大化category覆盖)
   ↓
2. 生成questions和responses
   - 80%开放式问题
   - 20%选择题
   ↓
3. Tag结构验证 ⭐新增
   - 检查格式是否严格符合要求
   - 不合格→保存到failed_examples.jsonl
   - 合格→继续
   ↓
4. GPT质量验证
   - 5维度评分
   - 不合格→保存到failed_examples.jsonl ⭐新增
   - 合格→保存到generated_examples.jsonl
   ↓
5. 生成报告
   - pipeline_report.json
   - 包含详细统计
```

### 结果分析

```bash
# 1. 查看总体统计
cat output/pipeline_report.json | python3 -m json.tool

# 2. 分析失败样本
python3 analyze_failures.py ./output

# 3. 查看问题类型分布
cat output/generated_examples.jsonl | python3 -c "
import json, sys
mc = sum(1 for l in sys.stdin if json.loads(l)['is_multiple_choice'])
total = sum(1 for l in open('output/generated_examples.jsonl'))
print(f'Multiple choice: {mc}/{total} ({mc/total*100:.1f}%)')
"
```

## 统计信息

### pipeline_report.json

```json
{
  "pipeline_stats": {
    "valid_examples": 80,
    "failed_validation": 20,

    // 问题类型统计 (选择题功能)
    "multiple_choice_questions": 16,
    "open_ended_questions": 64,

    // 失败类型统计 (tag验证功能)
    "tag_structure_failures": 12,
    "quality_validation_failures": 8,

    "generation_errors": 0,
    "total_items_processed": 100,
    "total_examples_generated": 80
  },
  "api_usage": {
    "total_requests": 180,
    "generation_requests": 100,
    "validation_requests": 80,  // 注意：节省了20次调用
    "total_cost_usd": 0.28
  }
}
```

## 成本影响

### 示例场景：100个样本

**配置**:
- examples_per_item: 3
- multiple_choice_ratio: 0.2
- 结构失败率: 20%

**API调用**:
- Question generation: 100 calls
- Response generation: 100 calls
- Tag validation: 本地，0 calls ✅
- Quality validation: 80 calls (节省20次) ✅

**成本**:
- 之前: $0.30
- 现在: $0.28 (节省6.7%)

## 测试验证

### 1. Tag结构验证测试
```bash
python3 test_tag_validation.py
# 预期: 17/17 tests passed
```

### 2. 选择题功能测试
```bash
chmod +x test_multiple_choice.sh
./test_multiple_choice.sh
# 预期: 生成样本中约20%为选择题
```

### 3. 失败样本保存测试
```bash
chmod +x test_failed_examples.sh
./test_failed_examples.sh
# 预期: 同时生成generated和failed文件
```

## 文档索引

### 详细文档

1. **选择题功能**:
   - [MULTIPLE_CHOICE_README.md](MULTIPLE_CHOICE_README.md) - 完整使用指南
   - [MULTIPLE_CHOICE_UPDATE.md](MULTIPLE_CHOICE_UPDATE.md) - 更新摘要

2. **失败样本保存**:
   - [FAILED_EXAMPLES_FEATURE.md](FAILED_EXAMPLES_FEATURE.md) - 功能说明
   - [FAILED_EXAMPLES_UPDATE.md](FAILED_EXAMPLES_UPDATE.md) - 更新摘要

3. **Tag结构验证**:
   - [TAG_STRUCTURE_VALIDATION.md](TAG_STRUCTURE_VALIDATION.md) - 验证规则
   - [TAG_VALIDATION_UPDATE.md](TAG_VALIDATION_UPDATE.md) - 更新摘要

### 其他文档

- [README_MAIN.md](README_MAIN.md) - Pipeline主文档
- [CATEGORY_SAMPLING_README.md](CATEGORY_SAMPLING_README.md) - 分类采样
- [VALIDATION_FIX_NOTES.md](VALIDATION_FIX_NOTES.md) - Validation修复

## 测试脚本

| 脚本 | 功能 | 预期结果 |
|------|------|----------|
| `test_tag_validation.py` | 测试tag结构验证 | 17/17 通过 |
| `test_multiple_choice.sh` | 测试选择题生成 | 约20%选择题 |
| `test_failed_examples.sh` | 测试失败样本保存 | 生成两个文件 |
| `analyze_failures.py` | 分析失败样本 | 显示详细统计 |

## 最佳实践

### 1. 开发迭代

```bash
# Step 1: 小规模测试
python3 data_construction_gpt_pipeline.py \
  --source data.json --output ./pilot --sample 50

# Step 2: 分析结果
python3 analyze_failures.py ./pilot

# Step 3: 查看失败原因
cat ./pilot/failed_examples.jsonl | head -5 | python3 -m json.tool

# Step 4: 优化prompt/参数

# Step 5: 重新测试
python3 data_construction_gpt_pipeline.py \
  --source data.json --output ./run2 --sample 50

# Step 6: 对比改进
python3 -c "
import json
p1 = json.load(open('./pilot/pipeline_report.json'))
p2 = json.load(open('./run2/pipeline_report.json'))
print(f'Pilot: {p1['pipeline_stats']['valid_examples']} valid')
print(f'Run2:  {p2['pipeline_stats']['valid_examples']} valid')
"
```

### 2. 生产运行

```bash
# 大规模运行
python3 data_construction_gpt_pipeline.py \
  --source data.json \
  --output ./production \
  --sample 10000 \
  --examples-per-item 3 \
  --sampling-strategy cluster \
  --multiple-choice-ratio 0.2 \
  --max-concurrent 20 \
  --checkpoint-interval 500

# 定期检查
watch -n 60 'wc -l ./production/*.jsonl'

# 完成后分析
python3 analyze_failures.py ./production
```

### 3. 质量监控

```python
import json
from pathlib import Path

def monitor_quality(output_dir):
    """监控数据质量"""
    output_path = Path(output_dir)

    # 读取报告
    with open(output_path / 'pipeline_report.json') as f:
        report = json.load(f)

    stats = report['pipeline_stats']

    # 计算指标
    total = stats['valid_examples'] + stats['failed_validation']
    success_rate = stats['valid_examples'] / total * 100
    structure_fail_rate = stats.get('tag_structure_failures', 0) / total * 100

    print(f"Success rate: {success_rate:.1f}%")
    print(f"Structure failure rate: {structure_fail_rate:.1f}%")

    # 警告
    if structure_fail_rate > 30:
        print("⚠️  High structure failure rate - check prompts!")
    if success_rate < 70:
        print("⚠️  Low success rate - review generation parameters!")

monitor_quality('./output')
```

## 性能优化建议

### 1. 并发控制

```bash
# 低配置机器
--max-concurrent 5

# 中等配置
--max-concurrent 10

# 高配置
--max-concurrent 20
```

### 2. 选择题比例

根据使用场景调整：

| 场景 | 推荐比例 | 原因 |
|------|---------|------|
| 冷启动训练 | 20% | 平衡多样性 |
| 评估benchmark | 50-100% | 标准化评估 |
| 深度推理训练 | 10% | 侧重开放式 |

### 3. 采样策略

```bash
# 最大category覆盖（推荐）
--sampling-strategy cluster

# 类别平衡
--sampling-strategy balanced

# 随机采样
--sampling-strategy random
```

## 故障排除

### 问题1: Tag结构失败率高（>30%）

**诊断**:
```bash
python3 analyze_failures.py ./output
# 查看Tag Structure Errors部分
```

**解决**: 优化prompt，强调格式要求

### 问题2: 没有生成failed_examples.jsonl

**原因**: 所有样本都通过了validation

**验证**: 查看pipeline_report.json中的failed_validation应为0

### 问题3: 选择题比例不准确

**原因**: 随机性导致波动

**解决**: 增加样本量，实际比例会接近目标值

## 向后兼容性

✅ 所有新功能都是**完全向后兼容**的：

- 不使用`--multiple-choice-ratio`时，默认为0.2（20%）
- failed_examples.jsonl只在有失败时才创建
- 所有现有脚本无需修改
- 成功样本的格式和位置不变

## 下一步

### 短期
- [x] 选择题支持
- [x] 失败样本保存
- [x] Tag结构验证
- [ ] 人工审核工具
- [ ] 批量validation优化

### 长期
- [ ] 多选题支持
- [ ] 难度分级
- [ ] 自动prompt优化
- [ ] 质量趋势分析

## 联系与反馈

如有问题或建议，请查看相关文档或创建issue。

---

**更新日期**: 2025-11-03
**版本**: 2.0
**包含功能**: 选择题 + 失败样本保存 + Tag结构验证
