# 失败样本保存功能

## 概述

Pipeline现在会**同时保存成功和失败的样本**，方便您进行质量对比分析。

## 功能说明

### 输出文件

Pipeline运行后会生成两个JSONL文件：

1. **generated_examples.jsonl** - 验证通过的样本（和之前一样）
2. **failed_examples.jsonl** - 验证失败的样本（新增）

### 失败样本格式

每个失败样本包含完整信息：

```json
{
  "id": "gpt_abc12345_visu_1699123456",
  "image": "path/to/image.jpg",
  "wiki_title": "Example Subject",
  "categories": ["Category1", "Category2"],
  "question": "What architectural style is shown?",
  "question_strategy": "visual_perception",
  "response": "<look>...</look><think>...</think><answer>...</answer>",
  "is_multiple_choice": false,
  "options": null,
  "correct_answer": null,
  "validation_result": {
    "content_quality": 5.5,
    "coherence": 6.0,
    "diversity": 4.5,
    "educational_value": 5.0,
    "overall_score": 5.8,
    "pass": false,
    "issues": [
      "Repetitive language in cycles",
      "Insufficient depth in reasoning"
    ],
    "validation_method": "gpt"
  },
  "failure_reason": [
    "Repetitive language in cycles",
    "Insufficient depth in reasoning"
  ],
  "overall_score": 5.8,
  "timestamp": "2025-11-03T21:00:00.000000"
}
```

### 关键字段说明

- **validation_result**: 完整的validation评分和反馈
- **failure_reason**: 失败的具体原因（从issues提取）
- **overall_score**: 总体评分（<7.0为失败）
- **timestamp**: 生成时间

## 使用方法

### 运行Pipeline

运行方式和之前完全相同：

```bash
python3 data_construction_gpt_pipeline.py \
  --source /data_ali/shunian/data/iceberg/scripts/data_clean.json \
  --output ./output \
  --sample 100 \
  --examples-per-item 3
```

Pipeline会自动保存失败的样本到 `output/failed_examples.jsonl`。

### 分析失败样本

使用分析脚本查看失败原因：

```bash
python3 analyze_failures.py ./output
```

这会显示：
1. 总体统计（成功率、失败率）
2. 最常见的失败原因
3. 评分分布
4. 成功vs失败的对比（按策略、题型等）
5. 样本示例对比

### 示例输出

```
================================================================================
OVERALL STATISTICS
================================================================================
Total examples generated: 150
Successful: 120 (80.0%)
Failed: 30 (20.0%)

================================================================================
VALIDATION FAILURE ANALYSIS
================================================================================

Total failed examples: 30

Most common validation issues:
  - Repetitive language in cycles: 15
  - Insufficient depth in reasoning: 12
  - Missing visual details: 8
  - Weak connection to knowledge: 5

Validation score distribution:
   4: ███████ (7)
   5: ██████████████ (14)
   6: █████████ (9)

================================================================================
SUCCESS vs FAILURE COMPARISON
================================================================================

Question strategies:
  visual_perception      :  25 valid,  5 failed (83.3% success)
  knowledge_integration  :  20 valid,  8 failed (71.4% success)
  multi_hop_reasoning    :  22 valid,  7 failed (75.9% success)
  ...

Question types:
  Multiple choice: 24 valid, 6 failed
  Open-ended:      96 valid, 24 failed

Response length:
  Valid:  avg 612 words
  Failed: avg 478 words
```

## 手动查看失败样本

### 使用Python

```python
import json

# 读取失败样本
with open('output/failed_examples.jsonl', 'r') as f:
    failed = [json.loads(line) for line in f]

# 查看最常见的失败原因
from collections import Counter
all_issues = []
for ex in failed:
    all_issues.extend(ex.get('failure_reason', []))

print("Top failure reasons:")
for reason, count in Counter(all_issues).most_common(5):
    print(f"  {reason}: {count}")

# 查看一个失败样本
sample = failed[0]
print(f"\nQuestion: {sample['question']}")
print(f"Score: {sample['overall_score']}")
print(f"Issues: {sample['failure_reason']}")
print(f"\nResponse:\n{sample['response'][:300]}...")
```

### 使用命令行工具

```bash
# 统计失败数量
wc -l output/failed_examples.jsonl

# 查看第一个失败样本
head -n 1 output/failed_examples.jsonl | python3 -m json.tool

# 提取所有失败原因
cat output/failed_examples.jsonl | \
  python3 -c "import json, sys; \
  issues = []; \
  [issues.extend(json.loads(l).get('failure_reason', [])) for l in sys.stdin]; \
  from collections import Counter; \
  [print(f'{r}: {c}') for r, c in Counter(issues).most_common(10)]"
```

## 常见失败原因及改进建议

### 1. Repetitive language in cycles

**原因**: 多个look-think循环中使用了重复的语言

**改进**:
- 调整temperature参数（增加多样性）
- 优化prompt，强调"每个cycle必须有新信息"
- 增加content length要求

### 2. Insufficient depth in reasoning

**原因**: think部分的推理不够深入

**改进**:
- 要求更长的think部分
- 在prompt中提供更详细的推理示例
- 增加content字段的信息量

### 3. Missing visual details

**原因**: look部分缺少具体的视觉观察

**改进**:
- 在prompt中强调"详细描述可见元素"
- 提供更丰富的图像描述content
- 增加look部分的字数要求

### 4. Weak connection to knowledge

**原因**: think部分没有很好地连接背景知识

**改进**:
- 确保content包含足够的背景信息
- 在prompt中强调知识整合
- 使用knowledge_integration策略

### 5. Answer doesn't address question

**原因**: 最终答案偏离了问题

**改进**:
- 在validation前检查答案相关性
- 调整response generation prompt
- 增加答案与问题的对应性要求

## 质量改进流程

### 1. 运行初始批次

```bash
python3 data_construction_gpt_pipeline.py \
  --source data.json \
  --output ./pilot_run \
  --sample 50 \
  --examples-per-item 3
```

### 2. 分析失败原因

```bash
python3 analyze_failures.py ./pilot_run
```

### 3. 根据分析调整参数

如果repetitive language很多：
```bash
# 增加temperature
python3 data_construction_gpt_pipeline.py \
  --source data.json \
  --output ./run2 \
  --sample 100 \
  --temperature 0.9  # 默认0.8
```

如果depth不够：
```bash
# 增加examples per item，选择更复杂的items
python3 data_construction_gpt_pipeline.py \
  --source data.json \
  --output ./run2 \
  --sample 100 \
  --sampling-strategy cluster  # 确保内容丰富的items
```

### 4. 对比改进效果

```python
import json

# 读取两次运行的结果
def get_stats(output_dir):
    with open(f'{output_dir}/pipeline_report.json') as f:
        report = json.load(f)
    valid = report['pipeline_stats']['valid_examples']
    failed = report['pipeline_stats']['failed_validation']
    return valid, failed, valid / (valid + failed)

pilot_valid, pilot_failed, pilot_rate = get_stats('./pilot_run')
run2_valid, run2_failed, run2_rate = get_stats('./run2')

print(f"Pilot: {pilot_rate:.1%} success rate")
print(f"Run2:  {run2_rate:.1%} success rate")
print(f"Improvement: {(run2_rate - pilot_rate)*100:.1f} percentage points")
```

## 高级用法

### 筛选特定类型的失败

```python
import json

with open('output/failed_examples.jsonl', 'r') as f:
    failed = [json.loads(line) for line in f]

# 筛选选择题失败
mc_failed = [ex for ex in failed if ex['is_multiple_choice']]
print(f"Multiple choice failures: {len(mc_failed)}")

# 筛选特定策略失败
strategy_failed = [ex for ex in failed if ex['question_strategy'] == 'multi_hop_reasoning']
print(f"Multi-hop reasoning failures: {len(strategy_failed)}")

# 筛选低分失败（<5分）
low_score = [ex for ex in failed if ex['overall_score'] < 5]
print(f"Very low score (<5): {len(low_score)}")
```

### 导出失败样本供人工审核

```python
import json
from pathlib import Path

output_dir = Path('output')
with open(output_dir / 'failed_examples.jsonl', 'r') as f:
    failed = [json.loads(line) for line in f]

# 创建人工审核格式
review_dir = output_dir / 'review'
review_dir.mkdir(exist_ok=True)

for i, ex in enumerate(failed[:10]):  # 前10个
    with open(review_dir / f'failed_{i+1}.txt', 'w') as f:
        f.write(f"Question: {ex['question']}\n\n")
        f.write(f"Response:\n{ex['response']}\n\n")
        f.write(f"Score: {ex['overall_score']}\n")
        f.write(f"Issues: {', '.join(ex['failure_reason'])}\n\n")
        f.write("=" * 80 + "\n")
        f.write("Review notes:\n")
        f.write("\n\n")

print(f"Exported {min(10, len(failed))} samples to {review_dir}/")
```

## 统计信息

在pipeline_report.json中，现在包含失败统计：

```json
{
  "pipeline_stats": {
    "valid_examples": 120,
    "failed_validation": 30,
    "multiple_choice_questions": 24,
    "open_ended_questions": 96,
    "generation_errors": 0,
    ...
  }
}
```

## 注意事项

1. **存储空间**: 失败样本也会占用磁盘空间，但通常比成功样本少
2. **隐私**: 失败样本包含完整的question和response，注意数据隐私
3. **版本控制**: 建议将failed_examples.jsonl加入.gitignore
4. **持续改进**: 定期分析失败样本，持续优化prompt和参数

## 禁用失败样本保存

如果不需要保存失败样本（节省存储空间），可以修改代码：

```python
# 在 run_async 方法中注释掉这几行
# if failed_examples:
#     self.append_failed_examples(failed_examples)
#     logger.info(f"Saved {len(failed_examples)} failed examples for analysis")
```

## 相关文档

- [主文档](README_MAIN.md)
- [选择题功能](MULTIPLE_CHOICE_README.md)
- [Validation修复](VALIDATION_FIX_NOTES.md)

---

**更新时间**: 2025-11-03
**版本**: 1.0
