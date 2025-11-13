# 选择题功能说明

## 概述

数据构造Pipeline现在支持生成**选择题**（Multiple Choice Questions），与开放式问题混合，用于训练视觉-语言模型的多样化推理能力。

## 功能特性

### 1. 题目类型

- **开放式问题（Open-ended Questions）**：需要详细解释和推理的问题
- **选择题（Multiple Choice Questions）**：
  - 4个选项 (A/B/C/D)
  - 1个正确答案
  - 3个合理的干扰项

### 2. 推理模式

**所有题目类型都保持完整的look-think-answer推理过程**：

#### 开放式问题示例：
```
<look>
[观察图像中的元素...]
</look>

<think>
[分析和推理...]
</think>

<answer>
[综合性回答...]
</answer>
```

#### 选择题示例：
```
Question: Which architectural style is primarily exhibited in this building?

A. Gothic Revival
B. Art Deco
C. Neoclassical
D. Brutalist

<look>
[仔细观察建筑的关键特征...]
</look>

<think>
[逐一分析各选项：
- A选项：Gothic Revival通常有...，但这个建筑...
- B选项：Art Deco的特点是...，这与图中...不符
- C选项：Neoclassical风格表现为...，这正是我们看到的...
- D选项：Brutalist风格以...著称，但图中...]
</think>

<answer>
The correct answer is C (Neoclassical). [详细解释为什么这是正确答案，以及为什么其他选项不正确]
</answer>
```

## 使用方法

### 基本用法

```bash
python3 data_construction_gpt_pipeline.py \
  --source /path/to/data.json \
  --output ./output_dir \
  --multiple-choice-ratio 0.2 \
  --examples-per-item 5 \
  --sample 100
```

### 参数说明

- `--multiple-choice-ratio`: 选择题占比（默认0.2 = 20%）
  - 0.0 = 全部开放式问题
  - 0.2 = 20%选择题，80%开放式
  - 0.5 = 各占一半
  - 1.0 = 全部选择题

### 完整示例

```bash
# 生成1000个样本，其中20%为选择题
python3 data_construction_gpt_pipeline.py \
  --source /data_ali/shunian/data/iceberg/scripts/data_clean.json \
  --output ./output_mc_20pct \
  --sample 1000 \
  --examples-per-item 3 \
  --sampling-strategy cluster \
  --multiple-choice-ratio 0.2 \
  --max-concurrent 10 \
  --seed 42

# 生成500个样本，其中50%为选择题
python3 data_construction_gpt_pipeline.py \
  --source /data_ali/shunian/data/iceberg/scripts/data_clean.json \
  --output ./output_mc_50pct \
  --sample 500 \
  --examples-per-item 4 \
  --multiple-choice-ratio 0.5

# 只生成选择题（100%）
python3 data_construction_gpt_pipeline.py \
  --source /data_ali/shunian/data/iceberg/scripts/data_clean.json \
  --output ./output_mc_only \
  --sample 200 \
  --examples-per-item 2 \
  --multiple-choice-ratio 1.0
```

## 测试

运行测试脚本验证功能：

```bash
chmod +x test_multiple_choice.sh
./test_multiple_choice.sh
```

测试脚本会：
1. 生成5个items × 3 examples = 15个训练样本
2. 显示选择题和开放式问题的分布
3. 展示一个选择题的完整示例

## 输出格式

### JSONL格式

每个example包含以下字段：

```json
{
  "id": "gpt_abc12345_visu_mc_1699123456",
  "image": "path/to/image.jpg",
  "wiki_title": "Example Subject",
  "categories": ["Category1", "Category2"],
  "question": "Which feature is most prominent in this image?",
  "question_strategy": "visual_perception",
  "complexity": "medium",
  "response": "<look>...</look><think>...</think><answer>...</answer>",
  "num_cycles": 2,
  "word_count": 456,
  "is_multiple_choice": true,
  "options": [
    "A. First option",
    "B. Second option",
    "C. Third option",
    "D. Fourth option"
  ],
  "correct_answer": "C",
  "gpt_generation_metadata": {...},
  "validation_metadata": {...}
}
```

### 关键字段说明

- `is_multiple_choice`: 布尔值，标识是否为选择题
- `options`: 选项列表（仅选择题）
- `correct_answer`: 正确答案字母（仅选择题）
- `response`: 完整的look-think-answer推理过程

## 统计报告

Pipeline完成后，`pipeline_report.json`会包含选择题统计：

```json
{
  "pipeline_stats": {
    "valid_examples": 150,
    "multiple_choice_questions": 30,
    "open_ended_questions": 120,
    "failed_validation": 5,
    ...
  },
  ...
}
```

## 设计理念

### 为什么需要选择题？

1. **多样化训练**：结合选择题和开放式问题，训练模型处理不同类型的任务
2. **评估便利**：选择题提供标准化的评估方式（正确/错误）
3. **实际应用**：许多真实场景需要从多个选项中做出选择
4. **推理训练**：通过分析多个选项，训练模型的比较和排除能力

### 为什么保持look-think推理？

即使是选择题，我们也保持完整的推理过程：

1. **系统性思考**：培养模型逐步分析的能力
2. **可解释性**：清晰展示为什么选择某个答案
3. **深度学习**：不仅知道答案，还理解原因
4. **一致性**：与开放式问题保持统一的推理模式

## 质量控制

### 选项生成质量

GPT会生成：
- **1个正确选项**：基于图像描述完全准确
- **3个干扰项**：
  - 表面看起来合理
  - 测试常见误解
  - 需要仔细观察才能排除

### Validation

选择题使用与开放式问题相同的validation逻辑：
- 检查必需的tags（look, think, answer）
- 评估推理质量
- 确保最终答案正确指向正确选项

## 最佳实践

### 推荐比例

- **20% 选择题**（默认）：平衡多样性和深度
- **30-50% 选择题**：侧重标准化评估
- **10% 选择题**：主要训练开放式推理

### 使用场景

1. **冷启动训练**：20-30%选择题
2. **评估benchmark**：50-100%选择题
3. **深度推理训练**：10-20%选择题

## 成本影响

选择题生成需要额外的API调用：
- 开放式：2次调用（question + response）
- 选择题：2次调用（mc_question + mc_response）

**成本相同**，因为调用次数一致，但选择题的prompt略长（包含选项）。

预估：每个选择题比开放式问题贵约5-10%。

## 故障排除

### 问题1：选择题比例不准确

**原因**：随机性导致实际比例有波动

**解决**：
- 增加样本量以接近目标比例
- 检查`pipeline_report.json`中的实际分布

### 问题2：选项解析失败

**症状**：日志中显示"Failed to generate multiple choice question"

**解决**：
- 检查API返回是否完整
- 验证prompt格式
- 查看`gpt_pipeline.log`获取详细错误

### 问题3：所有问题都是选择题/开放式

**原因**：`--multiple-choice-ratio`参数设置错误

**解决**：
```bash
# 检查参数
--multiple-choice-ratio 0.2  # ✓ 正确：20%
--multiple-choice-ratio 20   # ✗ 错误：会被当作2000%
```

## 示例：查看生成的数据

```python
import json

# 读取生成的数据
with open('output/generated_examples.jsonl', 'r') as f:
    examples = [json.loads(line) for line in f]

# 分类统计
mc_examples = [ex for ex in examples if ex['is_multiple_choice']]
open_examples = [ex for ex in examples if not ex['is_multiple_choice']]

print(f"Total: {len(examples)}")
print(f"Multiple choice: {len(mc_examples)} ({len(mc_examples)/len(examples)*100:.1f}%)")
print(f"Open-ended: {len(open_examples)} ({len(open_examples)/len(examples)*100:.1f}%)")

# 查看一个选择题示例
if mc_examples:
    mc = mc_examples[0]
    print(f"\nSample MC Question:")
    print(f"Q: {mc['question']}")
    for opt in mc['options']:
        print(f"  {opt}")
    print(f"Correct: {mc['correct_answer']}")
    print(f"\nResponse:\n{mc['response'][:500]}...")
```

## 后续优化

可能的改进方向：

1. **选项数量可配置**：支持3选项或5选项
2. **难度分级**：easy/medium/hard选择题
3. **多选题**：支持多个正确答案
4. **批量生成选项**：一次API调用生成多组选项以节省成本

## 相关文档

- [主文档](README_MAIN.md)
- [分类采样说明](CATEGORY_SAMPLING_README.md)
- [Validation修复说明](VALIDATION_FIX_NOTES.md)

---

**更新时间**: 2025-11-03
**版本**: 1.0
