# Tag结构严格验证 - 更新摘要

## 更新内容

在GPT质量验证之前，新增**严格的tag结构验证**，确保response格式完全符合要求。

## 核心变更

### 1. 新增验证函数

**GPTDataGenerator.validate_tag_structure()** (第821-963行):

验证规则：
- ✅ `<look>` / `<think>` 必须严格交替
- ✅ 可从 `<look>` 或 `<think>` 开始
- ✅ 以且仅以一个 `<answer>` 结尾
- ✅ 不允许tag外的非空文本
- ✅ tag内容不能为空

### 2. 验证流程修改

**两阶段验证**:

```
生成response
    ↓
【新增】阶段1: Tag结构验证 (validate_tag_structure)
    ├─ 失败 → 直接失败，不调用GPT API ⚡
    └─ 通过 → 继续
         ↓
阶段2: GPT质量验证 (validate_example)
    ├─ 失败 → 标记失败
    └─ 通过 → 保存
```

### 3. 修改的方法

**validate_example()** (第965-1110行):
```python
# 首先进行tag结构验证
structure_valid, structure_result = self.validate_tag_structure(response)

if not structure_valid:
    # 直接返回失败，不调用GPT API
    return False, {
        'overall_score': 0,
        'pass': False,
        'issues': [structure_result.get('message')],
        'validation_method': 'strict_tag_structure',
        'structure_error': structure_result.get('error'),
        ...
    }

# 结构通过后，才进行GPT质量验证
...
```

### 4. 新增统计

**pipeline_report.json**:
```json
{
  "pipeline_stats": {
    "valid_examples": 80,
    "failed_validation": 20,
    "tag_structure_failures": 12,      // 新增
    "quality_validation_failures": 8,   // 新增
    ...
  }
}
```

## 错误类型

| 错误代码 | 说明 | 示例 |
|---------|------|------|
| `no_valid_tags` | 没有有效tag | 纯文本 |
| `text_before_tags` | tag前有文字 | `intro <look>...` |
| `text_between_tags` | tag间有文字 | `<look>...</look> text <think>...` |
| `text_after_tags` | tag后有文字 | `...<answer>...</answer> text` |
| `no_final_answer` | 未以answer结尾 | `<look>...<think>...` |
| `multiple_answers` | 多个answer | `<answer>A1</answer><answer>A2</answer>` |
| `not_alternating` | 未严格交替 | `<look>L1</look><look>L2</look>` |
| `empty_tag_content` | tag内容为空 | `<look></look>` |

## 合法格式示例

```
✅ <look>观察</look><think>思考</think><answer>答案</answer>

✅ <think>思考</think><look>观察</look><answer>答案</answer>

✅ <look>L1</look><think>T1</think><look>L2</look><think>T2</think><answer>A</answer>
```

## 非法格式示例

```
❌ <look>L1</look><look>L2</look><answer>A</answer>
   错误: not_alternating

❌ 前言 <look>L</look><think>T</think><answer>A</answer>
   错误: text_before_tags

❌ <look>L</look><think>T</think>
   错误: no_final_answer
```

## 优势

### 1. 节省成本 💰
- 结构不合格的样本不调用validation API
- 典型场景：20%结构失败 → 节省约7%总成本

### 2. 快速反馈 ⚡
- 本地验证，无需等待API
- 立即知道格式问题

### 3. 明确诊断 🔍
- 精确的错误类型和位置
- 便于调试和优化prompt

### 4. 质量保证 ✅
- 确保所有通过的样本格式正确
- 为后续模型训练提供clean data

## 使用方法

### 自动启用（无需配置）

```bash
python3 data_construction_gpt_pipeline.py \
  --source /data_ali/shunian/data/iceberg/scripts/data_clean.json \
  --output ./output \
  --sample 100
```

Pipeline会自动：
1. 生成response
2. 验证tag结构
3. 结构通过后再验证质量
4. 分别统计两类失败

### 查看结构失败

```python
import json

with open('output/failed_examples.jsonl', 'r') as f:
    failed = [json.loads(line) for line in f]

# 筛选tag结构失败
structure_failures = [
    ex for ex in failed
    if ex.get('validation_result', {}).get('validation_method') == 'strict_tag_structure'
]

print(f"Structure failures: {len(structure_failures)}")

# 错误分布
from collections import Counter
errors = Counter([
    ex['validation_result']['structure_error']
    for ex in structure_failures
])
for error, count in errors.most_common():
    print(f"  {error}: {count}")
```

## 测试验证

### 运行测试

```bash
python3 test_tag_validation.py
```

### 测试结果

```
================================================================================
Summary: 17 passed, 0 failed out of 17 tests
================================================================================
```

测试覆盖：
- ✅ 4种合法格式
- ✅ 13种非法格式
- ✅ 所有错误类型

## Prompt优化建议

在generation prompt中强调格式：

```python
CRITICAL FORMAT REQUIREMENTS:
1. Response MUST consist of alternating <look> and <think> tags
2. Can start with either <look> or <think>
3. MUST end with exactly one <answer> tag
4. DO NOT write any text outside the tags
5. Each tag must have non-empty content
```

## 调试流程

### 1. 检查失败率

```bash
python3 analyze_failures.py ./output
```

### 2. 查看结构错误

```bash
cat output/failed_examples.jsonl | python3 -c "
import json, sys
from collections import Counter
errors = [json.loads(l)['validation_result'].get('structure_error', 'other')
          for l in sys.stdin
          if json.loads(l).get('validation_result', {}).get('validation_method') == 'strict_tag_structure']
for err, count in Counter(errors).most_common():
    print(f'{err}: {count}')
"
```

### 3. 针对性优化

根据最常见的错误类型调整prompt：
- `not_alternating` → 强调交替模式
- `text_between_tags` → 强调不要在tag外写文字
- `no_final_answer` → 强调必须以answer结尾

## 成本分析

### 示例场景

生成100个样本，20个结构失败：

**之前**:
- Generation: 100 × $0.002 = $0.20
- Validation: 100 × $0.001 = $0.10
- **总计**: $0.30

**现在**:
- Generation: 100 × $0.002 = $0.20
- Tag validation: 本地，无成本 ✅
- Validation: 80 × $0.001 = $0.08
- **总计**: $0.28

**节省**: $0.02 (6.7%)

### 更高失败率场景

如果结构失败率达到50%：
- **节省**: $0.05 (16.7%)

## 向后兼容

✅ **完全兼容**
- 所有现有脚本无需修改
- 自动启用，无需配置
- 不影响通过样本的数据格式

## 文件清单

### 新增文件
- `test_tag_validation.py` - 测试脚本 (17个测试用例)
- `TAG_STRUCTURE_VALIDATION.md` - 详细文档
- `TAG_VALIDATION_UPDATE.md` - 本文档（更新摘要）

### 修改文件
- `data_construction_gpt_pipeline.py`:
  - 新增 `validate_tag_structure()` 方法
  - 修改 `validate_example()` 方法
  - 新增统计字段

## 下一步

建议在大规模运行前：

1. **小规模测试** (50个样本)
   ```bash
   python3 data_construction_gpt_pipeline.py \
     --source data.json --output ./test --sample 50
   ```

2. **分析结果**
   ```bash
   python3 analyze_failures.py ./test
   ```

3. **优化prompt** (根据结构失败情况)

4. **重新测试** (验证改进效果)

5. **大规模运行**

## 相关文档

- [详细文档](TAG_STRUCTURE_VALIDATION.md) - 完整说明和示例
- [失败样本分析](FAILED_EXAMPLES_FEATURE.md) - 失败样本功能
- [主文档](README_MAIN.md) - Pipeline总体说明

---

**更新日期**: 2025-11-03
**版本**: 1.0
**测试状态**: ✅ 17/17 通过
