# 失败样本保存功能 - 更新摘要

## 更新内容

Pipeline现在会**自动保存验证失败的样本**到单独的文件，方便质量分析和对比。

## 核心变更

### 1. 新增输出文件

```
output_dir/
├── generated_examples.jsonl      # 验证通过的样本（和之前一样）
├── failed_examples.jsonl         # 验证失败的样本（新增）
├── pipeline_report.json          # 统计报告
└── checkpoint.json               # 检查点
```

### 2. 代码修改

**data_construction_gpt_pipeline.py**:

1. **新增文件路径**（第1259行）:
```python
self.failed_examples_file = self.output_dir / 'failed_examples.jsonl'
```

2. **修改返回类型** - `generate_full_example()`:
```python
# 之前: 返回 Optional[GeneratedExample]
# 现在: 返回 Tuple[Optional[GeneratedExample], Dict, bool]
return example, failed_data, is_valid
```

3. **新增保存方法**（第1302-1306行）:
```python
def append_failed_examples(self, examples: List[Dict]):
    """Append failed examples to separate file for analysis"""
    with open(self.failed_examples_file, 'a', encoding='utf-8') as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
```

4. **修改处理流程**:
   - `process_item()`: 现在返回 `(valid_examples, failed_examples)`
   - `process_batch()`: 分别收集valid和failed
   - `run_async()`: 同时保存两类样本

### 3. 新增工具

**analyze_failures.py** - 失败样本分析脚本:
- 统计失败原因
- 对比成功vs失败特征
- 显示样本对比
- 生成分析报告

**test_failed_examples.sh** - 测试脚本

## 失败样本格式

```json
{
  "id": "gpt_abc12345_visu_1699123456",
  "image": "path/to/image.jpg",
  "wiki_title": "Example Subject",
  "question": "What architectural style is shown?",
  "response": "<look>...</look><think>...</think><answer>...</answer>",
  "is_multiple_choice": false,
  "validation_result": {
    "overall_score": 5.8,
    "pass": false,
    "issues": ["Repetitive language", "Insufficient depth"],
    ...
  },
  "failure_reason": ["Repetitive language", "Insufficient depth"],
  "overall_score": 5.8,
  "timestamp": "2025-11-03T21:00:00.000000"
}
```

## 使用示例

### 基本使用（自动保存失败样本）

```bash
python3 data_construction_gpt_pipeline.py \
  --source /data_ali/shunian/data/iceberg/scripts/data_clean.json \
  --output ./output \
  --sample 100
```

运行后自动生成：
- `output/generated_examples.jsonl` - 成功样本
- `output/failed_examples.jsonl` - 失败样本（如果有）

### 分析失败样本

```bash
python3 analyze_failures.py ./output
```

输出示例：
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
Most common validation issues:
  - Repetitive language in cycles: 15
  - Insufficient depth in reasoning: 12
  - Missing visual details: 8
  ...
```

### 测试功能

```bash
chmod +x test_failed_examples.sh
./test_failed_examples.sh
```

## 手动查看

### Python方式

```python
import json

# 读取失败样本
with open('output/failed_examples.jsonl', 'r') as f:
    failed = [json.loads(line) for line in f]

print(f"Total failed: {len(failed)}")

# 查看一个样本
sample = failed[0]
print(f"Question: {sample['question']}")
print(f"Score: {sample['overall_score']}")
print(f"Issues: {sample['failure_reason']}")
```

### 命令行方式

```bash
# 统计失败数量
wc -l output/failed_examples.jsonl

# 查看第一个失败样本
head -n 1 output/failed_examples.jsonl | python3 -m json.tool

# 提取失败原因
cat output/failed_examples.jsonl | \
  python3 -c "import json, sys; \
  [print(json.loads(l)['failure_reason']) for l in sys.stdin]"
```

## 使用场景

### 1. 质量分析
识别最常见的质量问题，有针对性地改进prompt

### 2. 迭代优化
对比不同参数设置下的失败率和失败原因

### 3. 人工审核
导出失败样本供人工评审，确定是否为误判

### 4. 模型调优
分析哪些策略、题型更容易失败，调整生成策略

## 统计信息

pipeline_report.json中新增统计：

```json
{
  "pipeline_stats": {
    "valid_examples": 120,
    "failed_validation": 30,
    "generation_errors": 0,
    ...
  }
}
```

## 向后兼容

✅ **完全向后兼容**
- 所有现有脚本无需修改
- 如果没有失败样本，不会创建failed_examples.jsonl
- 成功样本的格式和存储位置不变

## 存储影响

- **额外存储**: 失败样本通常占总样本的10-20%
- **文件大小**: 与成功样本相似，每个约1-2KB
- **示例**: 1000个样本，200个失败 → 约400KB额外存储

## 最佳实践

### 1. 定期分析
每次大批量生成后运行分析：
```bash
python3 analyze_failures.py ./output
```

### 2. 持续改进
根据分析结果调整参数：
- 高repetition → 增加temperature
- 低depth → 使用更复杂的source items
- 格式问题 → 优化prompt

### 3. 保留失败样本
建议保留失败样本用于：
- A/B测试对比
- 模型调优
- 质量趋势分析

### 4. 人工审核抽查
定期抽查失败样本，确认validation的准确性

## 文件清单

### 新增文件
- `analyze_failures.py` - 失败样本分析工具
- `test_failed_examples.sh` - 测试脚本
- `FAILED_EXAMPLES_FEATURE.md` - 详细文档
- `FAILED_EXAMPLES_UPDATE.md` - 本文档（更新摘要）

### 修改文件
- `data_construction_gpt_pipeline.py` - 核心pipeline

## 常见问题

### Q: 如果所有样本都通过validation会怎样？
A: 不会创建failed_examples.jsonl文件，这是正常现象。

### Q: 失败样本会计入checkpoint吗？
A: 会。失败样本也是"已处理"，checkpoint记录的是处理的items数量。

### Q: 可以禁用失败样本保存吗？
A: 可以注释掉`run_async()`中的`append_failed_examples()`调用。

### Q: 失败样本占用太多空间怎么办？
A: 可以定期清理或只保留最近的失败样本用于分析。

## 相关文档

- [详细功能文档](FAILED_EXAMPLES_FEATURE.md)
- [选择题功能](MULTIPLE_CHOICE_README.md)
- [主README](README_MAIN.md)

---

**更新日期**: 2025-11-03
**版本**: 1.0
**作者**: Data ML Architect
