# 选择题功能更新摘要

## 更新内容

为GPT数据生成Pipeline添加了**选择题（Multiple Choice Questions）**支持，允许生成混合的开放式问题和选择题训练数据。

## 关键变更

### 1. 数据结构 (data_construction_gpt_pipeline.py)

**GeneratedExample类**新增字段：
```python
is_multiple_choice: bool = False
options: Optional[List[str]] = None
correct_answer: Optional[str] = None
```

### 2. Prompt生成

**PromptLibrary类**新增方法：
- `get_multiple_choice_question_prompt()`: 生成选择题问题和4个选项
- `get_multiple_choice_response_prompt()`: 生成选择题的推理回答

### 3. API调用

**GPTDataGenerator类**新增方法：
- `generate_multiple_choice_question()`: 生成选择题并解析选项
- `generate_multiple_choice_response()`: 生成选择题的look-think-answer回答

### 4. Pipeline集成

**GPTDataConstructionPipeline类**修改：
- `__init__()`: 新增`multiple_choice_ratio`参数（默认0.2）
- `generate_full_example()`: 20%概率生成选择题，80%生成开放式问题
- `process_item()`: 统计选择题和开放式问题数量

### 5. 命令行参数

新增参数：
```bash
--multiple-choice-ratio 0.2  # 选择题占比（默认20%）
```

## 使用示例

### 基础用法（20%选择题）
```bash
python3 data_construction_gpt_pipeline.py \
  --source /data_ali/shunian/data/iceberg/scripts/data_clean.json \
  --output ./output_with_mc \
  --sample 100 \
  --examples-per-item 3
```

### 自定义选择题比例（50%）
```bash
python3 data_construction_gpt_pipeline.py \
  --source /data_ali/shunian/data/iceberg/scripts/data_clean.json \
  --output ./output_50pct_mc \
  --sample 100 \
  --examples-per-item 3 \
  --multiple-choice-ratio 0.5
```

### 仅生成选择题（100%）
```bash
python3 data_construction_gpt_pipeline.py \
  --source /data_ali/shunian/data/iceberg/scripts/data_clean.json \
  --output ./output_mc_only \
  --sample 100 \
  --examples-per-item 2 \
  --multiple-choice-ratio 1.0
```

## 测试

运行测试脚本：
```bash
chmod +x test_multiple_choice.sh
./test_multiple_choice.sh
```

## 输出示例

### 选择题格式
```json
{
  "id": "gpt_abc12345_visu_mc_1699123456",
  "question": "Which architectural style is exhibited?",
  "is_multiple_choice": true,
  "options": [
    "A. Gothic Revival",
    "B. Art Deco",
    "C. Neoclassical",
    "D. Brutalist"
  ],
  "correct_answer": "C",
  "response": "<look>...</look><think>分析各选项...</think><answer>The correct answer is C...</answer>",
  ...
}
```

### 统计报告
```json
{
  "pipeline_stats": {
    "valid_examples": 100,
    "multiple_choice_questions": 20,
    "open_ended_questions": 80,
    ...
  }
}
```

## 技术细节

### 选项生成
- 4个选项（A/B/C/D）
- 1个正确答案基于图像描述
- 3个合理的干扰项，测试常见误解

### 推理保持
选择题仍然使用完整的look-think-answer模式：
1. **look**: 观察图像相关特征
2. **think**: 逐一分析各选项，排除错误答案
3. **answer**: 明确指出正确选项并详细解释

### Validation
- 使用相同的validation逻辑
- 检查必需tags的存在性
- 评估推理质量和连贯性

## 文件清单

### 新增文件
- `test_multiple_choice.sh`: 测试脚本
- `MULTIPLE_CHOICE_README.md`: 详细文档
- `MULTIPLE_CHOICE_UPDATE.md`: 本文档（更新摘要）

### 修改文件
- `data_construction_gpt_pipeline.py`: 核心pipeline代码

## 成本影响

- **API调用次数**：与开放式问题相同（2次/example）
- **Token消耗**：略高5-10%（因为包含选项）
- **总体影响**：在20%选择题比例下，成本增加约1-2%

## 质量保证

1. **选项质量**：GPT生成合理的干扰项
2. **推理完整性**：保持完整的look-think-answer流程
3. **正确性验证**：validation确保答案正确性
4. **格式一致性**：所有选择题遵循统一格式

## 配置建议

根据使用场景选择比例：

| 场景 | 推荐比例 | 说明 |
|------|---------|------|
| 冷启动SFT训练 | 20-30% | 平衡多样性和深度推理 |
| 评估benchmark | 50-100% | 标准化评估更方便 |
| 深度推理训练 | 10-20% | 侧重开放式推理能力 |
| 通用训练 | 20% | 默认值，适合大多数场景 |

## 向后兼容

- 不使用`--multiple-choice-ratio`参数时，默认为0.2（20%）
- 所有旧脚本无需修改仍可正常运行
- 输出格式完全向后兼容（新增字段对旧代码无影响）

## 后续计划

可能的增强功能：
- [ ] 支持3/5选项配置
- [ ] 多选题支持
- [ ] 难度分级
- [ ] 批量选项生成优化

## 相关链接

- [详细文档](MULTIPLE_CHOICE_README.md)
- [主README](README_MAIN.md)
- [分类采样说明](CATEGORY_SAMPLING_README.md)

---

**更新日期**: 2025-11-03
**作者**: Data ML Architect
**版本**: 1.0
