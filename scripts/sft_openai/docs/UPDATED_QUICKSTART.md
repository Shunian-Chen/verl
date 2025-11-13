# Updated Quick Start Guide - Category-Based Sampling

## 更新说明

数据构造管线已更新，集成了基于category的均匀采样功能，确保生成的训练数据具有更好的多样性和覆盖率。

## 环境变量设置

在 `.env` 文件中设置以下变量：

```bash
# Required
OPENAI_API_KEY=your-api-key-here
OPENAI_BASE_URL=https://api.openai.com/v1

# Optional - Models (defaults shown)
GENERATION_MODEL=gpt-4o-mini
VALIDATION_MODEL=gpt-4o-mini
```

或者直接在命令行中导出：

```bash
export OPENAI_API_KEY="your-key-here"
export OPENAI_BASE_URL="https://api.openai.com/v1"
export GENERATION_MODEL="gpt-4o-mini"
export VALIDATION_MODEL="gpt-4o-mini"
```

## 快速测试（5分钟）

### 方法1：使用便捷脚本

```bash
cd /data_ali/shunian/verl/scripts/sft_openai

# 100个样本，使用cluster采样
./run_with_balanced_sampling.sh 100 2 cluster
```

参数说明：
- `100`: 采样数量
- `2`: 每个数据项生成的问题数
- `cluster`: 采样策略（默认）

### 方法2：直接命令

```bash
python3 data_construction_gpt_pipeline.py \
  --source /data_ali/shunian/data/iceberg/scripts/data_clean.json \
  --output ./test_output \
  --sample 100 \
  --examples-per-item 2 \
  --sampling-strategy cluster \
  --seed 42
```

## 采样策略详解

### 可用策略

| 策略 | 说明 | 适用场景 |
|------|------|----------|
| `cluster` (默认) | 最大化category覆盖率 | **推荐用于生产** |
| `balanced` | 在categories间均衡采样 | 类似cluster |
| `random` | 纯随机采样 | Baseline比较 |
| `sequential` | 顺序取前N个 | 快速测试 |

### Cluster策略性能

| 样本数 | 覆盖Categories | 覆盖率 | 每个Category最大样本数 |
|--------|---------------|--------|---------------------|
| 100 | 99 | 99% | 1-2 |
| 1,000 | 996 | 99.6% | 1-2 |
| 10,000 | 9,801 | 9.8% | 1-3 |
| 100,000 | ~60,000 | ~60% | 2-5 |

## 完整命令行参数

```bash
python3 data_construction_gpt_pipeline.py \
  --source <path-to-source-json> \          # 必需：源数据路径
  --output <output-directory> \             # 必需：输出目录
  --sample <N> \                            # 可选：采样数量（默认：全部）
  --examples-per-item <N> \                 # 每项生成问题数（默认：2）
  --sampling-strategy <strategy> \          # 采样策略（默认：cluster）
  --seed <N> \                              # 随机种子（默认：42）
  --max-concurrent <N> \                    # 最大并发请求（默认：10）
  --batch-size <N> \                        # 批次大小（默认：100）
  --checkpoint-interval <N>                 # 检查点间隔（默认：500）
```

## 使用示例

### 小规模测试（100样本）

```bash
python3 data_construction_gpt_pipeline.py \
  --source /data_ali/shunian/data/iceberg/scripts/data_clean.json \
  --output ./output_test \
  --sample 100 \
  --examples-per-item 2 \
  --sampling-strategy cluster
```

预期成本：约 $18 (100 items × 2 examples × $0.091/example)

### 中等规模（1,000样本）

```bash
python3 data_construction_gpt_pipeline.py \
  --source /data_ali/shunian/data/iceberg/scripts/data_clean.json \
  --output ./output_1k \
  --sample 1000 \
  --examples-per-item 3 \
  --sampling-strategy cluster \
  --seed 42
```

预期成本：约 $273 (1,000 items × 3 examples × $0.091/example)

### 生产规模（50,000样本）

```bash
python3 data_construction_gpt_pipeline.py \
  --source /data_ali/shunian/data/iceberg/scripts/data_clean.json \
  --output ./output_production \
  --sample 50000 \
  --examples-per-item 3 \
  --sampling-strategy cluster \
  --seed 42 \
  --max-concurrent 20 \
  --batch-size 200
```

预期成本：约 $13,650 (50,000 items × 3 examples × $0.091/example)

## 验证采样效果

### 1. 测试采样策略

```bash
cd /data_ali/shunian/verl/scripts/sft_openai
./test_category_sampling.sh
```

这将比较所有采样策略并显示category分布统计。

### 2. 分析生成数据的category分布

```bash
python3 -c "
import json
from collections import Counter

with open('./output_test/generated_examples.jsonl', 'r') as f:
    examples = [json.loads(line) for line in f]

category_dist = Counter()
for ex in examples:
    metadata = ex.get('metadata', {})
    cats = metadata.get('source_categories', [])
    if cats:
        category_dist[cats[0]] += 1

print(f'Unique categories: {len(category_dist)}')
print(f'Total examples: {sum(category_dist.values())}')
print(f'Min per category: {min(category_dist.values())}')
print(f'Max per category: {max(category_dist.values())}')
print(f'Mean per category: {sum(category_dist.values()) / len(category_dist):.2f}')
"
```

## 成本优化建议

### 降低成本方案

1. **使用更小的sample size**
   ```bash
   --sample 10000  # 而不是 50000
   ```

2. **减少每项生成的examples**
   ```bash
   --examples-per-item 2  # 而不是 3
   ```

3. **使用cluster采样获得最大diversity**
   ```bash
   --sampling-strategy cluster  # 相同成本下更多category覆盖
   ```

### 成本对比

| 配置 | 样本数 | Examples/Item | Category覆盖 | 成本 |
|------|--------|---------------|--------------|------|
| 基础 | 1,000 | 2 | ~996 | $182 |
| 推荐 | 10,000 | 3 | ~9,800 | $2,730 |
| 高级 | 50,000 | 3 | ~40,000 | $13,650 |

## 输出文件说明

Pipeline完成后，输出目录将包含：

```
output_dir/
├── generated_examples.jsonl       # 所有生成的examples
├── checkpoint.json                # 断点续传文件
├── pipeline_report.json           # 统计报告
└── pipeline_log_*.txt            # 详细日志
```

## 故障排除

### 问题1：No module named 'category_sampling'

**解决方案**：确保 `category_sampling.py` 在同一目录
```bash
cd /data_ali/shunian/verl/scripts/sft_openai
ls category_sampling.py  # 应该存在
```

### 问题2：API Key错误

**解决方案**：检查环境变量
```bash
echo $OPENAI_API_KEY
echo $OPENAI_BASE_URL
```

### 问题3：采样策略不生效

**解决方案**：确认使用了 `--sample` 参数
```bash
# 错误 - 没有sample参数，策略不会生效
python3 data_construction_gpt_pipeline.py --source ... --output ...

# 正确
python3 data_construction_gpt_pipeline.py --source ... --output ... --sample 1000
```

## 下一步

1. ✅ 小规模测试（100样本）
2. ✅ 检查输出质量
3. ✅ 分析category分布
4. ⬜ 中等规模运行（1,000-10,000样本）
5. ⬜ 生产规模部署

## 参考文档

- `CATEGORY_SAMPLING_README.md` - Category采样详细说明
- `GPT_PIPELINE_README.md` - GPT Pipeline完整文档
- `QUICKSTART_GPT_PIPELINE.md` - 原始快速开始指南

## 技术支持

如有问题，请检查：
1. 日志文件: `pipeline_log_*.txt`
2. 统计报告: `pipeline_report.json`
3. 测试脚本: `./test_category_sampling.sh`
