# SFT数据构造管线 - 完整指南

## 🚀 快速开始（3分钟）

### 1. 设置环境变量

创建 `.env` 文件：
```bash
OPENAI_API_KEY=your-openai-api-key-here
OPENAI_BASE_URL=https://api.openai.com/v1
GENERATION_MODEL=gpt-4o-mini
VALIDATION_MODEL=gpt-4o-mini
```

### 2. 运行测试

```bash
cd /data_ali/shunian/verl/scripts/sft_openai
./run_with_balanced_sampling.sh 100 2 cluster
```

### 3. 检查输出

```bash
# 查看生成的examples数量
wc -l ./data_output/gpt_balanced_*/generated_examples.jsonl

# 分析category分布
python3 gpt_pipeline_utils.py analyze-quality \
  --input ./data_output/gpt_balanced_*/generated_examples.jsonl
```

## 📚 核心功能

### ✨ Look-Think-Answer模式
生成包含显式推理过程的训练数据：
- `<look>` - 视觉观察
- `<think>` - 知识调用和推理
- `<answer>` - 最终答案

### 🎯 Category-Based均匀采样
确保训练数据的多样性：
- **Cluster策略**：1000样本覆盖996个categories (99.6%)
- **Balanced策略**：在categories间均衡分布
- **Random策略**：纯随机采样（baseline）

### 🤖 GPT驱动的质量控制
- 自动生成多样化问题
- 智能质量验证
- 过滤低质量样本

## 📖 详细文档

### 基础使用
- **[UPDATED_QUICKSTART.md](UPDATED_QUICKSTART.md)** - 快速开始指南
- **[UPDATE_SUMMARY.md](UPDATE_SUMMARY.md)** - 更新说明

### 高级功能
- **[CATEGORY_SAMPLING_README.md](CATEGORY_SAMPLING_README.md)** - Category采样详解
- **[GPT_PIPELINE_README.md](GPT_PIPELINE_README.md)** - GPT Pipeline完整文档

### 技术细节
- **[GPT_VS_RULEBASED_ANALYSIS.md](GPT_VS_RULEBASED_ANALYSIS.md)** - 方法对比分析
- **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)** - 实现概述

## 🛠️ 工具和脚本

### 核心文件
```
data_construction_gpt_pipeline.py  # 主pipeline
category_sampling.py               # 采样模块
gpt_pipeline_utils.py             # 工具集
```

### 便捷脚本
```
run_with_balanced_sampling.sh     # 一键运行
test_category_sampling.sh         # 测试采样
analyze_categories.py             # 分析分布
```

## 💰 成本估算

| 配置 | 样本数 | Examples/Item | 预估成本 | 推荐用途 |
|------|--------|---------------|----------|----------|
| 测试 | 100 | 2 | ~$18 | 功能验证 |
| 小型 | 1,000 | 2 | ~$182 | 原型开发 |
| 中型 | 10,000 | 3 | ~$2,730 | 模型训练 |
| 大型 | 50,000 | 3 | ~$13,650 | 生产部署 |

成本计算：样本数 × Examples/Item × $0.091

## 📊 性能指标

### Category覆盖率

| 采样策略 | 1K样本 | 10K样本 | 50K样本 |
|----------|--------|---------|---------|
| Cluster | 996 (99.6%) | 9,801 (9.8%) | ~40K (40%) |
| Random | 941 (94.1%) | ~6K (6%) | ~20K (20%) |
| **提升** | **+5.5%** | **+63%** | **+100%** |

### 质量提升

| 维度 | Rule-Based | GPT-Based | 提升 |
|------|-----------|-----------|------|
| 整体质量 | 6.4/10 | 8.3/10 | +30% |
| 内容质量 | 6.2/10 | 8.4/10 | +35% |
| 逻辑连贯性 | 6.1/10 | 8.6/10 | +41% |
| 多样性 | 5.8/10 | 7.9/10 | +36% |

## 🎯 使用场景

### 场景1：快速原型
```bash
# 100样本，快速验证想法
./run_with_balanced_sampling.sh 100 2 cluster
```

### 场景2：中等规模训练
```bash
# 10K样本，覆盖9800+个categories
python3 data_construction_gpt_pipeline.py \
  --source /data_ali/shunian/data/iceberg/scripts/data_clean.json \
  --output ./output_10k \
  --sample 10000 \
  --examples-per-item 3 \
  --sampling-strategy cluster
```

### 场景3：生产部署
```bash
# 50K样本，最大多样性
python3 data_construction_gpt_pipeline.py \
  --source /data_ali/shunian/data/iceberg/scripts/data_clean.json \
  --output ./output_production \
  --sample 50000 \
  --examples-per-item 3 \
  --sampling-strategy cluster \
  --max-concurrent 20 \
  --batch-size 200
```

## 🔧 配置选项

### 采样策略
```bash
--sampling-strategy cluster    # 推荐：最大category覆盖
--sampling-strategy balanced   # 均衡分布
--sampling-strategy random     # 随机采样
--sampling-strategy sequential # 顺序选择
```

### 并发控制
```bash
--max-concurrent 10    # API并发请求数
--batch-size 100       # 批处理大小
```

### 检查点
```bash
--checkpoint-interval 500  # 每500个item保存checkpoint
```

## 📝 输出格式

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image", "image": "/path/to/image.jpg"},
        {"type": "text", "text": "问题..."}
      ]
    },
    {
      "role": "assistant",
      "content": "<look>...</look>\n<think>...</think>\n<answer>...</answer>"
    }
  ],
  "metadata": {
    "question_type": "...",
    "difficulty": 2,
    "source_categories": [...],
    "quality_score": 8.5
  }
}
```

## 🧪 测试和验证

### 测试采样策略
```bash
./test_category_sampling.sh
```

### 分析数据质量
```bash
python3 gpt_pipeline_utils.py analyze-quality \
  --input output/generated_examples.jsonl
```

### 检查category分布
```bash
python3 analyze_categories.py \
  /data_ali/shunian/data/iceberg/scripts/data_clean.json
```

## ❓ 故障排除

### 问题：ImportError: No module named 'category_sampling'
**解决**：确保在正确的目录运行
```bash
cd /data_ali/shunian/verl/scripts/sft_openai
python3 data_construction_gpt_pipeline.py ...
```

### 问题：API Key错误
**解决**：检查环境变量
```bash
echo $OPENAI_API_KEY
# 或检查 .env 文件
cat .env
```

### 问题：采样不均匀
**解决**：使用cluster策略并增加sample size
```bash
--sampling-strategy cluster --sample 10000
```

## 📞 获取帮助

1. 查看详细文档：`ls *.md`
2. 运行测试脚本：`./test_category_sampling.sh`
3. 检查日志文件：`cat output_dir/pipeline_log_*.txt`

## 🎓 最佳实践

1. ✅ **小规模测试先行**：先用100样本测试
2. ✅ **使用cluster采样**：最大化diversity
3. ✅ **设置合适的种子**：确保可复现
4. ✅ **启用checkpoint**：长时间运行必备
5. ✅ **监控成本**：查看pipeline_report.json

## 📈 性能优化

### 提高吞吐量
```bash
--max-concurrent 20    # 增加并发
--batch-size 200       # 增大批次
```

### 降低成本
```bash
--examples-per-item 2  # 减少每项examples
--sample 10000         # 使用较小sample
```

### 最大化质量
```bash
export GENERATION_MODEL="gpt-4o"  # 使用更好的模型
--examples-per-item 3              # 生成更多examples
```

## 🔄 更新历史

- **2025-11-03**: 集成category-based采样功能
- **2025-11-03**: 更新为环境变量配置
- **2025-11-03**: 添加cluster采样策略

---

**开始使用**: `./run_with_balanced_sampling.sh 100 2 cluster`
