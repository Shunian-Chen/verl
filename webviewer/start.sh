#!/bin/bash
# 推理结果查看器启动脚本

echo "========================================="
echo "    推理结果对比查看器"
echo "========================================="
echo ""
echo "启动服务中..."

cd /root/et/verl/webviewer

# 检查依赖
if ! python3 -c "import fastapi" 2>/dev/null; then
    echo "错误: FastAPI 未安装"
    echo "请运行: pip install fastapi uvicorn[standard]"
    exit 1
fi

# 启动服务
echo "服务启动在: http://0.0.0.0:7860"
echo ""
echo "功能特性："
echo "  • 多模型对比视图 - 同时查看多个模型对同一问题的回答"
echo "  • 完整模型名称显示 - 包含模型名和step数"
echo "  • XML标签高亮 - <look>、<think>、<answer>分别用不同颜色标识"
echo "  • 展开/收起功能 - 可查看完整问题和答案内容"
echo "  • 详细奖励信息 - 显示各项奖励分数和预测结果"
echo "  • 筛选功能 - 按正确性筛选，支持查找答案不一致的样本"
echo ""
echo "按 Ctrl+C 停止服务"
echo "========================================="

uvicorn app:app --host 0.0.0.0 --port 7860 --reload
