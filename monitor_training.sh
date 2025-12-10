#!/bin/bash
# 🔍 训练监控脚本 - 实时查看训练进度

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║     🔍 AIRR-ML-25 训练监控面板 🔍                          ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# 查找最新日志文件
LATEST_LOG=$(ls -t /home/thc1006/dev/airr-ml25-package/logs/auto_train_*.log 2>/dev/null | head -1)

if [ -z "$LATEST_LOG" ]; then
    echo -e "${RED}❌ 未找到训练日志${NC}"
    exit 1
fi

echo -e "${BLUE}📝 日志文件: $LATEST_LOG${NC}"
echo ""

# 检查进程
PROCESS=$(ps aux | grep championship_dl.py | grep -v grep)
if [ -z "$PROCESS" ]; then
    echo -e "${RED}❌ 训练进程未运行${NC}"
else
    PID=$(echo $PROCESS | awk '{print $2}')
    CPU=$(echo $PROCESS | awk '{print $3}')
    MEM=$(echo $PROCESS | awk '{print $4}')
    echo -e "${GREEN}✅ 训练进程运行中${NC}"
    echo "   PID: $PID | CPU: ${CPU}% | RAM: ${MEM}%"
fi
echo ""

# GPU状态
echo -e "${YELLOW}🎮 GPU 状态:${NC}"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit --format=csv,noheader | while IFS=',' read -r gpu_util mem_used mem_total temp power power_limit; do
    echo "   利用率: $gpu_util | 内存: $mem_used / $mem_total"
    echo "   温度: $temp | 功耗: $power / $power_limit"
done
echo ""

# 训练进度 - 查找最近的进度行
echo -e "${BLUE}📊 训练进度 (最近20行):${NC}"
tail -20 "$LATEST_LOG" | grep -E "Dataset|Loading|Train|Val|Epoch|AUC|Loss|Fold" | tail -10
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "💡 快速命令:"
echo "   实时日志: tail -f $LATEST_LOG"
echo "   停止训练: kill $PID"
echo "   GPU监控: watch -n 1 nvidia-smi"
echo ""
