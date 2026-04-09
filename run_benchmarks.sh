#!/bin/bash

# ============================================================================
# TC-HHMRL 基准测试运行脚本
# ============================================================================
# 运行三个场景的基准测试：moderate_practical, hard_stress, channel_harsh
# 支持冒烟版本（快速）和正式版本（完整）
# ============================================================================

set -e  # 任何命令失败则停止

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}TC-HHMRL 基准测试启动${NC}"
echo -e "${GREEN}========================================${NC}"

# ============================================================================
# 步骤 1: 检查和创建Python虚拟环境
# ============================================================================
echo -e "\n${YELLOW}[步骤 1] 检查Python虚拟环境...${NC}"

if [ ! -d ".venv" ]; then
    echo "创建虚拟环境..."
    python3 -m venv .venv
fi

echo "激活虚拟环境..."
source .venv/bin/activate

echo -e "${GREEN}✓ 虚拟环境已激活${NC}"

# ============================================================================
# 步骤 2: 安装/更新依赖
# ============================================================================
echo -e "\n${YELLOW}[步骤 2] 安装依赖...${NC}"
python -m pip install --upgrade pip -q
pip install -q -r requirements.txt
echo -e "${GREEN}✓ 依赖已安装${NC}"

# ============================================================================
# 步骤 3: 运行CPU冒烟版本（快速验证）
# ============================================================================
echo -e "\n${YELLOW}[步骤 3] 运行CPU冒烟版本...${NC}"
echo -e "${YELLOW}场景: moderate_practical, hard_stress, channel_harsh${NC}"
echo -e "${YELLOW}迭代次数: 2 (快速验证)${NC}"
echo -e "${YELLOW}种子: 101${NC}"
echo -e "${YELLOW}输出目录: logs/bench_cpu_smoke${NC}"

python -m scripts.benchmark_constraint_scenarios \
  --cfg configs/default.yaml \
  --device cpu \
  --scenarios moderate_practical hard_stress channel_harsh \
  --fast-mode \
  --meta-iters 2 \
  --seeds 101 \
  --out-dir logs/bench_cpu_smoke

echo -e "${GREEN}✓ CPU冒烟测试完成！${NC}"
echo -e "${GREEN}结果已保存到: logs/bench_cpu_smoke/${NC}"

# ============================================================================
# 步骤 4: 询问是否运行正式版本
# ============================================================================
echo -e "\n${YELLOW}========================================${NC}"
echo -e "${YELLOW}冒烟测试完成！${NC}"
echo -e "${YELLOW}========================================${NC}"

echo -e "\n选择下一步:"
echo "1) 跑中间版本（10迭代，2个种子，更快）"
echo "2) 跑正式版本（45迭代，5个种子，完整实验）"
echo "3) 跳过正式版本，仅保留冒烟结果"
echo ""

# 默认不自动运行，让用户选择
# 如果想要自动运行某个版本，修改下面的条件
if [ "${RUN_FORMAL:-0}" == "full" ]; then
    echo -e "${YELLOW}[步骤 4] 运行正式版本...${NC}"

    echo -e "${YELLOW}场景: moderate_practical, hard_stress, channel_harsh${NC}"
    echo -e "${YELLOW}迭代次数: 45${NC}"
    echo -e "${YELLOW}种子: 101 202 303 404 505${NC}"
    echo -e "${YELLOW}输出目录: logs/bench_cpu_full${NC}"
    echo -e "${YELLOW}预计耗时: 较长，建议使用 tmux/screen 挂后台${NC}"

    python -m scripts.benchmark_constraint_scenarios \
      --cfg configs/default.yaml \
      --device cpu \
      --scenarios moderate_practical hard_stress channel_harsh \
      --meta-iters 45 \
      --seeds 101 202 303 404 505 \
      --out-dir logs/bench_cpu_full

    echo -e "${GREEN}✓ 正式版本完成！${NC}"

elif [ "${RUN_FORMAL:-0}" == "mid" ]; then
    echo -e "${YELLOW}[步骤 4] 运行中间版本...${NC}"

    echo -e "${YELLOW}场景: moderate_practical, hard_stress, channel_harsh${NC}"
    echo -e "${YELLOW}迭代次数: 10${NC}"
    echo -e "${YELLOW}种子: 101 202${NC}"
    echo -e "${YELLOW}输出目录: logs/bench_cpu_mid${NC}"

    python -m scripts.benchmark_constraint_scenarios \
      --cfg configs/default.yaml \
      --device cpu \
      --scenarios moderate_practical hard_stress channel_harsh \
      --meta-iters 10 \
      --seeds 101 202 \
      --out-dir logs/bench_cpu_mid

    echo -e "${GREEN}✓ 中间版本完成！${NC}"
else
    echo -e "${YELLOW}跳过正式版本${NC}"
fi

# ============================================================================
# 完成
# ============================================================================
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}所有测试完成！${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "结果位置:"
echo "  - 冒烟版本: logs/bench_cpu_smoke/"
if [ "${RUN_FORMAL:-0}" == "full" ]; then
    echo "  - 正式版本: logs/bench_cpu_full/"
elif [ "${RUN_FORMAL:-0}" == "mid" ]; then
    echo "  - 中间版本: logs/bench_cpu_mid/"
fi
echo ""
echo "每个场景目录包含:"
echo "  - 图表: convergence.png, final_metrics.png, env_realism.png 等"
echo "  - 数据: training.csv, eval.csv, convergence.csv 等"
echo "  - 汇总: summary.json"
