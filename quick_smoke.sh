#!/bin/bash
# 快速启动脚本 - 仅运行冒烟版本

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

# 激活虚拟环境或创建
if [ ! -d ".venv" ]; then
    echo "创建虚拟环境..."
    python3 -m venv .venv
fi

source .venv/bin/activate

echo "更新pip并安装依赖..."
python -m pip install -q --upgrade pip
pip install -q -r requirements.txt

echo ""
echo "================================"
echo "运行CPU冒烟版本"
echo "场景: moderate_practical, hard_stress, channel_harsh"
echo "迭代: 2 | 种子: 101 | 设备: CPU"
echo "预计时间: 5-10 分钟"
echo "输出目录: logs/bench_cpu_smoke"
echo "================================"
echo ""

python -m scripts.benchmark_constraint_scenarios \
  --cfg configs/default.yaml \
  --device cpu \
  --scenarios moderate_practical hard_stress channel_harsh \
  --fast-mode \
  --meta-iters 2 \
  --seeds 101 \
  --out-dir logs/bench_cpu_smoke

echo ""
echo "✓ 冒烟测试完成！"
echo "结果已保存到: logs/bench_cpu_smoke/"
echo ""
echo "下一步可选："
echo "  中间版本:   RUN_FORMAL=mid ./run_benchmarks.sh"
echo "  正式版本:   RUN_FORMAL=full ./run_benchmarks.sh"
echo "  或参考:     BENCHMARK_GUIDE.md"
