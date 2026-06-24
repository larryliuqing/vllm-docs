#!/bin/bash

# MOE参数效果对比测试脚本
# 用途: 快速对比不同MOE参数配置的性能差异
# 使用方法: bash test_moe_comparison.sh

set -e

PORT_BASE=8002
MODEL_PATH="/models/Qwen/Qwen3-Omni-30B-A3B-Instruct"
RESULT_DIR="/home/bes/work/vllm-project/vllm-docs/omni-test/moe-results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "========================================"
echo "MOE 参数效果对比测试"
echo "========================================"
echo "测试维度:"
echo "  1. 内存占用对比"
echo "  2. 吞吐量对比"
echo "  3. 延迟稳定性对比"
echo "  4. 专家激活分布对比"
echo "========================================"

mkdir -p ${RESULT_DIR}

# 测试函数
test_config() {
    local CONFIG_NAME=$1
    local SCRIPT=$2
    local PORT=$3
    local LOG_FILE="${RESULT_DIR}/${CONFIG_NAME}_${TIMESTAMP}.log"

    echo ""
    echo "========================================"
    echo "测试配置: ${CONFIG_NAME}"
    echo "========================================"

    # 启动服务
    bash ${SCRIPT} 8192 ${PORT}

    # 等待服务就绪
    echo "等待服务就绪..."
    sleep 300

    # 测试性能
    echo "执行性能测试..."

    # 1. 简单请求测试
    local SIMPLE_RESULT=$(curl -s http://localhost:${PORT}/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{
            "model": "Qwen3-Omni-30B-A3B-Instruct",
            "messages": [{"role": "user", "content": "你好"}],
            "max_tokens": 50
        }' | jq -r '.choices[0].message.content')

    echo "简单请求结果: ${SIMPLE_RESULT}"

    # 2. 吞吐量测试 (批量请求)
    echo "测试吞吐量 (10个并发请求)..."
    local START_TIME=$(date +%s.%N)

    for i in {1..10}; do
        curl -s http://localhost:${PORT}/v1/chat/completions \
            -H "Content-Type: application/json" \
            -d '{
                "model": "Qwen3-Omni-30B-A3B-Instruct",
                "messages": [{"role": "user", "content": "请介绍一下MOE架构的优势"}],
                "max_tokens": 100
            }' > /dev/null &
    done

    wait
    local END_TIME=$(date +%s.%N)
    local DURATION=$(echo "$END_TIME - $START_TIME" | bc)
    local THROUGHPUT=$(echo "1000 / $DURATION" | bc)

    echo "吞吐量: ${THROUGHPUT} tok/s"

    # 3. 记录内存占用
    echo "记录内存占用..."
    npu-smi info -t usages -i 0 > "${RESULT_DIR}/${CONFIG_NAME}_memory_${TIMESTAMP}.txt"

    # 4. 记录专家激活统计 (如果有metrics端点)
    curl -s http://localhost:${PORT}/metrics > "${RESULT_DIR}/${CONFIG_NAME}_metrics_${TIMESTAMP}.txt" || true

    # 保存测试结果
    cat > "${RESULT_DIR}/${CONFIG_NAME}_summary_${TIMESTAMP}.md" << EOF
## MOE测试结果 - ${CONFIG_NAME}

**测试时间**: ${TIMESTAMP}
**配置**: ${SCRIPT}

### 性能指标

- **吞吐量**: ${THROUGHPUT} tok/s
- **测试时长**: ${DURATION} 秒
- **简单请求结果**: ${SIMPLE_RESULT}

### 内存占用

见: ${CONFIG_NAME}_memory_${TIMESTAMP}.txt

### Metrics数据

见: ${CONFIG_NAME}_metrics_${TIMESTAMP}.txt

### 日志文件

见: ${CONFIG_NAME}_${TIMESTAMP}.log
EOF

    echo "测试结果已保存: ${RESULT_DIR}/${CONFIG_NAME}_summary_${TIMESTAMP}.md"

    # 停止服务
    docker stop $(docker ps -q --filter "ancestor=vllm-omni:v0.20.2rc") 2>/dev/null || true
    sleep 10
}

# 测试配置列表
CONFIGS=(
    "basic|start_moe_basic.sh|8002"
    "expert_parallel|start_moe_expert_parallel.sh|8003"
    "eplb|start_moe_eplb.sh|8004"
    "optimized|start_moe_optimized.sh|8005"
)

echo ""
echo "开始测试..."
echo "将依次测试4种配置:"
echo "  1. Basic - 无优化基准"
echo "  2. Expert Parallel - 专家并行"
echo "  3. EPLB - 动态负载均衡"
echo "  4. Optimized - 全优化"
echo ""
echo "总预计时间: 40-50 分钟"
echo "结果保存目录: ${RESULT_DIR}"
echo ""

read -p "确认开始测试? (y/n): " CONFIRM

if [[ ${CONFIRM} == "y" ]]; then
    for CONFIG in "${CONFIGS[@]}"; do
        IFS='|' read -r NAME SCRIPT PORT <<< "$CONFIG"
        test_config "${NAME}" "${SCRIPT}" "${PORT}"
    done

    echo ""
    echo "========================================"
    echo "所有测试完成!"
    echo "========================================"
    echo ""
    echo "结果对比:"
    echo ""
    for CONFIG in "${CONFIGS[@]}"; do
        IFS='|' read -r NAME SCRIPT PORT <<< "$CONFIG"
        SUMMARY="${RESULT_DIR}/${NAME}_summary_${TIMESTAMP}.md"
        if [ -f "${SUMMARY}" ]; then
            echo "### ${NAME}"
            grep "吞吐量" "${SUMMARY}"
            echo ""
        fi
    done

    echo ""
    echo "详细结果见: ${RESULT_DIR}"
    echo "对比表格见: ${RESULT_DIR}/comparison_table_${TIMESTAMP}.md"
fi