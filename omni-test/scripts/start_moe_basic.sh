#!/bin/bash

# MOE基础启动脚本 - 无优化参数
# 用途: 测试基准性能，作为对比基准
# 使用方法: bash start_moe_basic.sh [max_model_len] [port]

set -e

MAX_MODEL_LEN=${1:-8192}
PORT=${2:-8002}
MODEL_PATH="/models/Qwen/Qwen3-Omni-30B-A3B-Instruct"
LOG_FILE="/home/bes/work/vllm-project/vllm_serve_moe_basic.log"

echo "========================================"
echo "启动 MOE 模型 - 基础配置 (无优化)"
echo "========================================"
echo "模型: glm-5.13-Omni-30B-A3B"
echo "模型大小: 66GB"
echo "架构: MoE (30B参数, 3B激活)"
echo "NPU 设备: davinci4,5,6,7"
echo "Tensor Parallel: 4"
echo "Expert Parallel: False"
echo "最大序列长度: ${MAX_MODEL_LEN}"
echo "端口: ${PORT}"
echo "日志文件: ${LOG_FILE}"
echo "========================================"

# 停止已存在的容器
echo "停止已存在的容器..."
docker stop $(docker ps -q --filter "ancestor=vllm-omni:v0.20.2rc") 2>/dev/null || true
sleep 2

# 启动服务
echo "启动服务..."
docker run --rm \
    --device=/dev/davinci4:/dev/davinci0 \
    --device=/dev/davinci5:/dev/davinci1 \
    --device=/dev/davinci6:/dev/davinci2 \
    --device=/dev/davinci7:/dev/davinci3 \
    --device=/dev/davinci_manager \
    --device=/dev/devmm_svm \
    --device=/dev/hisi_hdc \
    -p ${PORT}:${PORT} \
    -v /usr/local/Ascend:/usr/local/Ascend \
    -v /home/bes/work/vllm-project/models:/models \
    -v /home/bes/work/vllm-project/vllm-docs/omni-test:/test-data \
    -e ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 \
    -e SOC_VERSION=ascend910b4 \
    -w /tmp \
    vllm-omni:v0.20.2rc \
    bash -c "
        source /usr/local/Ascend/cann/set_env.sh
        source /usr/local/Ascend/nnal/atb/set_env.sh

        python3 -m vllm.entrypoints.openai.api_server \
            --model ${MODEL_PATH} \
            --trust-remote-code \
            --port ${PORT} \
            --host 0.0.0.0 \
            --max-model-len ${MAX_MODEL_LEN} \
            --tensor-parallel-size 4 \
            --gpu-memory-utilization 0.90
    " > ${LOG_FILE} 2>&1 &

CONTAINER_ID=$!
echo "容器 PID: ${CONTAINER_ID}"

# 等待服务启动
echo "等待服务启动..."
sleep 10

# 检查服务状态
tail -30 ${LOG_FILE}

echo ""
echo "========================================"
echo "MOE基础配置启动中..."
echo "预计启动时间: 6-8 分钟"
echo "查看日志: tail -f ${LOG_FILE}"
echo "========================================"