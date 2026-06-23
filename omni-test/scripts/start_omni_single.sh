#!/bin/bash

# Qwen2.5-Omni-7B 单卡测试启动脚本
# 使用方法: bash start_omni_single.sh [max_model_len] [port]

set -e

# 参数配置
MAX_MODEL_LEN=${1:-8192}
PORT=${2:-8001}
NPU_DEVICE=${3:-5}
MODEL_PATH="/models/Qwen/Qwen2___5-Omni-7B"
LOG_FILE="/home/bes/work/vllm-project/vllm_serve_omni_1npu.log"

echo "========================================"
echo "启动 Qwen2.5-Omni-7B 服务"
echo "========================================"
echo "NPU 设备: davinci${NPU_DEVICE}"
echo "最大序列长度: ${MAX_MODEL_LEN}"
echo "端口: ${PORT}"
echo "日志文件: ${LOG_FILE}"
echo "========================================"

# 停止已存在的容器
echo "停止已存在的容器..."
docker stop $(docker ps -q --filter "ancestor=vllm-omni:v0.20.2rc") 2>/dev/null || true

# 启动服务
echo "启动服务..."
docker run --rm \
    --device=/dev/davinci${NPU_DEVICE}:/dev/davinci0 \
    --device=/dev/davinci_manager \
    --device=/dev/devmm_svm \
    --device=/dev/hisi_hdc \
    -v /usr/local/Ascend:/usr/local/Ascend \
    -v /home/bes/work/vllm-project/models:/models \
    -v /home/bes/work/vllm-project/vllm-docs/docs/omni-test:/test-data \
    -e ASCEND_RT_VISIBLE_DEVICES=0 \
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
            --tensor-parallel-size 1 \
            --gpu-memory-utilization 0.85
    " > ${LOG_FILE} 2>&1 &

CONTAINER_ID=$!
echo "容器 PID: ${CONTAINER_ID}"

# 等待服务启动
echo "等待服务启动..."
sleep 10

# 检查服务状态
tail -20 ${LOG_FILE}

echo ""
echo "========================================"
echo "服务启动中，请等待约 4 分钟..."
echo "查看日志: tail -f ${LOG_FILE}"
echo "========================================"
