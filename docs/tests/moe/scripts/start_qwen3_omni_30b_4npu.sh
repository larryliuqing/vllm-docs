#!/bin/bash

# Qwen3-Omni-30B-A3B 4卡测试启动脚本
# 使用方法: bash start_qwen3_omni_30b_4npu.sh [max_model_len] [port]

set -e

# 参数配置
MAX_MODEL_LEN=${1:-8192}
PORT=${2:-8002}
MODEL_PATH="/models/Qwen/Qwen3-Omni-30B-A3B-Instruct"
LOG_FILE="/home/bes/work/vllm-project/vllm_serve_qwen3_omni_30b_4npu.log"

echo "========================================"
echo "启动 Qwen3-Omni-30B-A3B 服务 (4卡)"
echo "========================================"
echo "模型: Qwen3-Omni-30B-A3B-Instruct"
echo "模型大小: 66GB"
echo "NPU 设备: davinci4,5,6,7"
echo "最大序列长度: ${MAX_MODEL_LEN}"
echo "端口: ${PORT}"
echo "Tensor Parallel: 4"
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
echo "服务启动中，请等待约 6-8 分钟..."
echo "30B 模型加载需要更长时间"
echo "查看日志: tail -f ${LOG_FILE}"
echo "========================================"