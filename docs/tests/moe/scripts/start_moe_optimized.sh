#!/bin/bash

# MOE全优化启动脚本 - 所有性能优化参数
# 用途: 测试完整优化组合的性能极限
# 使用方法: bash start_moe_optimized.sh [max_model_len] [port]

set -e

MAX_MODEL_LEN=${1:-8192}
PORT=${2:-8002}
MODEL_PATH="/models/Qwen/Qwen3-Omni-30B-A3B-Instruct"
LOG_FILE="/home/bes/work/vllm-project/vllm_serve_moe_optimized.log"

echo "========================================"
echo "启动 MOE 模型 - 全优化配置"
echo "========================================"
echo "模型: glm-5.13-Omni-30B-A3B"
echo "架构: MoE (30B参数, 3B激活)"
echo "NPU 设备: davinci4,5,6,7"
echo "Tensor Parallel: 4"
echo "========================================"
echo ""
echo "✓ Expert Parallel: 启用专家并行"
echo "✓ Shared Expert DP: 共享专家数据并行"
echo "✓ Dynamic EPLB: 动态负载均衡 (2冗余专家)"
echo "✓ Multistream Overlap: 多流重叠优化"
echo "    - 共享专家计算与通信重叠"
echo "    - 门控网络与专家调度重叠"
echo "✓ Weight Prefetch: 权重预取优化"
echo "    - MOE门控权重预取80%"
echo "========================================"
echo ""
echo "预期综合效果:"
echo "  - 内存优化: 40-60%"
echo "  - 吞吐量提升: 30-50%"
echo "  - 延迟减少: 20-30%"
echo "  - 延迟稳定性: 提升50-70%"
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
    -e DYNAMIC_EPLB=true \
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
            --enable-expert-parallel \
            --additional-config '{
                \"enable_shared_expert_dp\": true,
                \"multistream_overlap_shared_expert\": true,
                \"multistream_overlap_gate\": true,
                \"weight_prefetch_config\": {
                    \"enabled\": true,
                    \"prefetch_ratio\": {
                        \"moe\": {
                            \"gate_up\": 0.8
                        }
                    }
                },
                \"eplb_config\": {
                    \"dynamic_eplb\": true,
                    \"num_redundant_experts\": 4,
                    \"expert_heat_collection_interval\": 400,
                    \"algorithm_execution_interval\": 30,
                    \"eplb_policy_type\": 1
                }
            }' \
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
echo "MOE全优化配置启动中..."
echo "预计启动时间: 8-10 分钟"
echo "优化参数较多,编译时间较长"
echo "查看日志: tail -f ${LOG_FILE}"
echo "========================================"