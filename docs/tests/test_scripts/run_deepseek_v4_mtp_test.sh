#!/bin/bash
# DS-V4-Flash-w4a8-mtp MTP 推测解码多 token 测试脚本
# 测试 num_speculative_tokens = 1, 3, 5 的效果对比
# 使用 NPU 0-7, TP=8
# 测试日期: 2026-06-29

set -e

# 设置CANN环境
source /usr/local/Ascend/cann-9.0.0/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

# 设置库路径
export LIBRARY_PATH=/usr/local/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

# HCCL相关配置
export HCCL_EXEC_TIMEOUT=204
export HCCL_CONNECT_TIMEOUT=120
export HCCL_IF_IP=127.0.0.1
export HCCL_INTRA_ROCE_ENABLE=0
export HCCL_INTRA_PCIE_ENABLE=1
export GLOO_SOCKET_IFNAME="lo"
export TP_SOCKET_IFNAME="lo"
export HCCL_SOCKET_IFNAME="lo"

# DS V4 专用环境变量
export VLLM_ASCEND_APPLY_DSV4_PATCH=1
export USE_MULTI_BLOCK_POOL=1
export USE_MULTI_GROUPS_KV_CACHE=1
export HCCL_BUFFSIZE=1024
export VLLM_ASCEND_ENABLE_FUSED_MC2=0
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export ASCEND_LAUNCH_BLOCKING=0

MODEL_PATH="/home/la/work/vllm-project/models/DS-V4-Flash-w4a8-mtp"
TEST_OUTPUT_DIR="/home/la/work/vllm-project/vllm-docs/docs/tests/test_scripts/mtp_test_results"
mkdir -p "$TEST_OUTPUT_DIR"

# 测试参数组合
SPEC_TOKENS_LIST=(1 3 5)
TEMPERATURE=0
MAX_TOKENS=256
PROMPTS=(
    "Hello, my name is"
    "The president of the United States is"
    "The capital of France is"
    "The future of AI is"
    "Write a short story about a robot learning to paint:"
)

for NUM_SPEC in "${SPEC_TOKENS_LIST[@]}"; do
    echo ""
    echo "========================================="
    echo "测试: num_speculative_tokens = $NUM_SPEC"
    echo "时间: $(date)"
    echo "========================================="

    # 清理之前的服务
    pkill -f "vllm serve" 2>/dev/null || true
    sleep 5

    # 启动 vLLM 服务
    vllm serve "$MODEL_PATH" \
        --host 0.0.0.0 \
        --port 8000 \
        --tensor-parallel-size 8 \
        --data-parallel-size 1 \
        --enable-expert-parallel \
        --api-server-count 1 \
        --max-model-len 8192 \
        --max-num-batched-tokens 10240 \
        --max-num-seqs 64 \
        --gpu-memory-utilization 0.95 \
        --trust-remote-code \
        --tokenizer-mode ds_v4 \
        --tool-call-parser ds_v4 \
        --enable-auto-tool-choice \
        --reasoning-parser deepseek_v4 \
        --enable-prefix-caching \
        --safetensors-load-strategy prefetch \
        --quantization ascend \
        --block-size 128 \
        --speculative-config "{\"num_speculative_tokens\": $NUM_SPEC, \"method\": \"mtp\", \"enforce_eager\": true}" \
        --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
        --async-scheduling \
        --additional-config '{"ascend_compilation_config": {"enable_npugraph_ex": true, "enable_static_kernel": false}, "enable_cpu_binding": "true", "enable_shared_expert_dp": true, "multistream_overlap_shared_expert": true, "multistream_dsa_preprocess": false}' \
        --seed 1024 \
        > "$TEST_OUTPUT_DIR/server_spec${NUM_SPEC}.log" 2>&1 &

    SERVER_PID=$!
    echo "服务 PID: $SERVER_PID"

    # 等待服务启动
    echo "等待服务启动..."
    for i in $(seq 1 120); do
        if curl -s http://127.0.0.1:8000/v1/completions \
            -H "Content-Type: application/json" \
            -d '{"model":"'"$MODEL_PATH"'","prompt":"test","max_tokens":1,"temperature":0}' \
            > /dev/null 2>&1; then
            echo "服务启动成功 (耗时 ${i}s)"
            break
        fi
        if [ $i -eq 120 ]; then
            echo "服务启动超时！"
            cat "$TEST_OUTPUT_DIR/server_spec${NUM_SPEC}.log"
            exit 1
        fi
        sleep 1
    done

    # 逐条 prompt 测试
    for idx in "${!PROMPTS[@]}"; do
        PROMPT="${PROMPTS[$idx]}"
        echo ""
        echo "--- Prompt $((idx+1)): ${PROMPT:0:50}... ---"

        RESPONSE_FILE="$TEST_OUTPUT_DIR/response_spec${NUM_SPEC}_prompt${idx}.json"
        curl -s http://127.0.0.1:8000/v1/completions \
            -H "Content-Type: application/json" \
            -d "{
                \"model\": \"$MODEL_PATH\",
                \"prompt\": \"$PROMPT\",
                \"max_tokens\": $MAX_TOKENS,
                \"temperature\": $TEMPERATURE,
                \"stream\": false
            }" > "$RESPONSE_FILE"

        # 提取生成文本和 token 数
        GENERATED_TEXT=$(python3 -c "
import json
with open('$RESPONSE_FILE') as f:
    data = json.load(f)
print(data['choices'][0]['text'][:200])
print('---TOKEN_STATS---')
print('prompt_tokens:', data['usage']['prompt_tokens'])
print('completion_tokens:', data['usage']['completion_tokens'])
" 2>/dev/null || echo "解析失败")
        echo "$GENERATED_TEXT"
    done

    # 获取服务 metrics（推测解码统计）
    echo ""
    echo "--- 获取 Metrics ---"
    curl -s http://127.0.0.1:8000/v1/metrics \
        > "$TEST_OUTPUT_DIR/metrics_spec${NUM_SPEC}.txt" 2>/dev/null || echo "metrics 接口不可用"

    # 从日志提取 acceptance 统计
    echo ""
    echo "--- 从日志提取 Spec Decode 统计 ---"
    grep -i "spec_decode\|draft\|accept\|reject" "$TEST_OUTPUT_DIR/server_spec${NUM_SPEC}.log" \
        | tail -20 || echo "未找到相关统计"

    # 停止服务
    echo ""
    echo "停止服务 (PID: $SERVER_PID)..."
    kill $SERVER_PID 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
    sleep 3
done

echo ""
echo "========================================="
echo "所有测试完成！"
echo "结果目录: $TEST_OUTPUT_DIR"
echo "========================================="
echo ""
echo "查看各 token 数的服务日志:"
for NUM_SPEC in "${SPEC_TOKENS_LIST[@]}"; do
    echo "  - num_speculative_tokens=$NUM_SPEC: less $TEST_OUTPUT_DIR/server_spec${NUM_SPEC}.log"
done