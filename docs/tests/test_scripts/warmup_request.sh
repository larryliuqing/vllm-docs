#!/bin/bash
# 预热请求脚本 - 消除PD分离服务的冷启动延迟
# 用法: bash warmup_request.sh [proxy_url] [model_path] [num_warmups]

PROXY_URL="${1:-http://127.0.0.1:8000}"
MODEL="${2:-/home/la/work/vllm-project/models/Qwen/Qwen2-VL-7B-Instruct}"
NUM_WARMUPS="${3:-3}"

echo "========================================="
echo "PD分离服务预热"
echo "========================================="
echo "Proxy URL: $PROXY_URL"
echo "Model:     $MODEL"
echo "Warmups:   $NUM_WARMUPS"
echo ""

for i in $(seq 1 $NUM_WARMUPS); do
    echo -n "  请求 #${i}... "
    start_ms=$(date +%s%3N)

    response=$(curl -s -w "\n%{http_code}" -X POST "${PROXY_URL}/v1/completions" \
        -H 'Content-Type: application/json' \
        -d "{
            \"model\": \"${MODEL}\",
            \"prompt\": \"Hello, warmup request\",
            \"max_tokens\": 5,
            \"temperature\": 0.0
        }")

    end_ms=$(date +%s%3N)
    latency_ms=$((end_ms - start_ms))

    http_code=$(echo "$response" | tail -1)

    if [ "$http_code" = "200" ]; then
        echo "✓ (${latency_ms}ms)"
    else
        echo "✗ HTTP ${http_code} (${latency_ms}ms)"
    fi

    sleep 1
done

echo ""
echo "预热完成"
