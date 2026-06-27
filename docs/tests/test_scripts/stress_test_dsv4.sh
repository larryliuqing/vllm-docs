#!/bin/bash
# DeepSeek V4 压力测试 - 逐步增加负载，观察是否OOM
API="http://127.0.0.1:8000/v1/completions"
MODEL="/home/la/work/vllm-project/models/DeepSeek-V4-Flash-w4a8-mtp"

echo "========================================="
echo " DeepSeek V4 压力测试"
echo " 测试逐步增加负载，观察OOM行为"
echo "========================================="
echo ""

# 辅助函数：并发请求
concurrent_req() {
  local prompt="$1"
  local max_tokens="$2"
  local concurrency="$3"

  echo "[$(date +%H:%M:%S)] 并发=$concurrency, max_tokens=$max_tokens"
  for i in $(seq 1 $concurrency); do
    curl -s "$API" -H "Content-Type: application/json" \
      -d "{\"model\": \"$MODEL\", \"prompt\": \"$prompt $i\", \"max_tokens\": $max_tokens, \"temperature\": 0.1}" \
      -o /dev/null -w "  req#$i: %{http_code} %{time_total}s\n" &
  done
  wait
  echo "  完成"
  echo ""
}

# 第 1 阶段：试探性 - 小并发 + 短输出
echo "=== 阶段 1: 低负载基线 ==="
concurrent_req "Write a short sentence" 10 1
concurrent_req "Write a short sentence" 10 2
sleep 2

# 第 2 阶段：增加输出长度
echo "=== 阶段 2: 增加输出长度 ==="
concurrent_req "Write a paragraph about AI" 100 1
concurrent_req "Write a paragraph about AI" 100 4
sleep 2

# 第 3 阶段：高并发 + 中输出
echo "=== 阶段 3: 高并发 + 中输出 ==="
concurrent_req "Explain machine learning in detail" 200 8
concurrent_req "Explain machine learning in detail" 200 16
sleep 2

# 第 4 阶段：高并发 + 长输出
echo "=== 阶段 4: 高并发 + 长输出 ==="
concurrent_req "Write a comprehensive guide about deep learning" 500 2
sleep 2

# 第 5 阶段：持续高压 - 长输出 + 频繁请求
echo "=== 阶段 5: 持续高压 ==="
for round in 1 2 3 4 5; do
  echo "--- 高压轮次 $round/5 ---"
  for i in $(seq 1 8); do
    curl -s "$API" -H "Content-Type: application/json" \
      -d "{\"model\": \"$MODEL\", \"prompt\": \"Write a long story about a robot $i\", \"max_tokens\": 1000, \"temperature\": 0.1}" \
      -o /dev/null -w "  req#$i: %{http_code} %{time_total}s\n" &
  done
  wait
  echo ""
done

echo ""
echo "========================================="
echo " 压力测试完成"
echo "========================================="
