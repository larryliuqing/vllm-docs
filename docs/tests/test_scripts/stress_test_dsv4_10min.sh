#!/bin/bash
# DeepSeek V4 10分钟持续压力测试
API="http://127.0.0.1:8000/v1/completions"
MODEL="/home/la/work/vllm-project/models/DeepSeek-V4-Flash-w4a8-mtp"
ENDTIME=$((SECONDS + 600))  # 10分钟

echo "========================================="
echo " DeepSeek V4 10分钟持续压力测试"
echo " 开始时间: $(date +%H:%M:%S)"
echo " 结束时间: $(date -d @$(($(date +%s) + 600)) +%H:%M:%S)"
echo "========================================="
echo ""

total_req=0
error_req=0
fail_req=0

while [ $SECONDS -lt $ENDTIME ]; do
  round_start=$SECONDS

  # 每轮并发数: 8~16 随机 (max-num-seqs=64 的安全范围内)
  concurrency=$((RANDOM % 9 + 8))
  max_tokens=$((RANDOM % 501 + 500))  # 500~1000 tok

  echo "[$(date +%H:%M:%S)] 轮次开始: 并发=$concurrency, out=$max_tokens"

  # 并行发送
  for i in $(seq 1 $concurrency); do
    (
      resp=$(curl -s -w "\n%{http_code}" "$API" -H "Content-Type: application/json" \
        -d "{\"model\": \"$MODEL\", \"prompt\": \"Write a detailed technical explanation about deep learning architecture and optimization $i\", \"max_tokens\": $max_tokens, \"temperature\": 0.1}")
      http_code=$(echo "$resp" | tail -1)
      echo "  req#$i: $http_code"
    ) &
  done
  wait

  # 统计
  round_elapsed=$((SECONDS - round_start))
  total_req=$((total_req + concurrency))

  echo "  本轮耗时: ${round_elapsed}s | 累计请求: $total_req"
  echo ""

  # 简短间隔避免雪崩
  sleep 2
done

echo "========================================="
echo " 压力测试结束"
echo " 结束时间: $(date +%H:%M:%S)"
echo " 总请求数: $total_req"
echo "========================================="