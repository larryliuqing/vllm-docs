#!/bin/bash
# DeepSeek V4 快速性能测试
API="http://127.0.0.1:8000/v1/completions"
MODEL="/home/la/work/vllm-project/models/DeepSeek-V4-Flash-w4a8-mtp"

echo "=== DeepSeek V4 性能测试 ==="
echo ""

# 1. 短文本生成 (10 tok 输出)
echo "--- 短文本生成 (max_tokens=10) ---"
for i in 1 2 3; do
  time curl -s "$API" -H "Content-Type: application/json" \
    -d "{\"model\": \"$MODEL\", \"prompt\": \"Hello, my name is\", \"max_tokens\": 10, \"temperature\": 0.1}" \
    -o /dev/null -w "请求 $i: 总耗时 %{time_total}s\n"
done

echo ""
echo "--- 中文本生成 (max_tokens=100) ---"
for i in 1 2 3; do
  time curl -s "$API" -H "Content-Type: application/json" \
    -d "{\"model\": \"$MODEL\", \"prompt\": \"Explain the theory of relativity in simple terms:\", \"max_tokens\": 100, \"temperature\": 0.1}" \
    -o /dev/null -w "请求 $i: 总耗时 %{time_total}s\n"
done

echo ""
echo "--- 长文本生成 (max_tokens=500) ---"
time curl -s "$API" -H "Content-Type: application/json" \
  -d "{\"model\": \"$MODEL\", \"prompt\": \"Write a comprehensive guide about artificial intelligence:\", \"max_tokens\": 500, \"temperature\": 0.1}" \
  -o /dev/null -w "请求 1: 总耗时 %{time_total}s\n"

echo ""
echo "--- 并发测试 (4并发, 50 tok) ---"
for i in $(seq 1 4); do
  curl -s "$API" -H "Content-Type: application/json" \
    -d "{\"model\": \"$MODEL\", \"prompt\": \"What is machine learning?\", \"max_tokens\": 50, \"temperature\": 0.1}" \
    -o /dev/null &
done
wait
echo "4并发请求完成"
