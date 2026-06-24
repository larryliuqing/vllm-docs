#!/bin/bash

# Qwen2.5-Omni-7B 4卡性能基准测试脚本

CONTAINER_IP=${1:-"172.17.0.2"}
PORT=${2:-"8001"}
BASE_URL="http://${CONTAINER_IP}:${PORT}"

echo "========================================"
echo "Qwen2.5-Omni-7B 4卡性能基准测试"
echo "========================================"
echo "服务地址: ${BASE_URL}"
echo "配置: 4卡 TP, max_model_len=16384"
echo "========================================"
echo ""

# 测试1: 文本生成速度
echo "【1】文本生成速度测试"
echo "----------------------------------------"
START=$(date +%s.%N)
RESPONSE=$(curl -s -X POST ${BASE_URL}/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/models/Qwen/Qwen2___5-Omni-7B",
    "messages": [{"role": "user", "content": "请详细介绍人工智能的发展历史、现状和未来趋势。"}],
    "max_tokens": 500,
    "temperature": 0.7
  }')
END=$(date +%s.%N)

TOKENS=$(echo $RESPONSE | python3 -c "import sys, json; print(json.load(sys.stdin)['usage']['completion_tokens'])")
ELAPSED=$(echo "$END - $START" | bc)
SPEED=$(echo "$TOKENS / $ELAPSED" | bc -l)

echo "生成 Token 数: $TOKENS"
echo "响应时间: $ELAPSED 秒"
echo "生成速度: $SPEED tokens/s"
echo ""

# 测试2: 简单问答速度
echo "【2】简单问答速度测试"
echo "----------------------------------------"
START=$(date +%s.%N)
RESPONSE=$(curl -s -X POST ${BASE_URL}/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/models/Qwen/Qwen2___5-Omni-7B",
    "messages": [{"role": "user", "content": "1+1等于几？"}],
    "max_tokens": 50,
    "temperature": 0.1
  }')
END=$(date +%s.%N)

TOKENS=$(echo $RESPONSE | python3 -c "import sys, json; print(json.load(sys.stdin)['usage']['completion_tokens'])")
ELAPSED=$(echo "$END - $START" | bc)
SPEED=$(echo "$TOKENS / $ELAPSED" | bc -l)

echo "生成 Token 数: $TOKENS"
echo "响应时间: $ELAPSED 秒"
echo "生成速度: $SPEED tokens/s"
echo ""

# 测试3: 多轮对话测试
echo "【3】多轮对话测试"
echo "----------------------------------------"
START=$(date +%s.%N)
RESPONSE=$(curl -s -X POST ${BASE_URL}/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/models/Qwen/Qwen2___5-Omni-7B",
    "messages": [
      {"role": "system", "content": "你是一个友好的助手。"},
      {"role": "user", "content": "你好！"},
      {"role": "assistant", "content": "你好！有什么我可以帮助你的吗？"},
      {"role": "user", "content": "请介绍一下你自己。"}
    ],
    "max_tokens": 100,
    "temperature": 0.7
  }')
END=$(date +%s.%N)

TOKENS=$(echo $RESPONSE | python3 -c "import sys, json; print(json.load(sys.stdin)['usage']['completion_tokens'])")
ELAPSED=$(echo "$END - $START" | bc)
SPEED=$(echo "$TOKENS / $ELAPSED" | bc -l)

echo "生成 Token 数: $TOKENS"
echo "响应时间: $ELAPSED 秒"
echo "生成速度: $SPEED tokens/s"
echo ""

echo "========================================"
echo "测试完成"
echo "========================================"