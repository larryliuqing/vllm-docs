#!/bin/bash

# Qwen2.5-Omni-7B 功能测试脚本
# 使用方法: bash test_omni.sh [container_ip] [port]

set -e

CONTAINER_IP=${1:-$(docker inspect $(docker ps -q --filter "ancestor=vllm-omni:v0.20.2rc" | head -1) 2>/dev/null | grep '"IPAddress"' | head -1 | awk -F'"' '{print $4}')}
PORT=${2:-8001}
BASE_URL="http://${CONTAINER_IP}:${PORT}"

echo "========================================"
echo "Qwen2.5-Omni-7B 功能测试"
echo "========================================"
echo "服务地址: ${BASE_URL}"
echo "========================================"
echo ""

# 测试1: 文本对话
echo "【测试1】文本对话"
echo "----------------------------------------"
curl -s -X POST ${BASE_URL}/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/models/Qwen/Qwen2___5-Omni-7B",
    "messages": [{"role": "user", "content": "你好，请用一句话介绍你自己。"}],
    "max_tokens": 100,
    "temperature": 0.7
  }' | python3 -c "import sys, json; data=json.load(sys.stdin); print('响应:', data['choices'][0]['message']['content']); print('Token:', data['usage'])"
echo ""

# 测试2: 图像理解（小图）
echo "【测试2】图像理解（128x128）"
echo "----------------------------------------"
curl -s -X POST ${BASE_URL}/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/models/Qwen/Qwen2___5-Omni-7B",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "请描述这张图片。"},
        {"type": "image_url", "image_url": {"url": "https://picsum.photos/seed/omni/128/128"}}
      ]
    }],
    "max_tokens": 200
  }' | python3 -c "import sys, json; data=json.load(sys.stdin); print('响应:', data['choices'][0]['message']['content']); print('Token:', data['usage'])"
echo ""

# 测试3: 图像理解（中图）
echo "【测试3】图像理解（256x256）"
echo "----------------------------------------"
curl -s -X POST ${BASE_URL}/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/models/Qwen/Qwen2___5-Omni-7B",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "请描述这张图片的内容。"},
        {"type": "image_url", "image_url": {"url": "https://picsum.photos/seed/test/256/256"}}
      ]
    }],
    "max_tokens": 200
  }' | python3 -c "import sys, json; data=json.load(sys.stdin); print('响应:', data['choices'][0]['message']['content']); print('Token:', data['usage'])"
echo ""

# 测试4: 多轮对话
echo "【测试4】多轮对话"
echo "----------------------------------------"
curl -s -X POST ${BASE_URL}/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/models/Qwen/Qwen2___5-Omni-7B",
    "messages": [
      {"role": "system", "content": "你是一个友好的助手。"},
      {"role": "user", "content": "你好！"},
      {"role": "assistant", "content": "你好！有什么我可以帮助你的吗？"},
      {"role": "user", "content": "请介绍一下人工智能。"}
    ],
    "max_tokens": 100
  }' | python3 -c "import sys, json; data=json.load(sys.stdin); print('响应:', data['choices'][0]['message']['content']); print('Token:', data['usage'])"
echo ""

# 测试5: 流式输出
echo "【测试5】流式输出"
echo "----------------------------------------"
curl -s -X POST ${BASE_URL}/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/models/Qwen/Qwen2___5-Omni-7B",
    "messages": [{"role": "user", "content": "从1数到3"}],
    "max_tokens": 20,
    "stream": true
  }' | grep -o '"content":"[^"]*"' | sed 's/"content":"//g' | sed 's/"//g' | tr -d '\n'
echo ""
echo ""

echo "========================================"
echo "测试完成"
echo "========================================"
