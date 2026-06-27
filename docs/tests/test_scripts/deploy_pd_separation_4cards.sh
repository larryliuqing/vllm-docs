#!/bin/bash
# PD分离部署脚本 - 4+4卡节点内测试
# 测试日期: 2026-06-24
# 测试配置: Prefill (NPU 0-3, TP=4) + Decode (NPU 4-7, TP=4)
# 镜像版本: vllm-ascend:v0.20.2rc

set -e

echo "========================================="
echo "PD分离部署脚本 - 4+4卡测试"
echo "========================================="
echo "配置: Prefill (NPU 0-3, TP=4) + Decode (NPU 4-7, TP=4)"
echo "========================================="

# 配置参数
IMAGE="vllm-ascend:v0.20.2rc"
MODEL_PATH="/home/la/work/vllm-project/models/Qwen/Qwen3-VL-32B-Instruct"
SCRIPTS_DIR="/root"

# 停止并删除旧容器
echo "[1/6] 清理旧容器..."
docker stop vllm-p vllm-d 2>/dev/null || true
docker rm vllm-p vllm-d 2>/dev/null || true

# 检查NPU状态
echo "[2/6] 检查NPU状态..."
npu-smi info | grep -E "NPU|Health|OK"

# 启动Prefill节点 (NPU 0-3)
echo "[3/6] 启动Prefill节点 (NPU 0-3, TP=4)..."
docker run -d \
  --name vllm-p \
  --network host \
  --privileged \
  --device=/dev/davinci0 \
  --device=/dev/davinci1 \
  --device=/dev/davinci2 \
  --device=/dev/davinci3 \
  --device=/dev/davinci_manager \
  --device=/dev/devmm_svm \
  --device=/dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
  -v /home/la/work/vllm-project/models:/home/la/work/vllm-project/models \
  -v /root/run_prefill_4cards.sh:/root/run_prefill.sh \
  ${IMAGE} \
  bash -c 'chmod +x /root/run_prefill.sh && /root/run_prefill.sh'

echo "等待Prefill节点启动 (30秒)..."
sleep 30

# 启动Decode节点 (NPU 4-7)
echo "[4/6] 启动Decode节点 (NPU 4-7, TP=4)..."
docker run -d \
  --name vllm-d \
  --network host \
  --privileged \
  --device=/dev/davinci4 \
  --device=/dev/davinci5 \
  --device=/dev/davinci6 \
  --device=/dev/davinci7 \
  --device=/dev/davinci_manager \
  --device=/dev/devmm_svm \
  --device=/dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
  -v /home/la/work/vllm-project/models:/home/la/work/vllm-project/models \
  -v /root/run_decode_4cards.sh:/root/run_decode.sh \
  ${IMAGE} \
  bash -c 'chmod +x /root/run_decode.sh && /root/run_decode.sh'

echo "等待Decode节点启动 (30秒)..."
sleep 30

# 检查容器状态
echo "[5/6] 检查容器状态..."
docker ps | grep vllm-

# 等待服务启动完成（最长90秒）
echo "[6/6] 等待服务启动完成..."
for i in {1..90}; do
  prefill_ready=$(docker logs vllm-p 2>&1 | grep -c "Application startup complete" || echo "0")
  decode_ready=$(docker logs vllm-d 2>&1 | grep -c "Application startup complete" || echo "0")

  if [ "$prefill_ready" -ge 1 ] && [ "$decode_ready" -ge 1 ]; then
    echo "✓ Prefill和Decode服务启动完成"
    break
  fi

  if [ $i -eq 90 ]; then
    echo "⚠ 服务启动超时，但继续部署..."
  fi

  echo -n "."
  sleep 1
done
echo ""

# 启动Proxy服务器
echo "启动Proxy服务器..."
docker exec -d vllm-d python \
  /vllm-workspace/vllm-ascend/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py \
  --host 127.0.0.1 \
  --prefiller-hosts 127.0.0.1 \
  --prefiller-ports 8100 \
  --decoder-hosts 127.0.0.1 \
  --decoder-ports 8200

sleep 5

echo ""
echo "========================================="
echo "✓ PD分离部署完成 - 4+4卡配置"
echo "========================================="
echo ""
echo "服务信息:"
echo "  - Prefill节点: http://127.0.0.1:8100 (NPU 0-3, TP=4)"
echo "  - Decode节点:  http://127.0.0.1:8200 (NPU 4-7, TP=4)"
echo "  - Proxy服务器: http://127.0.0.1:8000"
echo ""
echo "测试命令:"
echo "  curl -s http://127.0.0.1:8000/v1/completions \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{"
echo "      \"model\": \"/home/la/work/vllm-project/models/Qwen/Qwen3-VL-32B-Instruct\","
echo "      \"prompt\": \"Hello, how are you?\","
echo "      \"max_tokens\": 20"
echo "    }'"
echo ""
echo "查看日志:"
echo "  docker logs vllm-p  # Prefill节点日志"
echo "  docker logs vllm-d  # Decode节点日志"
echo ""