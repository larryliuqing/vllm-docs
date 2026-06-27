#!/bin/bash
# PD分离同构TP部署脚本 - Prefill(NPU 4-5, TP=2) + Decode(NPU 6-7, TP=2)
# 测试模型: Qwen2-VL-7B-Instruct
# KV Connector: MooncakeConnectorV1 (同构TP)
# 测试日期: 2026-06-25

set -e

echo "============================================="
echo "PD分离同构TP部署脚本 (2+2卡)"
echo "配置: Prefill (NPU 4-5, TP=2) + Decode (NPU 6-7, TP=2)"
echo "模型: Qwen2-VL-7B-Instruct"
echo "============================================="

# 配置参数
IMAGE="vllm-ascend:v0.20.2rc"
MODEL_PATH="/home/la/work/vllm-project/models/Qwen/Qwen2-VL-7B-Instruct"
SCRIPTS_DIR="/home/la/work/vllm-project/vllm-docs/docs/test_scripts"

# 停止并删除旧容器
echo "[1/7] 清理旧容器..."
docker stop vllm-p vllm-d 2>/dev/null || true
docker rm vllm-p vllm-d 2>/dev/null || true

# 检查NPU状态
echo "[2/7] 检查NPU状态..."
npu-smi info | grep -E "NPU|Health|OK"

# 启动Prefill节点 (NPU 4-5, TP=2)
echo "[3/7] 启动Prefill节点 (NPU 4-5, TP=2)..."
docker run -d \
  --name vllm-p \
  --network host \
  --privileged \
  --device=/dev/davinci4 \
  --device=/dev/davinci5 \
  --device=/dev/davinci_manager \
  --device=/dev/devmm_svm \
  --device=/dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
  -v /home/la/work/vllm-project/models:/home/la/work/vllm-project/models \
  -v ${SCRIPTS_DIR}/run_prefill_2cards.sh:/root/run_prefill.sh \
  ${IMAGE} \
  bash -c 'chmod +x /root/run_prefill.sh && /root/run_prefill.sh'

echo "等待Prefill节点启动 (30秒)..."
sleep 30

# 启动Decode节点 (NPU 6-7, TP=2)
echo "[4/7] 启动Decode节点 (NPU 6-7, TP=2)..."
docker run -d \
  --name vllm-d \
  --network host \
  --privileged \
  --device=/dev/davinci6 \
  --device=/dev/davinci7 \
  --device=/dev/davinci_manager \
  --device=/dev/devmm_svm \
  --device=/dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
  -v /home/la/work/vllm-project/models:/home/la/work/vllm-project/models \
  -v ${SCRIPTS_DIR}/run_decode_2cards.sh:/root/run_decode.sh \
  ${IMAGE} \
  bash -c 'chmod +x /root/run_decode.sh && /root/run_decode.sh'

echo "等待Decode节点启动 (30秒)..."
sleep 30

# 检查容器状态
echo "[5/7] 检查容器状态..."
docker ps | grep vllm-

# 等待服务启动完成（最长150秒）
echo "[6/7] 等待服务启动完成..."
for i in {1..150}; do
  if docker logs vllm-p 2>&1 | grep -q "Application startup complete" && \
     docker logs vllm-d 2>&1 | grep -q "Application startup complete"; then
    echo "✓ Prefill和Decode服务启动完成"
    break
  fi
  if [ $i -eq 150 ]; then
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

# 预热请求 - 消除冷启动影响
echo "发送预热请求..."
WARMUP_MODEL_PATH="/home/la/work/vllm-project/models/Qwen/Qwen2-VL-7B-Instruct"
for i in 1 2 3; do
  curl -s -X POST http://127.0.0.1:8000/v1/completions \
    -H 'Content-Type: application/json' \
    -d "{
      \"model\": \"${WARMUP_MODEL_PATH}\",
      \"prompt\": \"Hello, how are you?\",
      \"max_tokens\": 5,
      \"temperature\": 0.0
    }" > /dev/null
  echo "  预热请求 #${i} 完成"
  sleep 1
done
echo "预热完成"

echo ""
echo "============================================="
echo "✓ PD分离同构TP部署完成"
echo "============================================="
echo ""
echo "服务信息:"
echo "  - Prefill节点: http://127.0.0.1:8100 (NPU 4-5, TP=2)"
echo "  - Decode节点:  http://127.0.0.1:8200 (NPU 6-7, TP=2)"
echo "  - Proxy服务器: http://127.0.0.1:8000"
echo "  - KV Connector: MooncakeConnectorV1"
echo ""
echo "查看日志:"
echo "  docker logs vllm-p  # Prefill节点日志"
echo "  docker logs vllm-d  # Decode节点日志"
echo ""