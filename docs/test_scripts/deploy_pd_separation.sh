#!/bin/bash
# PD分离部署脚本 - 节点内测试
# 测试日期: 2026-06-24
# 测试环境: 192.168.0.190
# 镜像版本: vllm-ascend:v0.20.2rc

set -e

echo "========================================="
echo "PD分离部署脚本 - 节点内测试"
echo "========================================="

# 配置参数
IMAGE="vllm-ascend:v0.20.2rc"
MODEL_PATH="/root/models/Qwen2-VL-7B-Instruct"
SCRIPTS_DIR="/root"

# 停止并删除旧容器
echo "[1/6] 清理旧容器..."
docker stop vllm-p vllm-d 2>/dev/null || true
docker rm vllm-p vllm-d 2>/dev/null || true

# 启动Prefill节点
echo "[2/6] 启动Prefill节点 (NPU 0-1)..."
docker run -d \
  --name vllm-p \
  --network host \
  --privileged \
  --device=/dev/davinci0 \
  --device=/dev/davinci1 \
  --device=/dev/davinci_manager \
  --device=/dev/devmm_svm \
  --device=/dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
  -v /root/models:/root/models \
  -v ${SCRIPTS_DIR}/run_prefill.sh:/root/run_prefill.sh \
  ${IMAGE} \
  bash -c 'chmod +x /root/run_prefill.sh && /root/run_prefill.sh'

echo "等待Prefill节点启动..."
sleep 10

# 启动Decode节点
echo "[3/6] 启动Decode节点 (NPU 4-5)..."
docker run -d \
  --name vllm-d \
  --network host \
  --privileged \
  --device=/dev/davinci4 \
  --device=/dev/davinci5 \
  --device=/dev/davinci_manager \
  --device=/dev/devmm_svm \
  --device=/dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
  -v /root/models:/root/models \
  -v ${SCRIPTS_DIR}/run_decode.sh:/root/run_decode.sh \
  ${IMAGE} \
  bash -c 'chmod +x /root/run_decode.sh && /root/run_decode.sh'

echo "等待Decode节点启动..."
sleep 10

# 检查容器状态
echo "[4/6] 检查容器状态..."
docker ps | grep vllm-

# 等待服务启动
echo "[5/6] 等待服务启动完成 (约60秒)..."
for i in {1..60}; do
  if docker logs vllm-p 2>&1 | grep -q "Application startup complete"; then
    if docker logs vllm-d 2>&1 | grep -q "Application startup complete"; then
      echo "✓ 服务启动完成"
      break
    fi
  fi
  echo -n "."
  sleep 1
done
echo ""

# 启动Proxy服务器
echo "[6/6] 启动Proxy服务器..."
docker exec -d vllm-d python \
  /vllm-workspace/vllm-ascend/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py \
  --host 127.0.0.1 \
  --prefiller-hosts 127.0.0.1 \
  --prefiller-ports 8100 \
  --decoder-hosts 127.0.0.1 \
  --decoder-ports 8200

sleep 3

echo ""
echo "========================================="
echo "✓ PD分离部署完成"
echo "========================================="
echo ""
echo "服务信息:"
echo "  - Prefill节点: http://127.0.0.1:8100"
echo "  - Decode节点:  http://127.0.0.1:8200"
echo "  - Proxy服务器: http://127.0.0.1:8000"
echo ""
echo "测试命令:"
echo "  curl -s http://127.0.0.1:8000/v1/completions \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{"
echo "      \"model\": \"/root/models/Qwen2-VL-7B-Instruct\","
echo "      \"prompt\": \"Hello, how are you?\","
echo "      \"max_tokens\": 20"
echo "    }'"
echo ""
