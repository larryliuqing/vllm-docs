#!/bin/bash
# Prefill节点启动脚本 - 异构TP配置 (NPU 4-7, TP=4)
# Decode节点使用 NPU 2-3 (TP=2)
# 模型: Qwen2-VL-7B-Instruct
# 说明: Prefill TP=4, Decode TP=2, pd_head_ratio=4/2=2

source /usr/local/Ascend/cann-9.0.0/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

export LIBRARY_PATH=/usr/local/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

export HCCL_EXEC_TIMEOUT=204
export HCCL_CONNECT_TIMEOUT=120
export HCCL_IF_IP=127.0.0.1
export HCCL_INTRA_ROCE_ENABLE=0
export HCCL_INTRA_PCIE_ENABLE=1

export GLOO_SOCKET_IFNAME="lo"
export TP_SOCKET_IFNAME="lo"
export HCCL_SOCKET_IFNAME="lo"

export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7

echo "========================================="
echo "启动Prefill节点 (NPU 4-7, TP=4)"
echo "========================================="

# 使用MooncakeLayerwiseConnector支持异构TP (Prefill TP=4, Decode TP=2)
# pd_head_ratio 由 ascend_config.py 自动从 extra_config 计算
vllm serve "/home/la/work/vllm-project/models/Qwen/Qwen2-VL-7B-Instruct" \
  --host 127.0.0.1 \
  --port 8100 \
  --tensor-parallel-size 4 \
  --seed 1024 \
  --max-model-len 4096 \
  --max-num-batched-tokens 4096 \
  --trust-remote-code \
  --gpu-memory-utilization 0.85 \
  --kv-transfer-config \
  '{"kv_connector": "MooncakeLayerwiseConnector",
  "kv_buffer_device": "npu",
  "kv_role": "kv_producer",
  "kv_parallel_size": 1,
  "kv_port": "20001",
  "engine_id": "0",
  "kv_rank": 0,
  "kv_connector_extra_config": {
    "prefill": {
      "dp_size": 1,
      "tp_size": 4
    },
    "decode": {
      "dp_size": 1,
      "tp_size": 2
    }
  }
  }'