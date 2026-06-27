#!/bin/bash
# Prefill节点启动脚本 - 使用MooncakeConnector (使用NPU 0-3，TP=4)
# 测试日期: 2026-06-24
# 测试配置: 4+4卡 (Prefill: NPU 0-3, Decode: NPU 4-7)
# 镜像版本: vllm-ascend:v0.20.2rc

# 设置CANN环境
source /usr/local/Ascend/cann-9.0.0/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

# 设置库路径（解决libtransfer_engine.so找不到的问题）
export LIBRARY_PATH=/usr/local/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

# HCCL相关配置
export HCCL_EXEC_TIMEOUT=204
export HCCL_CONNECT_TIMEOUT=120
export HCCL_IF_IP=127.0.0.1
export HCCL_INTRA_ROCE_ENABLE=0
export HCCL_INTRA_PCIE_ENABLE=1

# 网络接口配置（同节点内使用本地回环）
export GLOO_SOCKET_IFNAME="lo"
export TP_SOCKET_IFNAME="lo"
export HCCL_SOCKET_IFNAME="lo"

# 指定使用的NPU设备（NPU 0-3）
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export PHYSICAL_DEVICES=$(ls /dev/davinci* 2>/dev/null | grep -o '[0-9]\+' | sort -n | paste -sd',' -)

echo "========================================="
echo "启动Prefill节点 (NPU 0-3, TP=4)"
echo "========================================="
echo "ASCEND_RT_VISIBLE_DEVICES: $ASCEND_RT_VISIBLE_DEVICES"

# 启动vllm Prefill服务
vllm serve "/home/la/work/vllm-project/models/Qwen/Qwen3-VL-32B-Instruct" \
  --host 127.0.0.1 \
  --port 8100 \
  --tensor-parallel-size 4 \
  --seed 1024 \
  --max-model-len 2000 \
  --max-num-batched-tokens 2000 \
  --trust-remote-code \
  --gpu-memory-utilization 0.8 \
  --kv-transfer-config \
  '{"kv_connector": "MooncakeConnectorV1",
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
      "tp_size": 4
    }
  }
  }'