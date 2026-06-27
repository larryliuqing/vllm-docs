#!/bin/bash
# DeepSeek-V4-Flash-w4a8-mtp 8卡启动脚本 (NPU 0-7, TP=8)
# 模型: DeepSeek-V4-Flash-w4a8-mtp (W4A8量化, MTP)
# 参考: vllm-ascend nightly test config: DeepSeek-V4-Flash-W8A8-A3.yaml
# 测试日期: 2026-06-26

# 设置CANN环境
source /usr/local/Ascend/cann-9.0.0/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

# 设置库路径
export LIBRARY_PATH=/usr/local/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

# HCCL相关配置
export HCCL_EXEC_TIMEOUT=204
export HCCL_CONNECT_TIMEOUT=120
export HCCL_IF_IP=127.0.0.1
export HCCL_INTRA_ROCE_ENABLE=0
export HCCL_INTRA_PCIE_ENABLE=1

# 网络接口配置
export GLOO_SOCKET_IFNAME="lo"
export TP_SOCKET_IFNAME="lo"
export HCCL_SOCKET_IFNAME="lo"

# DeepSeek V4 专用环境变量
export VLLM_ASCEND_APPLY_DSV4_PATCH=1
export USE_MULTI_BLOCK_POOL=1
export USE_MULTI_GROUPS_KV_CACHE=1
export HCCL_BUFFSIZE=1024
export VLLM_ASCEND_ENABLE_FUSED_MC2=0
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1

# 内存分配配置
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export ASCEND_LAUNCH_BLOCKING=0

# 指定使用的NPU设备 (NPU 0-7)
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

MODEL_PATH="/home/la/work/vllm-project/models/DeepSeek-V4-Flash-w4a8-mtp"

echo "========================================="
echo "启动 DeepSeek-V4-Flash-w4a8-mtp (TP=8)"
echo "NPU: 0-7"
echo "========================================="

vllm serve "$MODEL_PATH" \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 8 \
  --data-parallel-size 1 \
  --enable-expert-parallel \
  --api-server-count 1 \
  --max-model-len 8192 \
  --max-num-batched-tokens 10240 \
  --max-num-seqs 64 \
  --gpu-memory-utilization 0.95 \
  --trust-remote-code \
  --tokenizer-mode deepseek_v4 \
  --tool-call-parser deepseek_v4 \
  --enable-auto-tool-choice \
  --reasoning-parser deepseek_v4 \
  --enable-prefix-caching \
  --safetensors-load-strategy prefetch \
  --quantization ascend \
  --block-size 128 \
  --speculative-config '{"num_speculative_tokens": 1, "method": "mtp", "enforce_eager": true}' \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --async-scheduling \
  --additional-config '{"ascend_compilation_config": {"enable_npugraph_ex": true, "enable_static_kernel": false}, "enable_cpu_binding": "true", "enable_shared_expert_dp": true, "multistream_overlap_shared_expert": true, "multistream_dsa_preprocess": false}' \
  --seed 1024
