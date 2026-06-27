# PD分离测试报告 - Qwen3-VL-32B大模型

**测试日期**: 2026-06-24
**测试环境**: 192.168.0.190
**镜像版本**: vllm-ascend:v0.20.2rc
**测试模型**: Qwen3-VL-32B-Instruct (63GB, 64层, 5120隐藏层)

---

## 1. 测试环境

### 1.1 硬件配置

| 项目 | 配置 |
|------|------|
| **服务器类型** | Atlas 训练服务器 |
| **NPU 数量** | 8x 910B4 |
| **NPU 状态** | 全部健康 (OK) |
| **HBM 内存** | 32GB per NPU |

### 1.2 软件环境

| 组件 | 版本 |
|------|------|
| **操作系统** | Linux |
| **Driver** | 25.5.2 |
| **CANN** | 9.0.0 (容器内) |
| **vLLM** | 0.20.2 |
| **vLLM-Ascend** | v0.20.2rc |

---

## 2. PD分离配置

### 2.1 架构说明

本次测试采用**节点内PD分离**方案（4+4卡配置）：

- **Prefill节点 (vllm-p)**:
  - 使用NPU 0-3
  - Tensor Parallel Size = 4
  - Port: 8100
  - KV Port: 20001
  - Role: kv_producer

- **Decode节点 (vllm-d)**:
  - 使用NPU 4-7
  - Tensor Parallel Size = 4
  - Port: 8200
  - KV Port: 20002
  - Role: kv_consumer

- **Proxy服务器**:
  - Port: 8000
  - 协调Prefill和Decode节点

### 2.2 模型配置

**Qwen3-VL-32B-Instruct参数**:
- 总大小: 63GB (14个safetensors文件)
- Hidden Size: 5120
- Intermediate Size: 25600
- Attention Heads: 64
- Hidden Layers: 64

### 2.3 关键配置

**环境变量**:
```bash
# CANN环境
source /usr/local/Ascend/cann-9.0.0/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

# 库路径
export LIBRARY_PATH=/usr/local/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

# HCCL配置
export HCCL_EXEC_TIMEOUT=204
export HCCL_CONNECT_TIMEOUT=120
export HCCL_IF_IP=127.0.0.1
export HCCL_INTRA_ROCE_ENABLE=0
export HCCL_INTRA_PCIE_ENABLE=1
```

**KV Transfer配置**:
```json
{
  "kv_connector": "MooncakeConnectorV1",
  "kv_buffer_device": "npu",
  "kv_role": "kv_producer",  // Prefill节点
  "kv_parallel_size": 1,
  "kv_port": "20001",
  "engine_id": "0",
  "kv_rank": 0,
  "kv_connector_extra_config": {
    "prefill": {"dp_size": 1, "tp_size": 4},
    "decode": {"dp_size": 1, "tp_size": 4}
  }
}
```

---

## 3. 测试过程

### 3.1 部署步骤

1. **清理旧容器**: 停止并删除vllm-p和vllm-d容器
2. **启动Prefill节点**: NPU 0-3, TP=4, 约3分钟加载模型
3. **启动Decode节点**: NPU 4-7, TP=4, 约2分钟加载模型
4. **启动Proxy服务器**: 在Decode容器内启动负载均衡代理
5. **测试推理**: 通过Proxy发送请求验证PD分离功能

### 3.2 启动时间

| 节点 | 启动时间 | 状态 |
|------|----------|------|
| Prefill节点 | ~3分钟 | ✅ Application startup complete |
| Decode节点 | ~2分钟 | ✅ Application startup complete |
| Proxy服务器 | <5秒 | ✅ Running |

### 3.3 测试结果

**第一次测试请求**:
```bash
curl -s http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "/home/la/work/vllm-project/models/Qwen/Qwen3-VL-32B-Instruct",
    "prompt": "Hello, how are you?",
    "max_tokens": 20
  }'
```

**响应结果**:
```json
{
  "id": "cmpl-208d2d56-a075-4177-a503-3d1caef17fc0",
  "object": "text_completion",
  "model": "/home/la/work/vllm-project/models/Qwen/Qwen3-VL-32B-Instruct",
  "choices": [{
    "index": 0,
    "text": " I am having some trouble with my code. I am trying to create a function that takes in a",
    "finish_reason": "length"
  }],
  "usage": {
    "prompt_tokens": 6,
    "total_tokens": 26,
    "completion_tokens": 20
  }
}
```

**第二次测试请求**:
```bash
curl -s http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "/home/la/work/vllm-project/models/Qwen/Qwen3-VL-32B-Instruct",
    "prompt": "What is artificial intelligence?",
    "max_tokens": 30
  }'
```

**响应结果**:
```json
{
  "id": "cmpl-fd532dcd-3455-486d-85c9-97e1b5b3debd",
  "object": "text_completion",
  "model": "/home/la/work/vllm-project/models/Qwen/Qwen3-VL-32B-Instruct",
  "choices": [{
    "index": 0,
    "text": " Artificial intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think, learn, and make decisions like humans. These systems",
    "finish_reason": "length"
  }],
  "usage": {
    "prompt_tokens": 5,
    "total_tokens": 35,
    "completion_tokens": 30
  }
}
```

---

## 4. 关键发现

### 4.1 成功要点

✅ **大模型支持**: 成功加载63GB的Qwen3-VL-32B模型
✅ **4+4卡配置**: Prefill (TP=4) + Decode (TP=4) 正常运行
✅ **KV传输**: MooncakeConnector正常建立KV缓存传输
✅ **推理功能**: PD分离推理请求成功返回正确结果

### 4.2 与2+2卡测试对比

| 对比项 | 2+2卡测试 (Qwen2-VL-7B) | 4+4卡测试 (Qwen3-VL-32B) |
|--------|------------------------|--------------------------|
| 模型大小 | 7B (~15GB) | 32B (63GB) |
| NPU使用 | 4卡 (0-1, 4-5) | 8卡 (0-3, 4-7) |
| TP配置 | TP=2 | TP=4 |
| 加载时间 | ~3秒/worker | ~3分钟 (Prefill), ~2分钟 (Decode) |
| 推理成功 | ✅ | ✅ |
| KV传输 | MooncakeConnector | MooncakeConnector |

---

## 5. 测试结论

### 5.1 成功验证

✅ **PD分离架构稳定性**: 大模型（32B参数）PD分离部署成功
✅ **多NPU协同**: 8卡同时运行，Prefill和Decode节点独立工作
✅ **KV缓存传输**: MooncakeConnector在TP=4配置下正常工作
✅ **负载均衡**: Proxy服务器正确路由请求

### 5.2 性能建议

基于测试结果，建议后续优化方向：

1. **KV传输优化**:
   - 32B模型KV缓存更大，传输延迟可能成为瓶颈
   - 建议启用层优先级传输或压缩传输

2. **内存管理**:
   - 大模型内存占用高，建议监控HBM使用率
   - 可能需要调整`gpu_memory_utilization`参数

3. **异构TP配置**:
   - 可测试Prefill TP=8, Decode TP=2的异构配置
   - 使用`pd_head_ratio`特性优化资源利用率

---

## 6. 附录

### 6.1 启动命令

**Prefill节点**:
```bash
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
  vllm-ascend:v0.20.2rc \
  bash -c 'chmod +x /root/run_prefill.sh && /root/run_prefill.sh'
```

**Decode节点**:
```bash
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
  vllm-ascend:v0.20.2rc \
  bash -c 'chmod +x /root/run_decode.sh && /root/run_decode.sh'
```

**Proxy服务器**:
```bash
docker exec -d vllm-d python \
  /vllm-workspace/vllm-ascend/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py \
  --host 127.0.0.1 \
  --prefiller-hosts 127.0.0.1 \
  --prefiller-ports 8100 \
  --decoder-hosts 127.0.0.1 \
  --decoder-ports 8200
```

### 6.2 相关文件

- 启动脚本:
  - `/root/run_prefill_4cards.sh`
  - `/root/run_decode_4cards.sh`
  - `/root/deploy_pd_separation_4cards.sh`
- 模型路径: `/home/la/work/vllm-project/models/Qwen/Qwen3-VL-32B-Instruct`
- 测试文档:
  - `vllm-docs/docs/analysis/PD分离优化分析报告.md`
  - `vllm-docs/docs/analysis/PD分离源码深度分析.md`

---

**测试人员**: Claude
**测试状态**: ✅ 成功
**下一步**: 性能基准测试，多节点PD分离测试
