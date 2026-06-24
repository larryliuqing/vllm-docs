# PD分离测试报告

**测试日期**: 2026-06-24
**测试环境**: 192.168.0.190
**镜像版本**: vllm-ascend:v0.20.2rc
**测试模型**: Qwen2-VL-7B-Instruct

---

## 1. 测试环境

### 1.1 硬件配置

| 项目 | 配置 |
|------|------|
| **服务器类型** | Atlas 训练服务器 |
| **NPU 数量** | 8x 910B4 |
| **NPU 状态** | 全部健康 (OK) |
| **HCCS 状态** | OK (链路正常) |
| **RoCE 状态** | UP (光口已连接) |

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

本次测试采用**节点内PD分离**方案（场景1）：

- **Prefill节点 (vllm-p)**:
  - 使用NPU 0-1
  - Tensor Parallel Size = 2
  - Port: 8100
  - KV Port: 20001
  - Role: kv_producer

- **Decode节点 (vllm-d)**:
  - 使用NPU 4-5
  - Tensor Parallel Size = 2
  - Port: 8200
  - KV Port: 20002
  - Role: kv_consumer

- **Proxy服务器**:
  - Port: 8000
  - 协调Prefill和Decode节点

### 2.2 关键配置

**环境变量**:
```bash
# CANN环境
source /usr/local/Ascend/cann-9.0.0/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

# 库路径（解决libtransfer_engine.so找不到的问题）
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
    "prefill": {"dp_size": 1, "tp_size": 2},
    "decode": {"dp_size": 1, "tp_size": 2}
  }
}
```

---

## 3. 测试过程

### 3.1 遇到的问题与解决

#### 问题1: Mooncake库找不到

**错误信息**:
```
libtransfer_engine.so: cannot open shared object file: No such file or directory
```

**原因**: 容器内未设置正确的库路径

**解决方案**:
```bash
export LIBRARY_PATH=/usr/local/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
```

#### 问题2: CANN环境路径错误

**错误信息**:
```
/usr/local/Ascend/ascend-toolkit/set_env.sh: No such file or directory
```

**原因**: 容器内CANN路径与宿主机不同

**解决方案**: 使用容器内正确的CANN路径
```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh
```

#### 问题3: DP Size配置冲突

**错误信息**:
```
KV transfer 'prefill' config has a conflicting data parallel size. Expected 1, but got 2
```

**原因**: kv_connector_extra_config中的dp_size与实际配置不匹配

**解决方案**: 确保kv_connector_extra_config中的配置与实际的TP/DP配置一致
```json
{
  "prefill": {"dp_size": 1, "tp_size": 2},
  "decode": {"dp_size": 1, "tp_size": 2}
}
```

### 3.2 成功启动的服务

| 服务 | 状态 | 端口 | 说明 |
|------|------|------|------|
| vllm-p (Prefill) | ✅ Running | 8100 | Mooncake监听在 192.168.0.190:20001/20002 |
| vllm-d (Decode) | ✅ Running | 8200 | 应用启动完成 |
| Proxy Server | ✅ Running | 8000 | 负载均衡代理 |

### 3.3 测试结果

**测试请求**:
```bash
curl -s http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "/root/models/Qwen2-VL-7B-Instruct",
    "prompt": "Hello, how are you?",
    "max_tokens": 20
  }'
```

**响应结果**:
```json
{
  "id": "cmpl-8407ce6d-f25d-40d7-b9ed-2e80ec19f1cd",
  "object": "text_completion",
  "created": 1782272008,
  "model": "/root/models/Qwen2-VL-7B-Instruct",
  "choices": [{
    "index": 0,
    "text": " I'm just a robot, so I don't have feelings. But I'm here to help you",
    "finish_reason": "length"
  }],
  "usage": {
    "prompt_tokens": 6,
    "total_tokens": 26,
    "completion_tokens": 20
  }
}
```

**结论**: ✅ PD分离功能正常工作，推理请求成功返回结果

---

## 4. 关键发现

### 4.1 必需的配置项

1. **CANN环境变量**: 必须正确设置CANN和ATB的环境变量
2. **库路径**: 必须添加`/usr/local/lib`到`LD_LIBRARY_PATH`
3. **HCCL配置**: 同节点内通信需要设置`HCCL_INTRA_ROCE_ENABLE=0`和`HCCL_INTRA_PCIE_ENABLE=1`
4. **KV Transfer配置**: dp_size和tp_size必须与实际配置匹配

### 4.2 性能指标

- **模型加载时间**: ~3秒 (每个Worker)
- **模型权重大小**: 7.76 GB (每个TP rank)
- **预热时间**: ~7.7秒

---

## 5. 测试结论

### 5.1 成功要点

✅ 节点内PD分离功能正常运行
✅ MooncakeConnector成功建立连接
✅ KV Cache传输机制工作正常
✅ Prefill和Decode节点成功协同工作

### 5.2 建议

1. **文档改进**: 建议在部署文档中明确说明需要设置`LD_LIBRARY_PATH`
2. **配置验证**: 建议添加配置验证机制，自动检测dp_size/tp_size是否匹配
3. **错误提示**: 建议改进错误提示信息，明确指出缺少的库文件路径

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
  --device=/dev/davinci_manager \
  --device=/dev/devmm_svm \
  --device=/dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
  -v /root/models:/root/models \
  -v /root/run_prefill_final.sh:/root/run_prefill_final.sh \
  vllm-ascend:v0.20.2rc \
  bash -c 'chmod +x /root/run_prefill_final.sh && /root/run_prefill_final.sh'
```

**Decode节点**:
```bash
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
  -v /root/run_decode_final.sh:/root/run_decode_final.sh \
  vllm-ascend:v0.20.2rc \
  bash -c 'chmod +x /root/run_decode_final.sh && /root/run_decode_final.sh'
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

- 启动脚本: `/root/run_prefill_final.sh`, `/root/run_decode_final.sh`
- 测试文档: `vllm-docs/docs/components/PD_Separation_Test_Scenarios.md`
- 示例代码: `vllm-ascend/examples/disaggregated_prefill_v1/`

---

**测试人员**: Claude
**审核状态**: 待审核
**下一步**: 进行性能测试和多节点PD分离测试
