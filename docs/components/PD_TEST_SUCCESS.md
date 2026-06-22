# PD分离测试 - 成功报告

## 测试时间
2026-06-15 12:36-12:39

## 测试环境
- **方式**: Docker容器（multiprocess模式）
- **镜像**: vllm-ascend:v0.20.2rc
- **设备映射**: 物理NPU 4,5 → 逻辑ID 0,1
- **模型**: /models/Qwen/Qwen3-0.6B（本地模型）

## 关键配置

### 环境变量
```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HCCL_IF_IP=localhost
export HCCL_INTRA_PCIE_ENABLE=1
export HCCL_INTRA_ROCE_ENABLE=0
export VLLM_USE_MODELSCOPE=False
```

### 设备映射
- **容器启动参数**:
  ```bash
  docker run \
    --device=/dev/davinci4 \
    --device=/dev/davinci5 \
    --device=/dev/davinci_manager \
    --device=/dev/devmm_svm \
    --device=/dev/hisi_hdc \
    -e ASCEND_RT_VISIBLE_DEVICES=0,1
  ```

- **映射关系**:
  - 容器内逻辑ID 0 → 物理NPU 4
  - 容器内逻辑ID 1 → 物理NPU 5

### Prefill进程
```python
os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "0"
ktc = KVTransferConfig(
    kv_connector="MooncakeConnectorV1",
    kv_role="kv_producer",
    kv_port="30000",
    engine_id="0",
)
```

### Decode进程
```python
os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "1"
ktc = KVTransferConfig(
    kv_connector="MooncakeConnectorV1",
    kv_role="kv_consumer",
    kv_port="30100",
    engine_id="1",
)
```

## 测试结果

### ✓ 成功要点

1. **NPU设备识别**: 容器内正确识别2个NPU设备
   ```
   Available NPU count: 2
   Device 0: Ascend910B4
   Device 1: Ascend910B4
   ```

2. **Mooncake初始化成功**: TransferEngine正常初始化
   ```
   AscendDirectTransport register mem addr:0x12cc...
   Starting listening on path: tcp://192.168.122.11:30000
   ```

3. **Prefill阶段完成**: 2个prompt处理完成
   ```
   Processed prompts: 100%|██████████| 2/2 [00:11<00:00,  5.65s/it]
   est. speed input: 1.24 toks/s, output: 0.18 toks/s
   ```

4. **Decode阶段完成**: 接收KV cache并生成结果
   ```
   Processed prompts: 100%|██████████| 2/2 [00:16<00:00,  8.16s/it]
   est. speed input: 0.86 toks/s, output: 1.96 toks/s
   ```

5. **生成结果**:
   ```
   Prompt: 'Hello, how are you today?'
   Generated: " I'm sorry for the inconvenience. I'm sorry for the inconvenience. I'm"

   Prompt: 'Hi, what is your name?'
   Generated: " I'm a student in the school of engineering, and I'm interested in studying"
   ```

### 关键技术细节

1. **Eager模式**: 使用`--enforce-eager`避免torch.compile问题
   ```
   WARNING: Enforce eager set, disabling torch.compile and CUDAGraphs
   ```

2. **内存管理**: KV cache分配成功（22.2 GiB）
   ```
   Available KV cache memory: 22.2 GiB
   GPU KV cache size: 202,875 tokens
   Maximum concurrency for 2,000 tokens per request: 101.44x
   ```

3. **CPU绑定**: 自动分配CPU资源给每个NPU
   ```
   NPU0: main=[2 3 4 5 6 7 8 9 10 11 12 13]  acl=[14]
   NPU1: main=[18 19 20 21 22 23 24 25 26 27 28 29]  acl=[30]
   ```

## 与之前失败的对比

### 官方示例失败原因
- 使用modelscope模型：`deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`
- 该模型触发PyTorch Dynamo编译
- Dynamo需要`vllm_ascend_C`模块（容器未编译）
- 导致ImportError和编译失败

### 本次成功原因
- 使用本地模型：`/models/Qwen/Qwen3-0.6B`
- 使用`--enforce-eager`禁用编译
- 避开了自定义ops模块缺失问题
- Mooncake/ADXL正常初始化和通信

## 结论

### ✓ PD分离在容器multiprocess模式下完全可行！

**关键发现**:
1. 设备ID映射正确（逻辑ID 0,1 对应物理NPU 4,5）
2. Mooncake TransferEngine初始化成功
3. KV cache传输工作正常
4. Prefill和Decode进程成功协调

**注意事项**:
1. 必须使用`--enforce-eager`避免编译问题
2. 必须使用本地模型或确保容器有完整的编译ops
3. 必须设置正确的环境变量（HCCL_IF_IP=localhost等）
4. 必须使用spawn方式启动multiprocessing

## 下一步

1. ✓ **已验证**: 容器multiprocess PD分离可行
2. **可扩展**: 测试更复杂的模型（如DeepSeek-V2）
3. **可优化**: 启用torch.compile提升性能（需编译ops）
4. **可生产化**: 创建生产级部署方案（多节点、负载均衡等）