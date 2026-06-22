# KVTransfer 工作原理详解

## 1. 核心架构

KVTransfer 是 vLLM 中用于分布式 KV cache 传输的核心机制，主要用于以下场景：
- **P/D 分离**（Prefill-Decode Disaggregation）：将预填充和解码分离到不同实例
- **KV Cache 持久化**：跨实例、跨重启保留 KV cache
- **前缀缓存共享**：多个请求共享相同的 KV cache

### 架构组件

```
┌─────────────────────────────────────────────────────────────┐
│                    vLLM Engine                               │
│                                                              │
│  ┌──────────────┐              ┌──────────────────────────┐ │
│  │  Scheduler   │◄────────────►│   KVConnector (Scheduler)│ │
│  │  Process     │              │   - get_num_new_matched_tokens()│
│  │              │              │   - update_state_after_alloc() │
│  │              │              │   - build_connector_meta()     │
│  └──────┬───────┘              └──────────────────────────┘ │
│         │                                                    │
│         │ SchedulerOutput                                     │
│         │ (包含 KVConnectorMetadata)                         │
│         ▼                                                    │
│  ┌──────────────┐              ┌──────────────────────────┐ │
│  │   Worker     │◄────────────►│   KVConnector (Worker)   │ │
│  │   Process    │              │   - start_load_kv()      │ │
│  │              │              │   - wait_for_layer_load()│ │
│  │              │              │   - save_kv_layer()      │ │
│  └──────────────┘              └──────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. 核心角色与职责

### 2.1 Scheduler-side Connector（调度器端连接器）

运行在调度器进程中，负责：

1. **查询可加载的 KV cache 数量**
   ```python
   def get_num_new_matched_tokens(
       request: Request,
       num_computed_tokens: int
   ) -> tuple[int | None, bool]:
       """
       返回：
       - int: 可以从外部加载的 token 数量
       - bool: 是否异步加载
       """
   ```

2. **更新块分配后的状态**
   ```python
   def update_state_after_alloc(
       request: Request,
       blocks: KVCacheBlocks,
       num_external_tokens: int
   ):
       """记录需要加载/保存的请求信息"""
   ```

3. **构建元数据传递给 Worker**
   ```python
   def build_connector_meta(
       scheduler_output: SchedulerOutput
   ) -> KVConnectorMetadata:
       """构建传输所需的元数据"""
   ```

### 2.2 Worker-side Connector（工作器端连接器）

运行在每个 Worker 进程中，负责：

1. **加载 KV cache**
   ```python
   def start_load_kv(forward_context, **kwargs):
       """开始异步加载 KV cache 到 GPU 内存"""

   def wait_for_layer_load(layer_name: str):
       """等待特定层的 KV 加载完成"""
   ```

2. **保存 KV cache**
   ```python
   def save_kv_layer(layer_name, kv_layer, attn_metadata, **kwargs):
       """开始异步保存 KV cache"""

   def wait_for_save():
       """等待所有保存操作完成"""
   ```

---

## 3. 完整工作流程

### 3.1 加载远程 KV cache 流程（P → D）

```
┌─────────────────────────────────────────────────────────────┐
│                   Prefill Instance (Producer)                │
│                                                              │
│  1. 接收请求并执行 prefill                                    │
│  2. 计算 KV cache                                            │
│  3. 调用 save_kv_layer() 保存每一层的 KV                     │
│  4. 调用 wait_for_save() 等待保存完成                        │
│  5. 通过网络/存储传输 KV cache                                │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ KV Cache Transfer
                            │ (通过 Mooncake/NIXL/存储等)
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   Decode Instance (Consumer)                 │
│                                                              │
│  Scheduler 流程：                                            │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ 1. 调度新请求                                         │  │
│  │ 2. get_num_new_matched_tokens()                       │  │
│  │    查询有多少 KV cache 可从远程加载                    │  │
│  │ 3. update_state_after_alloc()                         │  │
│  │    记录请求信息和块分配                                │  │
│  │ 4. build_connector_meta()                             │  │
│  │    构建 KVConnectorMetadata 传递给 Worker             │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  Worker 流程：                                               │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ 1. bind_connector_metadata() 绑定元数据               │  │
│  │ 2. start_load_kv() 开始异步加载                       │  │
│  │ 3. 对每一层：                                          │  │
│  │    - 执行前向传播                                      │  │
│  │    - wait_for_layer_load() 等待该层加载完成           │  │
│  │ 4. wait_for_save() 等待所有保存完成                   │  │
│  │ 5. 清理元数据                                          │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  6. 使用加载的 KV cache 继续解码                             │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 详细的调度器执行流程

```python
# 文件：vllm-ascend/vllm_ascend/core/scheduler_dynamic_batch.py

def _schedule(self):
    for request in self.waiting:
        # 1. 查询本地已计算的 tokens
        num_new_local_computed_tokens = get_local_computed_tokens(request)

        # 2. 查询外部 KV cache 可用性
        if self.connector is not None:
            num_external_tokens, load_async = \
                self.connector.get_num_new_matched_tokens(
                    request, num_new_local_computed_tokens
                )

            if num_external_tokens is None:
                # 连接器需要更多时间确定，跳过该请求
                continue

        # 3. 计算总 tokens
        num_computed_tokens = (
            num_new_local_computed_tokens + num_external_tokens
        )

        # 4. 分配 KV cache 块
        new_blocks = self.kv_cache_manager.allocate_slots(
            request, num_tokens, num_computed_tokens
        )

        # 5. 更新连接器状态
        if self.connector is not None:
            self.connector.update_state_after_alloc(
                request,
                new_blocks,
                num_external_tokens
            )

        # 6. 如果是异步加载，设置特殊状态
        if load_async:
            request.status = RequestStatus.WAITING_FOR_REMOTE_KV
            # 请求将在下一个调度周期被处理
```

### 3.3 Worker 执行流程

```python
# 文件：vllm/vllm/v1/worker/gpu_model_runner.py

def execute_model(self, scheduler_output):
    # 1. 绑定连接器元数据
    if self.connector is not None:
        self.connector.bind_connector_metadata(
            scheduler_output.kv_connector_metadata
        )

    # 2. 开始加载 KV cache
    if self.connector is not None:
        self.connector.start_load_kv(forward_context)

    # 3. 执行模型前向传播（逐层）
    for layer_name, layer in self.model.named_modules():
        # 执行层前向传播
        hidden_states = layer(hidden_states)

        # 如果是注意力层，等待 KV 加载完成
        if has_kv_cache(layer):
            if self.connector is not None:
                self.connector.wait_for_layer_load(layer_name)

    # 4. 等待所有保存操作完成
    if self.connector is not None:
        self.connector.wait_for_save()
        self.connector.clear_connector_metadata()
```

---

## 4. 关键数据结构

### 4.1 KVConnectorMetadata

```python
@dataclass
class ExampleConnectorMetadata(KVConnectorMetadata):
    requests: list[ReqMeta]  # 需要加载/保存的请求列表

@dataclass
class ReqMeta:
    token_ids: torch.Tensor       # 请求的 token IDs
    slot_mapping: torch.Tensor    # GPU 内存槽位映射
    is_store: bool               # True: 保存, False: 加载
    mm_hashes: list[str]         # 多模态哈希（如果有）
```

### 4.2 Request 扩展字段

```python
# 文件：vllm/vllm/v1/request.py

class Request:
    # ... 其他字段 ...

    # KV Transfer 参数
    kv_transfer_params: dict[str, Any] | None = None

    # 已计算的 tokens 数量
    num_computed_tokens: int = 0

    # 请求状态
    status: RequestStatus  # WAITING, RUNNING, WAITING_FOR_REMOTE_KV 等
```

---

## 5. 不同连接器的实现方式

### 5.1 ExampleConnector（文件存储）

```python
# 最简单的实现，用于调试和理解原理
class ExampleConnector:
    def save_kv_layer(self, layer_name, kv_layer, ...):
        # 从 GPU 内存提取 KV cache
        kv_data = extract_kv_from_layer(kv_layer, slot_mapping)
        # 保存到磁盘文件
        safetensors.torch.save_file(
            {"kv_cache": kv_data},
            filename
        )

    def start_load_kv(self, forward_context, ...):
        # 从磁盘文件加载
        kv_data = safetensors.torch.load_file(filename)
        # 注入到 GPU 内存
        inject_kv_into_layer(gpu_kv_cache, kv_data, slot_mapping)
```

### 5.2 MooncakeConnector（网络传输）

```python
# 用于 P/D 分离的网络传输
class MooncakeConnector:
    def save_kv_layer(self, layer_name, kv_layer, ...):
        # 通过 Mooncake 框架发送到远程
        self.mooncake_session.send(
            kv_layer,
            destination_rank=decode_rank
        )

    def start_load_kv(self, forward_context, ...):
        # 从远程接收
        kv_data = self.mooncake_session.recv(
            source_rank=prefill_rank
        )
        # 直接加载到 GPU 内存
```

### 5.3 LMCacheConnector（持久化缓存）

```python
# 集成 LMCache 进行持久化存储
class LMCacheConnector:
    def save_kv_layer(self, layer_name, kv_layer, ...):
        # 存储到 LMCache 后端（内存/CPU/磁盘）
        self.lmcache_engine.store(
            request_id,
            kv_layer,
            layer_name
        )

    def start_load_kv(self, forward_context, ...):
        # 从 LMCache 查询并加载
        kv_data = self.lmcache_engine.retrieve(
            request_id,
            layer_name
        )
```

---

## 6. 性能优化技术

### 6.1 异步加载（Async Loading）

```python
# Scheduler 端
num_external_tokens, load_async = \
    connector.get_num_new_matched_tokens(request, ...)

if load_async:
    # 立即分配内存，但不阻塞调度
    request.status = RequestStatus.WAITING_FOR_REMOTE_KV
    # 下一个调度周期检查是否加载完成
```

**优势：**
- 调度器不被 IO 阻塞
- 可以并行调度其他请求
- 最大化 GPU 利用率

### 6.2 逐层流水线（Layer-wise Pipelining）

```python
# Worker 端
for layer in model.layers:
    # 第 N 层计算时，第 N+1 层 KV 正在后台加载
    if connector:
        connector.wait_for_layer_load(current_layer)  # 确保当前层就绪

    output = layer(input)  # 执行计算

    if connector:
        connector.save_kv_layer(next_layer, ...)  # 触发下一层加载
```

**优势：**
- 计算与 IO 重叠
- 隐藏网络/存储延迟
- 提升整体吞吐量

### 6.3 跨层块传输（Cross-layer Blocks）

```python
class KVConnectorBase_V1:
    @property
    def prefer_cross_layer_blocks(self) -> bool:
        """指示是否使用跨层 KV 块"""
        return True

    def register_cross_layers_kv_cache(self, kv_cache, attn_backend):
        """
        一次性传输所有层的 KV cache
        减少网络往返次数
        """
```

**优势：**
- 减少传输次数
- 降低协议开销
- 适合高延迟网络

---

## 7. 状态机与请求生命周期

```
┌─────────┐
│ WAITING │ ◄─────────────────────┐
└────┬────┘                        │
     │ 调度成功                     │
     ▼                             │
┌──────────────┐                   │
│ RUNNING      │                   │
└────┬─────────┘                   │
     │ 请求完成                     │
     ├─────────────────────────────►│
     │                              │
     │ 需要加载远程 KV              │
     ▼                              │
┌────────────────────┐              │
│WAITING_FOR_REMOTE_KV│             │
└────┬───────────────┘              │
     │ KV 加载完成                   │
     └──────────────────────────────┘
```

---

## 8. 关键配置参数

### 8.1 KVTransferConfig

```python
from vllm.config import KVTransferConfig

ktc = KVTransferConfig(
    kv_connector="MooncakeConnectorV1",  # 连接器类型
    kv_role="kv_producer",               # 角色：producer/consumer/both
    kv_rank=0,                           # 实例 rank
    kv_parallel_size=2,                  # 并行实例数
    kv_buffer_device="npu",              # 缓冲设备
    kv_buffer_size=1e9,                  # 缓冲区大小（字节）
    kv_port=30000,                       # 通信端口
    engine_id="0",                       # 引擎 ID
    kv_connector_extra_config={          # 额外配置
        "prefill": {"dp_size": 2, "tp_size": 2},
        "decode": {"dp_size": 2, "tp_size": 2}
    }
)
```

### 8.2 CacheConfig

```python
from vllm import LLM

llm = LLM(
    model="model_name",
    enable_prefix_caching=True,      # 启用前缀缓存
    block_size=16,                   # KV cache 块大小
    gpu_memory_utilization=0.9,      # GPU 内存利用率
    kv_transfer_config=ktc,          # KV 传输配置
)
```

---

## 9. 故障处理与容错

### 9.1 加载失败处理

```python
# Scheduler 端
num_external_tokens, load_async = \
    connector.get_num_new_matched_tokens(request, ...)

if num_external_tokens is None:
    # 无法确定匹配 tokens，跳过请求
    # 等待下次重试
    continue

# Worker 端
failed_blocks = connector.get_block_ids_with_load_errors()
if failed_blocks:
    # 根据策略处理：recompute 或 fail
    if kv_load_failure_policy == "recompute":
        # 重新计算失败的块
        recompute_blocks(failed_blocks)
    else:
        # 立即失败请求
        raise KVLoadError("Failed to load KV cache")
```

### 9.2 抢占处理

```python
def handle_preemptions(self, kv_connector_metadata):
    """
    处理被抢占的请求或被驱逐的块
    在它们被覆盖之前保存状态
    """
    for preempted_request in preempted_requests:
        # 异步保存 KV cache
        self.save_kv_async(preempted_request)
```

---

## 10. 监控与调试

### 10.1 性能指标

```python
stats = connector.get_kv_connector_stats()
print(f"KV Load Time: {stats.load_time_ms} ms")
print(f"KV Save Time: {stats.save_time_ms} ms")
print(f"Cache Hit Rate: {stats.cache_hit_rate}%")
print(f"Bytes Transferred: {stats.bytes_transferred}")
```

### 10.2 事件追踪

```python
events = connector.take_events()
for event in events:
    print(f"Event: {event.type}, Time: {event.timestamp}")
    print(f"Request: {event.request_id}, Tokens: {event.num_tokens}")
```

---

## 11. 最佳实践

### 11.1 场景选择

| 场景 | 推荐连接器 | 配置建议 |
|------|-----------|---------|
| **单节点 P/D 分离** | MooncakeConnectorV1 | kv_buffer_device="npu", 使用 RDMA |
| **跨节点 P/D 分离** | MooncakeLayerwiseConnector | 启用逐层传输，优化网络拓扑 |
| **持久化前缀缓存** | UCMConnector | 配置 NFS/3FS 后端，设置合理缓存大小 |
| **大规模分布式** | AscendStoreConnector | 使用 FabricMem 模式，启用远荣后端 |
| **调试和开发** | ExampleConnector | 使用本地文件存储 |

### 11.2 性能调优

1. **块大小优化**
   ```python
   # 较大的 block_size 减少碎片，但增加内存占用
   block_size=64  # 适合大模型
   block_size=16  # 适合小模型，减少浪费
   ```

2. **缓冲区大小**
   ```python
   # 根据模型大小和网络带宽调整
   kv_buffer_size=2e9  # 2GB，适合高带宽网络
   kv_buffer_size=512e6  # 512MB，适合低带宽网络
   ```

3. **异步加载**
   ```python
   # 启用异步加载以隐藏延迟
   load_kv_async=True
   ```

---

## 12. 总结

KVTransfer 通过以下核心机制实现高效的分布式 KV cache 管理：

1. **双角色架构**：Scheduler 端负责调度决策，Worker 端负责实际传输
2. **异步操作**：支持异步加载/保存，最大化计算与 IO 重叠
3. **逐层流水线**：在模型执行过程中逐层加载/保存 KV cache
4. **可扩展设计**：通过插件式连接器支持多种传输后端
5. **容错机制**：完善的错误处理和重试机制

这种设计使得 vLLM 能够支持复杂的部署场景，包括 P/D 分离、跨实例 KV cache 共享、持久化缓存等，同时保持高性能和可靠性。
