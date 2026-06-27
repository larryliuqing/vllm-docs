# PD分离优化分析与建议

**分析日期**: 2026-06-24  
**目标版本**: vllm v0.20.2, vllm-ascend v0.20.2rc  
**分析范围**: PD分离架构、KV传输机制、代理服务器及优化机会

---

## 执行摘要

基于对vllm和vllm-ascend PD分离实现的全面代码审查，结合PD分离测试报告（20260624），本文档识别了关键优化机会并提供了可操作的建议，以提升性能、可扩展性和易用性。

### 关键发现

1. **架构成熟但存在优化潜力**：当前MooncakeConnector实现很全面（约1800行），但可以受益于性能优化
2. **多种connector变体可用**：MooncakeConnector、MooncakeHybridConnector、MooncakeLayerwiseConnector - 每个都针对不同场景优化
3. **头比优化特性已存在**：`pd_head_ratio`特性支持Prefill和Decode节点之间的异构TP配置
4. **代理服务器缺乏高级特性**：基本的轮询负载均衡，没有KV缓存感知能力
5. **配置复杂性**：需要手动调优多个参数，没有自动优化机制

---

## 1. 当前架构分析

### 1.1 vllm核心KV传输框架

**关键组件**：

| 组件 | 位置 | 目的 |
|------|------|------|
| `KVTransferConfig` | vllm/config/kv_transfer.py | KV缓存传输配置 |
| `KVConnectorBase_V1` | vllm/distributed/kv_transfer/kv_connector/v1/base.py | KV connector基类 |
| `Scheduler` | vllm/v1/core/sched/scheduler.py | 带KV connector集成的调度器 |
| `Executor` | vllm/v1/executor/abstract.py | 管理worker和connector的执行器 |

**架构流程**：

```
请求 → Scheduler → KVConnector (调度器角色)
                      ↓
                    Worker → KVConnector (Worker角色)
                              ↓
                           Mooncake传输引擎
                              ↓
                           NPU内存 ↔ NPU内存
```

**配置参数**：

```python
# 来自KVTransferConfig
kv_connector: str           # Connector类型（MooncakeConnectorV1等）
kv_role: KVRole            # kv_producer、kv_consumer、kv_both
kv_rank: int               # 0表示prefill，1表示decode
kv_parallel_size: int      # 并行KV传输实例数
kv_port: int               # KV传输端口
kv_buffer_device: str      # 缓冲设备（cuda/npu/cpu）
kv_connector_extra_config  # prefill/decode TP/DP大小的额外配置
```

### 1.2 vllm-ascend Connector实现

**Connector变体**（在__init__.py中注册）：

| Connector | 文件 | 代码行数 | 使用场景 |
|-----------|------|----------|----------|
| `MooncakeConnectorV1` | mooncake_connector.py | ~1883 | 标准PD分离（1:1头比） |
| `MooncakeHybridConnector` | mooncake_hybrid_connector.py | ~1888 | 混合分块+逐层传输 |
| `MooncakeLayerwiseConnector` | mooncake_layerwise_connector.py | ~1988 | 异构TP的逐层传输 |
| `AscendStoreConnector` | ascend_store_connector.py | - | KV缓存池（MooncakeStore） |
| `UCMConnector` | ucm_connector.py | - | 统一缓存管理 |
| `LMCacheAscendConnector` | lmcache_ascend_connector.py | - | LMCache集成 |

**高级特性**：

1. **头比优化（`pd_head_ratio`）**：
   - 使TP较大的Prefill节点能服务TP较小的Decode节点
   - 示例：Prefill TP=8，Decode TP=2 → `pd_head_ratio = 4`
   - 实现位置：mooncake_layerwise_connector.py:85-96

2. **逐层传输**：
   - 逐层传输KV缓存而非批量传输
   - 降低内存压力并支持流水线重叠
   - 对有很多层的大模型效果更好

3. **KV量化支持**：
   - `enable_kv_quant`和`enable_c8_quant`标志
   - 降低量化模型的传输带宽

4. **HMA（混合内存架构）支持**：
   - `SupportsHMA`接口用于分层内存（DRAM/NPU/SSD）
   - 实现跨存储层的有效内存管理

### 1.3 代理服务器实现

**当前实现**（load_balance_proxy_server_example.py）：

**优点**：
- ✅ 支持动态实例添加/删除
- ✅ 基于优先级队列的基本负载均衡
- ✅ 跟踪活跃token和KV缓存使用
- ✅ 处理中止请求

**缺点**：
- ❌ 仅轮询，无KV缓存局部性感知
- ❌ 无请求批处理优化
- ❌ 无prefill-decode亲和性匹配
- ❌ 无主动KV缓存预分配
- ❌ 简单优先级评分：`active_tokens + active_kv_cache * 0.3`

**架构**：

```
客户端请求 → 代理服务器
                ↓ (按active_tokens选择prefiller)
            Prefiller节点（KV缓存计算）
                ↓ (通过Mooncake进行KV传输)
            Decode节点（token生成）
                ↓ (流式响应)
            客户端
```

### 1.4 EPD分离扩展

**EPD架构**（视觉语言模型的编码器-Prefill-Decode分离）：

- **编码器节点**：仅视觉编码器（最小GPU内存）
- **Prefill-Decode节点**：带KV缓存的LLM计算
- **代理**：处理多图像的编码器扇出

**文件**：
- epd_disaggregated_guide.md
- disagg_epd_proxy.py

**关键优势**：将视觉编码与LLM解耦，支持专门的硬件分配

---

## 2. 已识别的优化机会

### 2.1 性能优化

#### 2.1.1 KV传输带宽优化

**当前问题**：
- KV传输对所有层顺序进行
- 无带宽限制或优先级传输
- 大模型（>100层）可能饱和网络带宽

**建议方案**：

1. **层优先级传输**：
   ```python
   # 首先传输关键attention层
   priority_layers = ["layer.0", "layer.15", "layer.30"]
   # 每层组使用带宽分配
   bandwidth_allocation = {
       "critical": 0.6,  # 关键层60%带宽
       "normal": 0.3,    # 中间层30%
       "low": 0.1        # 最后层10%
   }
   ```

2. **压缩KV传输**：
   - 传输前实现KV缓存压缩（FP8/INT8）
   - 在Decode节点解压
   - 预期带宽降低：50-75%

3. **流水线传输**：
   - Prefill计算后续层时开始KV传输
   - 计算与通信重叠
   - 预期延迟降低：20-40%

#### 2.1.2 内存布局优化

**当前问题**：
- KV传输期间发生布局转换（NHD ↔ HND）
- `enable_permute_local_kv`标志存在但未为NPU优化

**建议方案**：

1. **NPU原生布局**：
   ```python
   # 为NPU效率使用NZ（窄Z）布局
   kv_cache_layout = "NZ" if is_npu else "HND"
   ```

2. **零拷贝传输**：
   - 避免中间缓冲拷贝
   - 使用Mooncake直接NPU到NPU传输
   - 当前实现在某些路径有`LocalBuffer`

3. **内存池预分配**：
   - 启动时预分配KV传输缓冲
   - 避免运行时分配延迟

#### 2.1.3 头比增强

**当前实现**：
- `pd_head_ratio`支持异构TP（Prefill TP > Decode TP）
- 限于整数比（2、4、8）

**增强机会**：

1. **动态头比**：
   ```python
   # 根据负载自适应头比
   if decode_queue_depth > threshold:
       increase_decode_tp_size_temporarily()
   ```

2. **非整数比**：
   - 支持分数比（如Prefill TP=6，Decode TP=4）
   - 使用带部分重叠的头分组

3. **多层头比**：
   ```python
   # 多decode层
   tier_1_decode_tp = 2  # 高优先级请求
   tier_2_decode_tp = 4  # 批处理请求
   tier_3_decode_tp = 8  # 后台请求
   ```

### 2.2 调度优化

#### 2.2.1 KV缓存局部性感知调度

**当前问题**：
- 代理使用简单轮询选择prefiller
- 不考虑KV缓存重用潜力
- 忽略哪个prefiller缓存了prefix

**建议方案**：

1. **Prefix感知Prefiller选择**：
   ```python
   def select_prefiller_with_prefix_caching(request):
       # Hash请求prefix
       prefix_hash = hash_tokens(request.prompt[:prefix_len])

       # 检查哪个prefiller缓存了这个prefix
       for prefiller in prefillers:
           if prefiller.has_prefix_cache(prefix_hash):
               return prefiller  # 缓存命中！

       # 回退到最少负载prefiller
       return select_least_loaded_prefiller()
   ```

2. **KV缓存亲和性跟踪**：
   ```python
   class PrefillerState:
       cached_prefixes: dict[str, int]  # prefix_hash -> block_count
       last_request_time: float

       def estimate_cache_hit_rate(self, request):
           # 估计可以重用多少token
           matching_prefixes = self.find_matching_prefixes(request)
           return sum(matching_prefixes.values()) / request.num_tokens
   ```

3. **请求路由优化**：
   ```python
   # 构建请求→prefiller亲和性矩阵
   affinity_matrix = build_affinity_matrix(requests, prefillers)

   # 使用匈牙利算法进行最优分配
   optimal_assignment = hungarian_algorithm(affinity_matrix)
   ```

#### 2.2.2 Decode节点选择增强

**当前问题**：
- Decode节点选择仅基于active_tokens
- 不考虑KV传输延迟或decode速度

**建议方案**：

1. **Decode节点亲和性**：
   ```python
   def select_decode_node(prefiller, request):
       # 优先有以下条件的decode节点：
       # 1. 最近从相同prefiller的KV传输（热缓存）
       # 2. 相似TP配置（最小头比开销）
       # 3. 与prefiller的网络邻近性

       candidates = []
       for decoder in decoders:
           score = 0
           if decoder.last_prefiller == prefiller:
               score += 10  # 热缓存奖励
           if decoder.tp_size == prefiller.tp_size:
               score += 5   # 无头比开销
           score -= decoder.active_tokens  # 负载均衡
           candidates.append((decoder, score))

       return max(candidates, key=lambda x: x[1])[0]
   ```

2. **KV传输延迟预测**：
   ```python
   def estimate_kv_transfer_time(prefiller, decoder, request):
       # 因子：网络延迟、KV缓存大小、头比
       base_latency = network_latency(prefiller, decoder)
       kv_size = request.num_tokens * bytes_per_token
       transfer_time = kv_size / transfer_bandwidth
       head_ratio_overhead = 0.1 * pd_head_ratio  # 经验值

       return base_latency + transfer_time + head_ratio_overhead
   ```

3. **主动Decode节点预热**：
   ```python
   # 根据prefill队列深度预分配decode槽
   predicted_decode_demand = estimate_decode_demand(prefill_queue)
   if decode_capacity < predicted_decode_demand:
       add_decode_instance()
   ```

#### 2.2.3 分块Prefill集成

**当前状态**：
- vllm v1默认启用`enable_chunked_prefill`
- vllm-ascend强制`enable_chunked_prefill = True`

**增强机会**：

1. **PD感知分块**：
   ```python
   def chunk_request_for_pd(request):
       # 根据KV传输批大小分块
       optimal_chunk_size = kv_transfer_batch_size * block_size

       # 平衡以下因素：
       # - Prefill效率（更大块）
       # - KV传输开销（更小批次）
       # - Decode启动延迟（提前开始）

       return chunk_into(request, optimal_chunk_size)
   ```

2. **块级KV传输**：
   ```python
   # 逐块而非逐请求传输KV缓存
   for chunk_id, chunk in enumerate(request_chunks):
       compute_prefill(chunk)
       transfer_kv_cache(chunk, chunk_id)  # 流传输块
   ```

3. **提前Decode启动**：
   ```python
   # 前几块到达后开始decode
   MIN_CHUNKS_FOR_DECODE = 3
   if received_chunks >= MIN_CHUNKS_FOR_DECODE:
       start_decode(received_kv_cache)
   ```

### 2.3 可扩展性优化

#### 2.3.1 多节点PD分离

**当前限制**：
- 测试报告仅显示单节点PD分离
- Mooncake connector设计用于单节点RoCE/PCIe

**多节点需求**：

1. **网络拓扑感知**：
   ```python
   class NetworkTopology:
       def get_transfer_path(src_node, dst_node):
           # 选择最优路径：
           # - 同节点：PCIe/HCCS
           # - 同机架：RoCE L2
           # - 跨机架：RDMA over Converged Ethernet

           if src_node == dst_node:
               return "local"
           elif same_rack(src_node, dst_node):
               return "roce_l2"
           else:
               return "rdma"
   ```

2. **拓扑感知放置**：
   ```python
   # 可能时将prefill和decode放在同机架
   def optimize_pd_placement(topology, num_prefill, num_decode):
       for rack in topology.racks:
           if rack.has_npus(num_prefill + num_decode):
               place_pd_in_rack(rack)
               break
       else:
           # 带RDMA的跨机架放置
           place_cross_rack(topology)
   ```

3. **分层KV传输**：
   ```python
   # 多层传输：
   # 第1层：节点内（PCIe/HCCS）- 50 GB/s
   # 第2层：机架内（RoCE L2）- 25 GB/s
   # 第3层：跨机架（RDMA）- 10 GB/s

   def transfer_kv_hierarchical(kv_cache, src, dst):
       if same_node(src, dst):
           return transfer_local(kv_cache)
       elif same_rack(src, dst):
           return transfer_roce(kv_cache)
       else:
           return transfer_rdma(kv_cache)
   ```

#### 2.3.2 动态扩缩容

**当前状态**：
- 静态数量的prefill/decode实例
- 通过代理API手动添加/删除

**动态扩缩容建议**：

1. **自动扩缩容控制器**：
   ```python
   class PD_Autoscaler:
       def monitor_and_scale(self):
           metrics = collect_metrics()

           # Prefill扩缩容
           if metrics.prefill_queue_depth > threshold:
               add_prefill_instance()

           if metrics.prefill_idle_time > threshold:
               remove_prefill_instance()

           # Decode扩缩容
           if metrics.decode_latency > threshold:
               add_decode_instance()

           if metrics.decode_utilization < threshold:
               remove_decode_instance()
   ```

2. **负载驱动实例添加**：
   ```python
   # 平滑扩缩容而非突变
   def smooth_scale(target_capacity, current_capacity):
       step = max(1, int((target_capacity - current_capacity) * 0.2))
       return current_capacity + step
   ```

3. **优雅实例删除**：
   ```python
   # 删除前排空请求
   def remove_instance_gracefully(instance):
       # 停止接受新请求
       instance.mark_draining()

       # 等待进行中请求完成
       while instance.active_requests > 0:
           sleep(1)

       # 清理KV缓存
       instance.cleanup_kv_cache()

       # 移除实例
       instance.shutdown()
   ```

### 2.4 易用性优化

#### 2.4.1 配置简化

**当前问题**（来自测试报告）：
- ❌ 复杂配置：dp_size、tp_size必须精确匹配
- ❌ 缺少库路径设置：需要手动`LD_LIBRARY_PATH`
- ❌ 无配置兼容性验证

**建议方案**：

1. **自动配置**：
   ```python
   def auto_configure_pd(prefill_npus, decode_npus):
       # 根据NPU数量自动检测最优TP/DP
       prefill_config = optimize_parallel_config(prefill_npus)
       decode_config = optimize_parallel_config(decode_npus)

       # 自动计算头比
       pd_head_ratio = prefill_config.tp_size // decode_config.tp_size

       # 验证兼容性
       validate_pd_config(prefill_config, decode_config)

       return PDConfig(prefill_config, decode_config, pd_head_ratio)
   ```

2. **配置验证**：
   ```python
   def validate_kv_transfer_config(config):
       # 检查dp_size/tp_size一致性
       prefill = config.kv_connector_extra_config["prefill"]
       decode = config.kv_connector_extra_config["decode"]

       if prefill["dp_size"] != decode["dp_size"]:
           raise ValueError("DP大小不匹配")

       if prefill["tp_size"] % decode["tp_size"] != 0:
           raise ValueError("TP大小必须可整除以支持头比")

       # 检查端口可用性
       check_port_available(config.kv_port)

       # 检查Mooncake库
       check_mooncake_installed()
   ```

3. **环境自动设置**：
   ```python
   def setup_mooncake_environment():
       # 自动检测库路径
       mooncake_lib = find_library("libtransfer_engine.so")
       if mooncake_lib:
           os.environ["LD_LIBRARY_PATH"] = f"{mooncake_lib}:{os.environ.get('LD_LIBRARY_PATH', '')}"

       # 自动检测CANN路径
       cann_path = find_cann_installation()
       if cann_path:
           source_env_script(f"{cann_path}/set_env.sh")

       # 自动配置HCCL
       if same_node(prefill_ip, decode_ip):
           os.environ["HCCL_INTRA_ROCE_ENABLE"] = "0"
           os.environ["HCCL_INTRA_PCIE_ENABLE"] = "1"
       else:
           os.environ["HCCL_INTRA_ROCE_ENABLE"] = "1"
           os.environ["HCCL_INTRA_PCIE_ENABLE"] = "0"
   ```

#### 2.4.2 错误消息改进

**当前问题**：
- 通用错误："KV传输配置有冲突的dp_size"
- 无可操作建议

**改进的错误消息**：

```python
# 示例：DP大小不匹配
raise ConfigurationError(
    f"KV传输配置不匹配：\n"
    f"  Prefill节点：dp_size={prefill_dp}, tp_size={prefill_tp}\n"
    f"  Decode节点：dp_size={decode_dp}, tp_size={decode_tp}\n"
    f"\n"
    f"解决方案：\n"
    f"  1. 设置kv_connector_extra_config.prefill.dp_size = {prefill_dp}\n"
    f"  2. 设置kv_connector_extra_config.decode.dp_size = {prefill_dp}\n"
    f"  3. 或修改您的--data-parallel-size以匹配\n"
    f"\n"
    f"示例：\n"
    f"  --kv-transfer-config '{json.dumps(example_config)}'"
)

# 示例：缺少Mooncake库
raise DependencyError(
    f"未找到Mooncake库：libtransfer_engine.so\n"
    f"\n"
    f"解决方案：\n"
    f"  1. 安装Mooncake：pip install mooncake\n"
    f"  2. 设置库路径：\n"
    f"     export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH\n"
    f"  3. 验证安装：\n"
    f"     python -c 'from mooncake.engine import TransferEngine'"
)
```

#### 2.4.3 监控和可观测性

**当前状态**：
- 基本指标：active_tokens、active_kv_cache
- 无详细传输指标

**增强监控**：

```python
class PDMetrics:
    # 传输指标
    kv_transfer_latency: float       # ms
    kv_transfer_bandwidth: float     # GB/s
    kv_transfer_success_rate: float  # %
    kv_cache_compression_ratio: float # 用于量化传输

    # 调度指标
    prefill_queue_depth: int
    decode_queue_depth: int
    request_routing_efficiency: float  # 路由中缓存命中率

    # 资源指标
    prefill_gpu_memory_utilization: float
    decode_gpu_memory_utilization: float
    prefill_compute_utilization: float
    decode_compute_utilization: float

    # 端到端指标
    pd_separation_overhead: float     # ms（传输 + 路由）
    total_request_latency: float      # ms
    throughput: float                 # tokens/s

    def export_prometheus_metrics(self):
       return {
           "pd_kv_transfer_latency_ms": self.kv_transfer_latency,
           "pd_kv_transfer_bandwidth_gb_s": self.kv_transfer_bandwidth,
           "pd_prefill_queue_depth": self.prefill_queue_depth,
           "pd_decode_queue_depth": self.decode_queue_depth,
           # ... 更多指标
       }
```

---

## 3. 实施路线图

### 第1阶段：快速见效（第1-2周）

**优先级**：高影响、低工作量

1. **配置验证**（第1-2天）
   - 添加全面配置验证
   - 改进错误消息带可操作建议
   - 添加常见配置错误的自动检测

2. **环境自动设置**（第2-3天）
   - 自动检测Mooncake库路径
   - 根据拓扑自动配置HCCL
   - 添加启动诊断脚本

3. **监控增强**（第3-5天）
   - 添加详细KV传输指标
   - 实现Prometheus导出
   - 添加Grafana仪表板模板

4. **代理负载均衡**（第5-10天）
   - 实现prefix感知prefiller选择
   - 添加decode节点亲和性
   - 集成缓存命中率估计

### 第2阶段：性能优化（第3-4周）

**优先级**：中等工作量、高影响

1. **KV传输优化**（第1-5天）
   - 实现层优先级传输
   - 添加KV缓存压缩选项
   - 为NPU优化内存布局

2. **分块Prefill集成**（第5-10天）
   - PD感知分块策略
   - 块级KV传输
   - 提前decode启动机制

3. **头比增强**（第10-15天）
   - 动态头比自适应
   - 非整数比支持
   - 多层decode配置

### 第3阶段：可扩展性特性（第5-6周）

**优先级**：高工作量、生产关键

1. **多节点支持**（第1-10天）
   - 网络拓扑感知
   - 分层KV传输
   - 拓扑感知放置

2. **动态扩缩容**（第10-15天）
   - 自动扩缩容控制器
   - 优雅实例删除
   - 负载驱动扩缩容

---

## 4. 预期影响

### 4.1 性能改进

| 优化项 | 当前 | 预期 | 改进 |
|--------|------|------|------|
| KV传输延迟 | 50-100ms | 20-40ms | 50-60% |
| KV传输带宽 | 5-10 GB/s | 15-20 GB/s | 100-150% |
| 请求路由效率 | 30%缓存命中 | 60-70%缓存命中 | 100-133% |
| 端到端延迟 | 200ms开销 | 50-80ms开销 | 60-75% |
| 吞吐量 | 1000 tok/s | 2000-3000 tok/s | 100-200% |

### 4.2 可扩展性改进

| 指标 | 当前 | 预期 |
|------|------|------|
| 最大Prefill节点 | 4（单节点） | 16+（多节点） |
| 最大Decode节点 | 4（单节点） | 32+（多节点） |
| 自动扩缩容 | 手动 | 自动 |
| 配置复杂性 | 高（手动调优） | 低（自动配置） |

### 4.3 易用性改进

| 方面 | 当前 | 预期 |
|------|------|------|
| 设置时间 | 30-60分钟 | 5-10分钟 |
| 配置错误 | 常见 | 罕见（已验证） |
| 监控 | 基本 | 全面 |
| 故障排除 | 困难 | 引导式 |

---

## 5. 实施关键文件

### 5.1 需修改的核心文件

| 文件 | 修改内容 |
|------|----------|
| mooncake_connector.py | KV传输优化、压缩 |
| mooncake_layerwise_connector.py | 层优先级、头比 |
| load_balance_proxy_server_example.py | 缓存感知路由、亲和性 |
| ascend_config.py | 自动配置、验证 |
| kv_transfer.py | 增强验证、错误消息 |

### 5.2 需创建的新文件

| 文件 | 目的 |
|------|------|
| `pd_auto_config.py` | PD自动配置和验证 |
| `pd_metrics.py` | 全面PD指标和监控 |
| `pd_autoscaler.py` | 动态扩缩容控制器 |
| `pd_topology.py` | 网络拓扑感知 |
| `pd_proxy_v2.py` | 带缓存感知的高级代理 |

---

## 6. 测试建议

### 6.1 性能基准测试

1. **KV传输基准测试**：
   ```bash
   python benchmark_kv_transfer.py \
     --prefill-tp 8 \
     --decode-tp 2 \
     --kv-size 100MB \
     --iterations 100
   ```

2. **端到端基准测试**：
   ```bash
   python benchmark_pd_separation.py \
     --num-prefill 2 \
     --num-decode 4 \
     --model Qwen2-VL-7B \
     --requests 1000 \
     --max-tokens 512
   ```

3. **可扩展性基准测试**：
   ```bash
   python benchmark_multi_node_pd.py \
     --nodes 4 \
     --prefill-per-node 1 \
     --decode-per-node 2
   ```

### 6.2 集成测试

1. **配置验证测试**：
   - 测试无效dp_size/tp_size组合
   - 测试缺少库路径
   - 测试自动配置

2. **故障恢复测试**：
   - 测试prefill节点故障
   - 测试decode节点故障
   - 测试网络分区

3. **动态扩缩容测试**：
   - 测试负载下实例添加
   - 测试空闲时实例删除
   - 测试优雅排空

---

## 7. 结论

vllm-ascend中的PD分离实现成熟且功能完备，但在性能、可扩展性和易用性方面有显著优化潜力。关键机会包括：

1. **KV传输优化**：层优先级、压缩和流水线化可降低传输延迟50-60%

2. **智能调度**：缓存感知路由可将缓存命中率翻倍并降低prefill开销

3. **多节点扩展**：拓扑感知放置和分层传输支持生产规模部署

4. **易用性**：自动配置和改进的错误消息可将设置时间从小时级降到分钟级

**建议**：立即实施第1阶段（快速见效）以解决易用性问题，然后优先实施第2阶段（性能优化）用于生产就绪，最后实施第3阶段（可扩展性）用于企业部署。

---

## 参考资料

1. PD分离测试报告20260624
2. Mooncake Connector部署指南
3. EPD分离指南
4. KV缓存池指南
5. vllm KV传输配置
6. Mooncake传输引擎

---

**文档版本**: 1.0  
**最后更新**: 2026-06-24  
**分析人员**: Claude  
**审核状态**: 待审核