# PD分离（Prefill-Decode Disaggregation）技术文档

## 概述

PD分离（Prefill-Decode Disaggregation）是一种将大语言模型推理过程中的Prefill阶段和Decode阶段分离到不同实例（或设备）上执行的技术。本文档详细介绍PD分离在vLLM、vLLM-Ascend、Mooncake和HIXL项目中的实现方案。

## 1. PD分离背景与动机

### 1.1 传统推理架构的问题

**Prefill和Decode的特性差异：**

| 特性 | Prefill阶段 | Decode阶段 |
|-----|------------|-----------|
| 计算模式 | 计算密集型 | 访存密集型 |
| Token数量 | 大量（整个Prompt） | 少量（每次1个） |
| 延迟特征 | 高延迟、可预测 | 低延迟、迭代式 |
| 内存访问 | 大量KV Cache写入 | 少量KV Cache读写 |
| GPU利用率 | 高（计算bound） | 低（访存bound） |

**混合执行的痛点：**

```
传统架构（Prefill + Decode 混合）:
┌─────────────────────────────────┐
│       单个GPU实例               │
│  ┌─────────┐   ┌───────────┐  │
│  │ Prefill │ → │  Decode   │  │
│  │ (高负载) │   │ (低负载)  │  │
│  └─────────┘   └───────────┘  │
│        ↓             ↓        │
│    资源争抢、相互阻塞          │
└─────────────────────────────────┘

问题：
1. Prefill阻塞Decode → TTFT（首Token延迟）增加
2. Decode占用资源 → Prefill吞吐量降低
3. 资源利用率低 → 无法针对性优化
```

### 1.2 PD分离的优势

```
PD分离架构:
┌─────────────┐         KV Cache Transfer        ┌─────────────┐
│   Prefill   │ ─────────────────────────────→  │   Decode    │
│  Instance   │    (Mooncake / HIXL传输)        │  Instance   │
│ (计算优化)  │                                  │ (访存优化)  │
└─────────────┘                                  └─────────────┘

优势：
1. 独立优化 → Prefill实例专注计算，Decode实例专注访存
2. 资源隔离 → 互不干扰，各自达到最优利用率
3. 灵活扩展 → 可独立扩容Prefill或Decode实例
4. 降低延迟 → Prefill不阻塞Decode，TTFT显著降低
```

## 2. PD分离架构总览

### 2.1 系统架构图

```
┌──────────────────────────────────────────────────────────────────┐
│                        客户端请求                                 │
└────────────────────────┬─────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────────┐
│                     负载均衡层 / 代理层                           │
│            (Load Balance Proxy / Disaggregated Proxy)            │
└────────────┬─────────────────────────────┬───────────────────────┘
             │                             │
             │ (Prefill请求)               │ (Decode请求)
             ▼                             ▼
┌────────────────────────┐      ┌────────────────────────┐
│   Prefill Instance     │      │   Decode Instance      │
│  ┌──────────────────┐  │      │  ┌──────────────────┐  │
│  │  vLLM /          │  │      │  │  vLLM /          │  │
│  │  vLLM-Ascend     │  │      │  │  vLLM-Ascend     │  │
│  │  (kv_producer)   │  │      │  │  (kv_consumer)   │  │
│  └──────────────────┘  │      │  └──────────────────┘  │
│           │            │      │            ▲           │
│           ▼            │      │            │           │
│  ┌──────────────────┐  │      │  ┌──────────────────┐  │
│  │  KV Transfer     │  │      │  │  KV Transfer     │  │
│  │  Connector       │  │      │  │  Connector       │  │
│  │  (Mooncake/      │──┼──────┼─→│  (Mooncake/      │  │
│  │   HIXL/NIXL)     │  │ KV   │  │   HIXL/NIXL)     │  │
│  └──────────────────┘  │Cache │  └──────────────────┘  │
│                        │Transfer                      │
└────────────────────────┘      └────────────────────────┘
```

### 2.2 组件职责

| 组件 | 项目 | 职责 |
|-----|------|------|
| **Prefill Instance** | vLLM / vLLM-Ascend | 处理Prompt，生成KV Cache |
| **Decode Instance** | vLLM / vLLM-Ascend | 基于KV Cache生成Token |
| **KV Transfer Connector** | vLLM | KV Cache传输抽象层 |
| **Transfer Engine** | Mooncake / HIXL | 底层传输引擎 |
| **Load Balance Proxy** | vLLM-Ascend | 请求路由、负载均衡 |

## 3. vLLM中的PD分离实现

### 3.1 KV Transfer配置

**配置文件：** `vllm/config/kv_transfer.py`

```python
@config
class KVTransferConfig:
    """KV Cache传输配置"""

    kv_connector: str | None = None
    """KV连接器类型：
    - 'MooncakeConnector': 使用Mooncake传输
    - 'NIXLConnector': 使用NIXL传输
    - 'P2pNcclConnector': 使用NCCL P2P传输
    - 'LMCacheConnector': 使用LMCache
    """

    kv_role: KVRole | None = None
    """实例角色：
    - 'kv_producer': Prefill实例，生产KV Cache
    - 'kv_consumer': Decode实例，消费KV Cache
    - 'kv_both': 混合模式，既生产也消费
    """

    kv_rank: int | None = None
    """实例排名：0=Prefill，1=Decode"""

    kv_parallel_size: int = 1
    """并行实例数量（P2pNccl需要2）"""

    kv_ip: str = "127.0.0.1"
    kv_port: int = 14579
    """KV连接器IP和端口"""
```

### 3.2 KV Connector架构

**架构层次：**

```
vllm/distributed/kv_transfer/
├── kv_connector/
│   ├── base.py                  # Connector基类
│   ├── factory.py               # Connector工厂
│   └── v1/
│       ├── base.py              # V1基类
│       ├── mooncake/            # Mooncake Connector
│       │   ├── mooncake_connector.py
│       │   └── mooncake_utils.py
│       ├── nixl/                # NIXL Connector
│       │   ├── connector.py
│       │   ├── metadata.py
│       │   └── scheduler.py
│       ├── p2p/                 # P2P NCCL Connector
│       │   ├── p2p_nccl_connector.py
│       │   └── p2p_nccl_engine.py
│       ├── lmcache_connector.py # LMCache Connector
│       └── multi_connector.py   # 多Connector组合
```

### 3.3 Connector接口

**基类定义：** `vllm/distributed/kv_transfer/kv_connector/v1/base.py`

```python
class KVConnectorBase_V1(ABC):
    """KV Connector V1基类"""

    @abstractmethod
    def send_kv_caches(
        self,
        request_ids: list[str],
        kv_caches: list[tuple[torch.Tensor, torch.Tensor]],
        **kwargs,
    ) -> None:
        """发送KV Cache到对端实例"""
        pass

    @abstractmethod
    def recv_kv_caches(
        self,
        request_ids: list[str],
        **kwargs,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """从对端实例接收KV Cache"""
        pass

    @abstractmethod
    def drop_kv_caches(self, request_ids: list[str]) -> None:
        """丢弃KV Cache（请求结束时调用）"""
        pass

    @abstractmethod
    def check_kv_transfer_done(self, request_id: str) -> bool:
        """检查KV Cache传输是否完成"""
        pass
```

### 3.4 Mooncake Connector实现

**文件：** `vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py`

```python
class MooncakeConnector(KVConnectorBase_V1):
    """基于Mooncake Transfer Engine的KV Connector"""

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
    ):
        self.vllm_config = vllm_config
        self.role = role  # PRODUCER / CONSUMER

        # 初始化Mooncake Transfer Engine
        from mooncake.engine import TransferEngine
        self.transfer_engine = TransferEngine(
            local_ip=config.kv_ip,
            local_port=config.kv_port,
        )

        # Bootstrap服务器（协调P/D实例）
        self.bootstrap_server = MooncakeBootstrapServer(...)

    def send_kv_caches(self, request_ids, kv_caches, **kwargs):
        """发送KV Cache"""
        # 1. 注册传输区域
        for req_id, (k_cache, v_cache) in zip(request_ids, kv_caches):
            transfer_id = self._get_transfer_id(req_id)
            self.transfer_engine.register_transfer_region(
                transfer_id,
                [k_cache, v_cache],
            )

        # 2. 通知对端准备接收
        self.bootstrap_server.notify_transfer_ready(request_ids)

        # 3. 等待传输完成
        for req_id in request_ids:
            self._wait_for_transfer_complete(req_id)

    def recv_kv_caches(self, request_ids, **kwargs):
        """接收KV Cache"""
        # 1. 等待传输就绪
        transfer_regions = self.bootstrap_server.wait_for_transfer_ready(
            request_ids
        )

        # 2. 执行传输
        kv_caches = []
        for req_id, region_info in zip(request_ids, transfer_regions):
            k_cache, v_cache = self._allocate_kv_cache()
            self.transfer_engine.pull(
                region_info.remote_addr,
                [k_cache, v_cache],
            )
            kv_caches.append((k_cache, v_cache))

        return kv_caches
```

## 4. vLLM-Ascend中的PD分离实现

### 4.1 Ascend专用Connector

**文件结构：**

```
vllm_ascend/distributed/kv_transfer/
├── __init__.py
├── ascend_multi_connector.py       # 多Connector组合
├── kv_p2p/                         # P2P模式
│   ├── mooncake_connector.py       # Mooncake P2P
│   ├── mooncake_hybrid_connector.py # Mooncake混合模式
│   └── mooncake_layerwise_connector.py # 分层传输
├── kv_pool/                        # Pool模式（KV Pool）
│   ├── ascend_store/               # Ascend Store实现
│   │   ├── ascend_store_connector.py
│   │   ├── backend/
│   │   │   ├── mooncake_backend.py
│   │   │   └── memcache_backend.py
│   │   ├── pool_scheduler.py
│   │   └── pool_worker.py
│   └── cpu_offload/                # CPU卸载
│       └── cpu_offload_connector.py
└── utils/
    ├── mooncake_transfer_engine.py
    └── utils.py
```

### 4.2 Mooncake P2P Connector

**文件：** `vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py`

```python
class MooncakeConnector(KVConnectorBase_V1):
    """Ascend专用Mooncake P2P Connector"""

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
    ):
        self.vllm_config = vllm_config
        self.role = role

        # 初始化Mooncake Transfer Engine（NPU版本）
        from mooncake.engine import TransferEngine
        self.transfer_engine = TransferEngine(
            local_ip=config.kv_ip,
            local_port=config.kv_port,
        )

        # NPU特定的内存管理
        self.npu_memory_pool = self._init_npu_memory_pool()

        # ZMQ通信（协调协议）
        self.zmq_context = zmq.Context()
        self.socket = make_zmq_socket(self.zmq_context, ...)

    def send_kv_caches(self, request_ids, kv_caches, **kwargs):
        """发送KV Cache（NPU优化版本）"""
        for req_id, (k_cache, v_cache) in zip(request_ids, kv_caches):
            # 1. 转换为NPU内存布局
            k_cache_npu = self._convert_to_npu_layout(k_cache)
            v_cache_npu = self._convert_to_npu_layout(v_cache)

            # 2. 注册到Mooncake
            transfer_id = self._make_transfer_id(req_id)
            self.transfer_engine.register_segment(
                transfer_id,
                k_cache_npu.data_ptr(),
                k_cache_npu.nbytes + v_cache_npu.nbytes,
            )

            # 3. 通知Decode实例
            self._send_metadata(req_id, transfer_id, ...)

        # 4. 等待传输完成
        self._wait_for_completion(request_ids)

    def recv_kv_caches(self, request_ids, **kwargs):
        """接收KV Cache（NPU优化版本）"""
        kv_caches = []
        for req_id in request_ids:
            # 1. 从Prefill实例获取元数据
            metadata = self._recv_metadata(req_id)

            # 2. 分配NPU内存
            k_cache, v_cache = self._allocate_npu_kv_cache(
                metadata.num_blocks,
                metadata.block_size,
            )

            # 3. 执行Pull传输
            self.transfer_engine.pull(
                metadata.remote_segment_id,
                k_cache.data_ptr(),
                k_cache.nbytes,
            )
            self.transfer_engine.pull(
                metadata.remote_segment_id + k_cache.nbytes,
                v_cache.data_ptr(),
                v_cache.nbytes,
            )

            kv_caches.append((k_cache, v_cache))

        return kv_caches
```

### 4.3 KV Pool模式

**Ascend Store Connector：**

```python
class AscendStoreConnector(KVConnectorBase_V1):
    """基于KV Pool的Connector（支持多对多传输）"""

    def __init__(self, vllm_config, role):
        self.role = role

        # Pool调度器
        self.pool_scheduler = PoolScheduler(...)

        # 后端存储
        self.backend = self._init_backend(vllm_config.kv_connector_extra_config)

    def send_kv_caches(self, request_ids, kv_caches, **kwargs):
        """发送到KV Pool"""
        for req_id, (k_cache, v_cache) in zip(request_ids, kv_caches):
            # 序列化KV Cache
            serialized = self._serialize_kv_cache(k_cache, v_cache)

            # 存储到Pool
            self.backend.put(req_id, serialized)

    def recv_kv_caches(self, request_ids, **kwargs):
        """从KV Pool接收"""
        kv_caches = []
        for req_id in request_ids:
            # 从Pool读取
            serialized = self.backend.get(req_id)

            # 反序列化
            k_cache, v_cache = self._deserialize_kv_cache(serialized)
            kv_caches.append((k_cache, v_cache))

        return kv_caches
```

**支持的后端：**

| 后端 | 文件 | 说明 |
|-----|------|------|
| **Mooncake Backend** | `mooncake_backend.py` | 基于Mooncake的分布式存储 |
| **Memcache Backend** | `memcache_backend.py` | 基于Memcached的缓存 |
| **Yuanrong Backend** | `yuanrong_backend.py` | 华为远戎存储 |

### 4.4 负载均衡代理

**文件：** `vllm-ascend/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py`

```python
class LoadBalanceProxy:
    """PD分离负载均衡代理"""

    def __init__(self, prefill_instances, decode_instances):
        self.prefill_instances = prefill_instances
        self.decode_instances = decode_instances

        # 请求队列
        self.prefill_queue = asyncio.Queue()
        self.decode_queue = asyncio.Queue()

        # 实例健康检查
        self.health_checker = HealthChecker(...)

    async def handle_request(self, request):
        """处理客户端请求"""
        if self._is_prefill_request(request):
            # Prefill请求 → 路由到Prefill实例
            instance = self._select_prefill_instance()
            response = await self._forward_to_prefill(instance, request)

            # 记录KV Cache位置
            self.kv_location_cache[request.id] = {
                'prefill_instance': instance.id,
                'kv_transfer_id': response.kv_transfer_id,
            }

            return response

        else:
            # Decode请求 → 路由到Decode实例
            instance = self._select_decode_instance()

            # 注入KV Cache位置信息
            kv_info = self.kv_location_cache[request.id]
            request.kv_transfer_id = kv_info['kv_transfer_id']

            response = await self._forward_to_decode(instance, request)
            return response

    def _select_prefill_instance(self):
        """选择Prefill实例（负载均衡）"""
        # 基于队列长度选择
        return min(
            self.prefill_instances,
            key=lambda x: x.queue_length
        )

    def _select_decode_instance(self):
        """选择Decode实例（负载均衡）"""
        # 基于当前批大小选择
        return min(
            self.decode_instances,
            key=lambda x: x.current_batch_size
        )
```

---

## 5. Mooncake传输引擎集成

### 5.1 Mooncake架构

**Mooncake是一个KV Cache为中心的 disaggregated 架构**

```
Mooncake 架构层次:
┌──────────────────────────────────────────────┐
│              Transfer Engine                  │
│         (单侧零拷贝传输引擎)                  │
└────────────────────┬─────────────────────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
        ▼                         ▼
┌───────────────┐         ┌───────────────┐
│ Mooncake Store│         │   P2P Store   │
│ (分布式存储)   │         │ (点对点存储)   │
└───────────────┘         └───────────────┘
        │                         │
        └────────────┬────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────┐
│         Segment Manager                      │
│    (内存段注册、地址映射)                     │
└──────────────────────────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────┐
│        Transport Layer                       │
│   (RDMA / TCP / NVLink / HIXL)               │
└──────────────────────────────────────────────┘
```

### 5.2 Transfer Engine API

**核心接口：**

```python
from mooncake.engine import TransferEngine

# 初始化
engine = TransferEngine(
    local_ip="192.168.1.100",
    local_port=12345,
    rpc_port=12346,
)

# 注册内存段
segment_id = engine.register_segment(
    base_addr=buffer.data_ptr(),  # 内存基地址
    size=buffer.nbytes,            # 内存大小
)

# 发送（Push模式）
engine.push(
    target_ip="192.168.1.101",
    target_port=12345,
    segment_id=segment_id,
    remote_segment_id=remote_segment_id,
    size=transfer_size,
)

# 接收（Pull模式）
engine.pull(
    source_ip="192.168.1.100",
    source_port=12345,
    segment_id=remote_segment_id,
    local_segment_id=local_segment_id,
    size=transfer_size,
)

# 等待完成
engine.wait_for_completion(transfer_id)
```

### 5.3 vLLM-Ascend中的Mooncake集成

**文件：** `vllm_ascend/distributed/kv_transfer/utils/mooncake_transfer_engine.py`

```python
class MooncakeTransferEngineWrapper:
    """Mooncake Transfer Engine的NPU封装"""

    def __init__(self, local_ip, local_port):
        from mooncake.engine import TransferEngine
        self.engine = TransferEngine(local_ip, local_port)

        # NPU内存管理
        self.npu_segments = {}  # segment_id → NPU buffer

        # 全局实例
        global global_te
        global_te = self

    def register_npu_buffer(self, buffer: torch.Tensor) -> int:
        """注册NPU内存段"""
        segment_id = self.engine.register_segment(
            base_addr=buffer.data_ptr(),
            size=buffer.nbytes,
        )
        self.npu_segments[segment_id] = buffer
        return segment_id

    def push_kv_cache(
        self,
        target_ip: str,
        target_port: int,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        transfer_id: str,
    ):
        """推送KV Cache到目标实例"""
        # 注册K Cache
        k_segment = self.register_npu_buffer(k_cache)

        # 注册V Cache
        v_segment = self.register_npu_buffer(v_cache)

        # 执行Push传输
        self.engine.push(
            target_ip=target_ip,
            target_port=target_port,
            segment_id=k_segment,
            remote_segment_id=transfer_id,  # 对端已知的ID
            size=k_cache.nbytes,
        )

        self.engine.push(
            target_ip=target_ip,
            target_port=target_port,
            segment_id=v_segment,
            remote_segment_id=transfer_id + k_cache.nbytes,
            size=v_cache.nbytes,
        )

    def pull_kv_cache(
        self,
        source_ip: str,
        source_port: int,
        transfer_id: str,
        num_blocks: int,
        block_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """从源实例拉取KV Cache"""
        # 分配NPU内存
        k_cache = torch_npu.npu.empty(
            (num_blocks, block_size),
            dtype=torch.float16,
            device="npu",
        )
        v_cache = torch_npu.npu.empty(
            (num_blocks, block_size),
            dtype=torch.float16,
            device="npu",
        )

        # 注册本地段
        k_segment = self.register_npu_buffer(k_cache)
        v_segment = self.register_npu_buffer(v_cache)

        # 执行Pull传输
        self.engine.pull(
            source_ip=source_ip,
            source_port=source_port,
            segment_id=transfer_id,
            local_segment_id=k_segment,
            size=k_cache.nbytes,
        )

        self.engine.pull(
            source_ip=source_ip,
            source_port=source_port,
            segment_id=transfer_id + k_cache.nbytes,
            local_segment_id=v_segment,
            size=v_cache.nbytes,
        )

        # 等待完成
        self.engine.wait_for_completion(transfer_id)

        return k_cache, v_cache
```

---

## 6. HIXL传输库集成

### 6.1 HIXL简介

**HIXL（Huawei Xfer Library）是华为的单侧零拷贝传输库**

| 特性 | 说明 |
|-----|------|
| **单侧传输** | 只需发起端操作，接收端无需主动参与 |
| **零拷贝** | 直接在设备内存间传输，无CPU拷贝 |
| **KV Cache语义** | 原生支持KV Cache块传输 |
| **Fabric Memory** | 支持远程内存直接访问 |
| **高带宽** | D2RH/RH2D传输达64-119 GB/s |

### 6.2 HIXL架构

```
HIXL 架构:
┌──────────────────────────────────────────────┐
│            HIXL API Layer                     │
│   push_blocks / pull_blocks / switch_role    │
└────────────────────┬─────────────────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────┐
│         Cache Manager                         │
│   (KV Cache块管理、生命周期)                   │
└────────────────────┬─────────────────────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
        ▼                         ▼
┌───────────────┐         ┌───────────────┐
│ Fabric Memory │         │ Local Memory  │
│ (远程内存映射) │         │  (本地NPU)    │
└───────────────┘         └───────────────┘
        │                         │
        └────────────┬────────────┘
                     │
                     ▼
┌──────────────────────────────────────────────┐
│       HCCL Communication                     │
│   (集合通信底层，支持RDMA)                    │
└──────────────────────────────────────────────┘
```

### 6.3 HIXL API示例

**文件：** `hixl/examples/python/push_blocks_sample.py`

```python
import torch_npu
from hixl import CacheManager, TransferBackend

# 初始化Cache Manager
cache_manager = CacheManager(
    local_device="npu:0",
    remote_ip="192.168.1.101",
    remote_port=12345,
    backend=TransferBackend.HCCL_RDMA,
)

# Push KV Cache块
push_blocks_sample(cache_manager)

def push_blocks_sample(cache_manager):
    """推送KV Cache块示例"""
    # 准备KV Cache
    num_blocks = 1024
    block_size = 128
    k_cache = torch_npu.npu.randn(
        (num_blocks, block_size),
        dtype=torch.float16,
        device="npu",
    )
    v_cache = torch_npu.npu.randn(
        (num_blocks, block_size),
        dtype=torch.float16,
        device="npu",
    )

    # 注册KV Cache块
    block_ids = cache_manager.register_blocks(
        [k_cache, v_cache],
        block_size=block_size,
    )

    # 推送块到远程实例
    cache_manager.push_blocks(
        block_ids=block_ids,
        target_ip="192.168.1.101",
        target_port=12345,
        transfer_id="request_123",
    )

    # 等待推送完成
    cache_manager.wait_for_push_complete(block_ids)

def pull_blocks_sample(cache_manager):
    """拉取KV Cache块示例"""
    # 分配接收缓冲区
    num_blocks = 1024
    block_size = 128
    k_cache = torch_npu.npu.empty(
        (num_blocks, block_size),
        dtype=torch.float16,
        device="npu",
    )
    v_cache = torch_npu.npu.empty(
        (num_blocks, block_size),
        dtype=torch.float16,
        device="npu",
    )

    # 拉取块
    cache_manager.pull_blocks(
        source_ip="192.168.1.100",
        source_port=12345,
        transfer_id="request_123",
        local_blocks=[k_cache, v_cache],
    )

    # 等待拉取完成
    cache_manager.wait_for_pull_complete()

def switch_role_sample(cache_manager):
    """切换角色示例（Producer → Consumer）"""
    # Prefill完成后切换为Consumer
    cache_manager.switch_role(
        from_role="kv_producer",
        to_role="kv_consumer",
    )
```

### 6.4 vLLM-Ascend中的HIXL集成

**Mooncake Backend使用HIXL：**

```python
# vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/backend/mooncake_backend.py

class MooncakeBackend:
    """基于Mooncake+HIXL的后端"""

    def __init__(self, config):
        from mooncake.engine import TransferEngine
        # Mooncake底层使用HIXL
        self.transfer_engine = TransferEngine(
            transport="hixl",  # 使用HIXL传输层
        )

    def put(self, key: str, value: torch.Tensor):
        """存储到远程Pool"""
        segment_id = self.transfer_engine.register_segment(
            base_addr=value.data_ptr(),
            size=value.nbytes,
        )

        # Push到所有Consumer实例
        for consumer_ip in self.consumer_ips:
            self.transfer_engine.push(
                target_ip=consumer_ip,
                segment_id=segment_id,
                remote_segment_id=self._hash_key(key),
                size=value.nbytes,
            )

    def get(self, key: str) -> torch.Tensor:
        """从远程Pool读取"""
        # 找到Producer实例
        producer_ip = self._locate_producer(key)

        # 分配本地缓冲区
        buffer = torch_npu.npu.empty(...)

        # Pull from Producer
        self.transfer_engine.pull(
            source_ip=producer_ip,
            segment_id=self._hash_key(key),
            local_segment_id=self.register_segment(buffer),
            size=buffer.nbytes,
        )

        return buffer
```

---

## 7. 测试场景与测试方法

### 7.1 测试场景分类

| 测试场景 | 说明 | 测试重点 |
|---------|------|---------|
| **1P1D模式** | 1个Prefill实例 + 1个Decode实例 | 基础功能、传输性能 |
| **多P多D模式** | N个Prefill实例 + M个Decode实例 | 负载均衡、并发处理 |
| **KV Pool模式** | Prefill写入Pool，Decode从Pool读取 | 多对多传输、Pool性能 |
| **Layerwise模式** | 分层传输KV Cache | 分层优化、带宽利用 |
| **EPLB模式** | Expert Parallel Load Balancing | MoE模型、专家负载均衡 |
| **CPU Offload模式** | KV Cache卸载到CPU | 内存优化、CPU传输 |

### 7.2 基础功能测试

**测试脚本：** `vllm-ascend/tests/e2e/multicard/2-cards/test_disaggregated_encoder.py`

```python
def test_disaggregated_prefill_basic():
    """基础PD分离测试"""

    # 1. 启动Prefill实例
    prefill_config = KVTransferConfig(
        kv_connector="MooncakeConnector",
        kv_role="kv_producer",
        kv_rank=0,
        kv_ip="127.0.0.1",
        kv_port=14579,
    )

    prefill_llm = LLM(
        model="deepseek-ai/deepseek-v4",
        kv_transfer_config=prefill_config,
        tensor_parallel_size=1,
    )

    # 2. 启动Decode实例
    decode_config = KVTransferConfig(
        kv_connector="MooncakeConnector",
        kv_role="kv_consumer",
        kv_rank=1,
        kv_ip="127.0.0.1",
        kv_port=14579,
    )

    decode_llm = LLM(
        model="deepseek-ai/deepseek-v4",
        kv_transfer_config=decode_config,
        tensor_parallel_size=1,
    )

    # 3. Prefill阶段
    prefill_prompts = ["Hello, how are you?"]
    prefill_outputs = prefill_llm.generate(
        prefill_prompts,
        SamplingParams(max_tokens=1),  # 只生成首Token
    )

    # 4. Decode阶段（使用KV Cache）
    decode_outputs = decode_llm.generate(
        prefill_prompts,
        SamplingParams(max_tokens=100),
    )

    # 5. 验证结果
    assert decode_outputs[0].outputs[0].text != ""
    assert len(decode_outputs[0].outputs[0].token_ids) == 100
```

### 7.3 性能基准测试

**测试脚本：** `vllm-ascend/examples/offline_disaggregated_prefill_npu.py`

```python
def benchmark_pd_separation():
    """PD分离性能基准"""

    import time
    from statistics import mean

    # 测试参数
    num_requests = 100
    prompt_lengths = [128, 512, 1024, 2048]
    decode_lengths = [16, 32, 64, 128]

    results = {
        'prefill_latency': [],
        'kv_transfer_latency': [],
        'decode_latency': [],
        'total_latency': [],
        'throughput': [],
    }

    for prompt_len in prompt_lengths:
        for decode_len in decode_lengths:
            # 生成测试Prompt
            prompts = generate_prompts(num_requests, prompt_len)

            # Prefill阶段
            start_prefill = time.time()
            prefill_outputs = prefill_llm.generate(
                prompts,
                SamplingParams(max_tokens=1),
            )
            prefill_latency = time.time() - start_prefill

            # KV Transfer阶段
            start_transfer = time.time()
            # 等待KV Cache传输完成（通过Connector内部实现）
            for req_id in get_request_ids(prefill_outputs):
                wait_for_kv_transfer(req_id)
            transfer_latency = time.time() - start_transfer

            # Decode阶段
            start_decode = time.time()
            decode_outputs = decode_llm.generate(
                prompts,
                SamplingParams(max_tokens=decode_len),
            )
            decode_latency = time.time() - start_decode

            # 记录结果
            results['prefill_latency'].append(prefill_latency)
            results['kv_transfer_latency'].append(transfer_latency)
            results['decode_latency'].append(decode_latency)
            results['total_latency'].append(
                prefill_latency + transfer_latency + decode_latency
            )
            results['throughput'].append(
                num_requests * decode_len / (prefill_latency + decode_latency)
            )

    # 打印性能报告
    print_performance_report(results)
```

### 7.4 负载均衡测试

**测试脚本：** `vllm-ascend/examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py`

```python
async def test_load_balance_proxy():
    """负载均衡代理测试"""

    # 启动多个Prefill和Decode实例
    prefill_instances = [
        start_prefill_instance(i) for i in range(3)
    ]
    decode_instances = [
        start_decode_instance(i) for i in range(5)
    ]

    # 启动负载均衡代理
    proxy = LoadBalanceProxy(
        prefill_instances=prefill_instances,
        decode_instances=decode_instances,
    )

    # 并发请求测试
    num_concurrent_requests = 100
    tasks = [
        proxy.handle_request(generate_random_request())
        for _ in range(num_concurrent_requests)
    ]

    results = await asyncio.gather(*tasks)

    # 验证负载均衡效果
    prefill_loads = [inst.request_count for inst in prefill_instances]
    decode_loads = [inst.request_count for inst in decode_instances]

    # 计算负载方差（期望低方差）
    prefill_variance = variance(prefill_loads)
    decode_variance = variance(decode_loads)

    assert prefill_variance < threshold_prefill
    assert decode_variance < threshold_decode
```

### 7.5 KV Pool模式测试

**测试脚本：** `vllm-ascend/tests/ut/distributed/ascend_store/test_kv_transfer.py`

```python
def test_kv_pool_mode():
    """KV Pool模式测试"""

    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store import (
        AscendStoreConnector,
    )

    # 初始化Pool Connector
    connector = AscendStoreConnector(
        vllm_config=config,
        role=KVConnectorRole.PRODUCER,
    )

    # 测试写入
    request_ids = ["req_1", "req_2", "req_3"]
    kv_caches = [
        (torch.randn(1024, 128), torch.randn(1024, 128))
        for _ in range(3)
    ]

    connector.send_kv_caches(request_ids, kv_caches)

    # 测试读取（Consumer端）
    consumer_connector = AscendStoreConnector(
        vllm_config=config,
        role=KVConnectorRole.CONSUMER,
    )

    received_kv_caches = consumer_connector.recv_kv_caches(request_ids)

    # 验证数据一致性
    for i, (k_sent, v_sent) in enumerate(kv_caches):
        k_recv, v_recv = received_kv_caches[i]
        assert torch.allclose(k_sent, k_recv)
        assert torch.allclose(v_sent, v_recv)
```

---

## 8. 硬件支持与验证

### 8.1 硬件支持需求

| 硬件组件 | 最低要求 | 推荐配置 |
|---------|---------|---------|
| **NPU设备** | Ascend 910B | Ascend 910B3 / Atlas 800I A2 |
| **网络接口** | RoCEv2支持 | 100Gbps RDMA网卡 |
| **内存** | 64GB HBM | 128GB HBM（KV Pool模式） |
| **CPU** | 16核 | 32核（CPU Offload模式） |
| **存储** | NVMe SSD | 高性能NVMe（KV Cache持久化） |

### 8.2 硬件验证方法

**检查NPU设备：**

```bash
# 查看NPU设备信息
npu-smi info

# 输出示例：
# Device 0:
#   Chip Name: Ascend 910B3
#   Memory: 64GB HBM
#   Compute Units: 192 Cube Cores
```

**检查RDMA网络：**

```bash
# 查看RDMA网卡
ibv_devinfo

# 输出示例：
# hca_id: mlx5_0
#   transport: InfiniBand
#   link_layer: Ethernet
#   rate: 100 Gbps
#   state: PORT_ACTIVE

# 测试RDMA连通性
ibv_rc_pingpong -d mlx5_0 -g 0

# 测试带宽
ib_write_bw -d mlx5_0 -a <remote_ip>
```

**检查HCCL支持：**

```python
import torch_npu

# 检查HCCL可用性
print(torch_npu.npu.is_hccl_available())

# 检查设备通信能力
world_size = torch_npu.npu.device_count()
print(f"NPU devices: {world_size}")

# 测试集合通信
import torch.distributed as dist
dist.init_process_group(backend="hccl")
tensor = torch.randn(1024, device="npu")
dist.all_reduce(tensor)
```

### 8.3 性能指标验证

**关键性能指标：**

| 指标 | 测量方法 | 期望值 |
|-----|---------|-------|
| **KV Transfer带宽** | HIXL带宽测试 | ≥64 GB/s |
| **Prefill吞吐量** | Prefill实例benchmark | ≥2000 tokens/s |
| **Decode吞吐量** | Decode实例benchmark | ≥100 tokens/s |
| **TTFT（首Token延迟）** | 从请求到首Token | ≤100ms |
| **KV Transfer延迟** | 从Prefill到Decode可用 | ≤50ms |
| **内存占用** | NPU内存监控 | ≤80% HBM |

**性能测试命令：**

```bash
# HIXL带宽测试
cd hixl/tests
python test_bandwidth.py --devices npu:0,npu:1 --size 1GB

# vLLM-Ascend PD分离性能测试
cd vllm-ascend/examples
python offline_disaggregated_prefill_npu.py \
    --model deepseek-v4 \
    --num-requests 100 \
    --prompt-length 1024 \
    --decode-length 128 \
    --tensor-parallel-size 2

# 输出性能报告：
# Prefill Latency: 0.234s
# KV Transfer Latency: 0.045s
# Decode Latency: 1.523s
# Throughput: 8.5 tokens/s
```

---

## 9. 实际部署示例

### 9.1 1P1D部署示例

```bash
# Prefill实例（rank=0）
python -m vllm.entrypoints.openai.api_server \
    --model deepseek-ai/deepseek-v4 \
    --kv-role kv_producer \
    --kv-rank 0 \
    --kv-parallel-size 2 \
    --kv-ip 192.168.1.100 \
    --kv-port 14579 \
    --tensor-parallel-size 1 \
    --port 8000

# Decode实例（rank=1）
python -m vllm.entrypoints.openai.api_server \
    --model deepseek-ai/deepseek-v4 \
    --kv-role kv_consumer \
    --kv-rank 1 \
    --kv-parallel-size 2 \
    --kv-ip 192.168.1.101 \
    --kv-port 14579 \
    --tensor-parallel-size 1 \
    --port 8001
```

### 9.2 负载均衡代理部署

```bash
# 启动负载均衡代理
python load_balance_proxy_server_example.py \
    --prefill-ips 192.168.1.100:8000,192.168.1.102:8000 \
    --decode-ips 192.168.1.101:8001,192.168.1.103:8001 \
    --proxy-port 9000

# 客户端请求发送到代理
curl http://192.168.1.10:9000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "deepseek-v4",
        "prompt": "Hello, how are you?",
        "max_tokens": 100
    }'
```

---

## 10. 故障排查指南

### 10.1 常见问题

| 问题 | 原因 | 解决方案 |
|-----|------|---------|
| **KV Transfer超时** | 网络不通、RDMA未配置 | 检查RDMA连通性，增加timeout |
| **OOM（内存不足）** | KV Cache过大 | 减小batch_size，启用CPU Offload |
| **性能低于预期** | RDMA带宽不足 | 检查网卡配置，使用HIXL backend |
| **Decode等待KV** | Prefill未完成传输 | 检查Prefill日志，增加prefill实例 |
| **负载不均衡** | Proxy配置错误 | 调整负载均衡策略 |

### 10.2 日志分析

```bash
# Prefill实例日志
# 正常：KV Cache generated and registered
# 异常：Failed to register segment / Transfer timeout

# Decode实例日志
# 正常：KV Cache received successfully
# 异常：Waiting for KV Cache / Transfer failed

# Mooncake/HIXL日志
# 正常：Push/Pull completed, bandwidth XX GB/s
# 异常：RDMA connection failed / Memory registration failed
```

---

## 11. 总结

### 11.1 核心要点

1. **PD分离显著提升性能**：Prefill和Decode独立优化，互不阻塞
2. **KV Transfer是关键**：Mooncake/HIXL提供高效传输
3. **vLLM提供抽象层**：KV Connector统一接口，支持多种后端
4. **vLLM-Ascend深度优化**：NPU专用Connector、负载均衡代理
5. **硬件要求严格**：RDMA、HBM是性能基础

### 11.2 技术选型建议

| 场景 | 推荐方案 | Connector | Backend |
|-----|---------|----------|---------|
| **单节点1P1D** | P2P模式 | MooncakeP2PConnector | HIXL |
| **多节点多P多D** | Pool模式 | AscendStoreConnector | Mooncake Backend |
| **MoE模型** | EPLB模式 | MooncakeLayerwiseConnector | HIXL + HCCL |
| **内存受限** | CPU Offload | CPUOffloadConnector | Memcached |
| **生产环境** | 代理模式 | MultiConnector | Mooncake + HIXL |

### 11.3 参考文档

- **vLLM KV Transfer**: `vllm/distributed/kv_transfer/`
- **vLLM-Ascend Connector**: `vllm_ascend/distributed/kv_transfer/`
- **Mooncake文档**: https://github.com/kvcache-ai/Mooncake
- **HIXL文档**: `hixl/examples/` 和 `hixl/tests/`
- **PD分离示例**: `vllm-ascend/examples/disaggregated_prefill_v1/`

---

*文档版本：v1.0*
*创建日期：2026-06-20*
*基于vLLM、vLLM-Ascend、Mooncake、HIXL源码分析*
