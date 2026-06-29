# Mooncake 与 HIXL 集成详解

> 本文档深度解析 Mooncake 与 HIXL（Huawei Xfer Library）的集成关系，阐述 HIXL 如何为 Mooncake 提供高性能的 KV Cache 传输能力。

---

## 一、HIXL 概述

### 1.1 HIXL 定位

**HIXL（Huawei Xfer Library）** 是华为昇腾的单边通信库，面向集群场景提供简单、可靠、高效的点对点数据传输能力。

**核心优势**：
- **单边零拷贝通信**: 无需远端节点参与，直接数据传输
- **多链路支持**: HCCS、RDMA、PCIe 等多种高速互联协议
- **跨设备互联**: A2 与 A3 系列芯片无缝高速互联
- **极简 API**: 10 余个核心调用，降低集成门槛
- **高性能**: HCCS 带宽 119GB/s，RDMA 带宽 22GB/s

### 1.2 HIXL 核心组件

| 组件 | 说明 | 职责 |
|------|------|------|
| **HIXL Engine** | 核心传输引擎 | 提供基础传输接口（D2D、D2H、H2D） |
| **LLM-DataDist** | KV Cache 语义层 | 提供 KV Cache 语义的数据传输接口 |
| **ADXL** | Ascend Data Xfer Library | 底层数据传输抽象 |

### 1.3 HIXL 性能数据

**测试环境**: 昇腾 A3 芯片，传输 128M 数据

| 传输链路 | 带宽 | 对比 RoCE | 说明 |
|---------|------|----------|------|
| **HCCS** | 119 GB/s | 5.45x | 芯片内高速互联 |
| **RDMA** | 22 GB/s | 1.0x | 跨节点 RDMA 传输 |
| **RoCE** | 22 GB/s | - | 标准 RoCEv2 |

---

## 二、Mooncake 与 HIXL 的集成架构

### 2.1 集成关系

```mermaid
graph TB
    subgraph Mooncake["Mooncake 层"]
        TE[Transfer Engine<br/>传输引擎]
        MS[Mooncake Store<br/>分布式 KVCache]
        P2P[P2P Store<br/>节点间对象共享]
    end
    
    subgraph Transport_Layer["传输协议层<br/>Mooncake Transport 实现"]
        ADT[AscendDirectTransport<br/>ADXL Engine 封装]
        HCCL[HCCL Transport<br/>基于 HCCL 集合通信]
        HRT[HeterogeneousRdmaTransport<br/>异构 RDMA 互联]
        RDMA[RDMA Transport<br/>ibverbs]
        TCP[TCP Transport<br/>asio socket]
    end
    
    subgraph HIXL["HIXL / 硬件层"]
        ADXL[ADXL Engine<br/>Ascend Data Xfer Library]
        HCCS[HCCS<br/>芯片内互联<br/>119 GB/s]
        RDMA_HW[RDMA<br/>跨节点传输<br/>22 GB/s]
        PCIe[PCIe<br/>芯片间互联]
    end
    
    subgraph Hardware["硬件设备"]
        A2[Atlas A2 系列<br/>910B / 910-93]
        A3[Atlas A3 系列<br/>950 (A3)]
    end
    
    MS --> TE
    P2P --> TE
    
    TE --> ADT
    TE --> HCCL
    TE --> HRT
    TE --> RDMA
    TE --> TCP
    
    ADT --> ADXL
    HCCL --> HCCS
    HRT --> RDMA_HW
    RDMA --> RDMA_HW
    TCP --> PCIe
    
    ADXL --> HCCS
    ADXL --> RDMA_HW
    ADXL --> PCIe
    
    HCCS --> A2
    HCCS --> A3
    RDMA_HW --> A2
    RDMA_HW --> A3
    PCIe --> A2
    PCIe --> A3
    
    style Mooncake fill:#e1f5ff
    style Transport_Layer fill:#f8bbd0
    style HIXL fill:#fff9c4
    style Hardware fill:#ffccbc
```

### 2.2 集成方式

Mooncake 通过 Mooncake 集成层与 HIXL 集成：

- **C++ 层面**：`AscendDirectTransport` 直接封装 ADXL Engine（`adxl/adxl_engine.h`），编译时通过 `-DUSE_ASCEND_DIRECT=ON` 启用
- **Python 层面**：`mooncake-integration/transfer_engine/transfer_engine_py.cpp` 通过 pybind11 暴露给 Python
- **运行时传输**：ADXL Engine 负责自动选择 HCCS/RDMA/PCIe 链路

**编译时启用**：
```bash
# 编译 Mooncake 时启用 HIXL 功能
cmake -DUSE_ASCEND_DIRECT=ON ..
make -j
make install
```

**运行时配置**：
```python
from mooncake.store import MooncakeDistributedStore

store = MooncakeDistributedStore()
store.setup(
    store_ip="192.168.1.1:12345",
    metadata_url="http://master:8080",
    segment_size=1024 * 1024 * 1024,  # 1GB
    local_buffer=20 * 1024 * 1024,    # 20MB
    protocol="ascend",                # 使用 Ascend 协议（HIXL）
    grpc_url=""
)
```

---

## 三、HIXL 为 Mooncake 提供的能力

### 3.1 传输类型

HIXL 支持 4 种传输类型，在 Ascend Direct Transport 中通过 ADXL Engine 实现：

| 传输类型 | 方向 | 场景 |
|---------|------|------|
| **D2D (Device-to-Device)** | NPU → NPU | 同/跨节点 NPU 间 KVCache 传输 |
| **D2H (Device-to-Host)** | NPU → CPU | KVCache 卸载到 Host |
| **H2D (Host-to-Device)** | CPU → NPU | 从 Host 加载回 NPU |
| **H2H (Host-to-Host)** | CPU → CPU | Host 内存传输 |

传输通过 `AscendDirectTransport` 实现，内部使用 `adxl::Engine`（ADXL）的 `Transfer` / `TransferAsync` 接口：

```cpp
// 文件: mooncake-transfer-engine/include/transport/ascend_transport/ascend_direct_transport/ascend_direct_transport.h

class AscendDirectTransport : public Transport {
public:
    // 通过 ADXL Engine 执行传输
    // adxl_->Transfer(src_desc, dest_desc, stream, timeout_ms)
    // adxl_->TransferAsync(src_desc, dest_desc, callback)
    
    Status submitTransfer(BatchID batch_id,
                          const std::vector<TransferRequest>& entries) override;
    
    Status submitTransferTask(
        const std::vector<TransferTask*>& task_list) override;
    
    int install(std::string& local_server_name,
                std::shared_ptr<TransferMetadata> meta,
                std::shared_ptr<Topology> topo) override;
    
    const char* getName() const override { return "ascend_direct"; }
    
    int registerLocalMemory(void* addr, size_t length,
                            const std::string& location, bool remote_accessible,
                            bool update_metadata) override;
};
```

#### **1. batch_put_from**（Python 绑定调用 Transfer Engine 写入）

**功能**: 批量上传数据到 Mooncake Store（零拷贝）

**接口**:
```python
def batch_put_from(
    self,
    keys: List[str],           # 对象标识符列表
    buffer_ptrs: List[int],    # 内存地址列表
    sizes: List[int],          # 缓冲区大小列表
    config: ReplicateConfig = None  # 复制配置（可选）
) -> List[int]                 # 状态码列表（0=成功，负数=错误）
```

**使用示例**:
```python
# 注册缓冲区（必须先注册）
data_ptr = tensor.data_ptr()
addr = (data_ptr + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT
store.register_buffer(addr, buffer_size)

# 批量上传
keys = ["kv_cache_0", "kv_cache_1", "kv_cache_2"]
addrs = [addr, addr + size, addr + 2 * size]
sizes = [size, size, size]

results = store.batch_put_from(keys, addrs, sizes)
```

---

#### **2. batch_get_into**（Python 绑定调用 Transfer Engine 读取）

**功能**: 批量从 Mooncake Store 下载数据（零拷贝）

**接口**:
```python
def batch_get_into(
    self,
    keys: List[str],           # 对象标识符列表
    buffer_ptrs: List[int],    # 内存地址列表
    sizes: List[int]           # 缓冲区大小列表
) -> List[int]                 # 读取字节数列表（正数=成功，负数=错误）
```

**使用示例**:
```python
# 批量下载
keys = ["kv_cache_0", "kv_cache_1", "kv_cache_2"]
addrs = [remote_addr, remote_addr + size, remote_addr + 2 * size]
sizes = [size, size, size]

results = store.batch_get_into(keys, addrs, sizes)
for key, result in zip(keys, results):
    if result > 0:
        print(f"Retrieved {key}: {result} bytes")
    else:
        print(f"Failed to retrieve {key}: error {result}")
```

---

#### **3. batch_put_from_multi_buffers**

**功能**: 批量上传多缓冲区数据（零拷贝）

**接口**:
```python
def batch_put_from_multi_buffers(
    self,
    keys: List[str],                    # 对象标识符列表
    all_buffer_ptrs: List[List[int]],   # 所有缓冲区地址列表
    all_sizes: List[List[int]],         # 所有缓冲区大小列表
    config: ReplicateConfig = None      # 复制配置（可选）
) -> List[int]                          # 状态码列表
```

---

#### **4. batch_get_into_multi_buffers**

**功能**: 批量下载到多缓冲区（零拷贝）

**接口**:
```python
def batch_get_into_multi_buffers(
    self,
    keys: List[str],                    # 对象标识符列表
    all_buffer_ptrs: List[List[int]],   # 所有缓冲区地址列表
    all_sizes: List[List[int]]          # 所有缓冲区大小列表
) -> List[int]                          # 读取字节数列表
```

---

### 3.2 传输类型支持

HIXL 支持 4 种传输类型：

| 传输类型 | 说明 | 源内存 | 目标内存 | 适用场景 |
|---------|------|--------|---------|---------|
| **D2D** | Device to Device | NPU 内存 | NPU 内存 | 同节点 NPU 间传输 |
| **D2H** | Device to Host | NPU 内存 | Host 内存 | KV Cache 卸载到 CPU |
| **H2D** | Host to Device | Host 内存 | NPU 内存 | KV Cache 加载回 NPU |
| **H2H** | Host to Host | Host 内存 | Host 内存 | CPU 间传输 |

**使用示例**:
```python
# D2D 传输（NPU 到 NPU）
schema = "d2d"
tensor = torch.ones(...).npu()          # NPU tensor
target_tensor = torch.zeros(...).npu()  # NPU tensor

# D2H 传输（NPU 到 Host）
schema = "d2h"
tensor = torch.ones(...).npu()          # NPU tensor
target_tensor = torch.zeros(...).pin_memory=True).cpu()  # CPU tensor

# H2D 传输（Host 到 NPU）
schema = "h2d"
tensor = torch.ones(...).pin_memory=True).cpu()  # CPU tensor
target_tensor = torch.zeros(...).npu()  # NPU tensor

# H2H 传输（Host 到 Host）
schema = "h2h"
tensor = torch.ones(...).pin_memory=True).cpu()  # CPU tensor
target_tensor = torch.zeros(...).pin_memory=True).cpu()  # CPU tensor
```

---

### 3.3 多链路支持

HIXL 支持多种传输链路，自动选择最优路径：

#### **HCCS（Huawei Cache Coherence System）**

**特点**:
- 芯片内高速互联
- 带宽: 119 GB/s
- 延迟: 极低
- 适用: 同节点内 NPU 间传输

**配置**:
```bash
# 使用 HCCS（禁用 RDMA）
export HCCL_INTRA_ROCE_ENABLE=0
export HCCL_INTRA_PCIE_ENABLE=1
```

---

#### **RDMA**

**特点**:
- 跨节点高速传输
- 带宽: 22 GB/s
- 延迟: 低
- 适用: 跨节点 NPU 间传输

**配置**:
```bash
# 使用 RDMA
export HCCL_INTRA_ROCE_ENABLE=1
```

---

#### **PCIe**

**特点**:
- 芯片间互联
- 带宽: 中等
- 延迟: 中等
- 适用: 同节点不同芯片间传输

**配置**:
```bash
# 使用 PCIe
export HCCL_INTRA_ROCE_ENABLE=0
export HCCL_INTRA_PCIE_ENABLE=1
```

---

## 四、Mooncake + HIXL 使用示例

### 4.1 完整示例：批量 KV Cache 传输

```python
import torch
import torch_npu
from mooncake.store import MooncakeDistributedStore

# 配置参数
SEGMENT_SIZE = 1024 * 1024 * 1024  # 1GB
LOCAL_BUFFER = 20 * 1024 * 1024    # 20MB
ALIGNMENT = 2 * 1024 * 1024        # 2MB 对齐

# 1. 初始化 Mooncake Store
store = MooncakeDistributedStore()
store.setup(
    store_ip="192.168.1.1:12345",
    metadata_url="http://master:8080",
    segment_size=SEGMENT_SIZE,
    local_buffer=LOCAL_BUFFER,
    protocol="ascend",  # 使用 HIXL
    grpc_url=""
)

# 2. 创建 NPU tensor（模拟 KV Cache）
# 假设：33 层，61 个 block，每个 block 144KB
tensor = torch.ones(33, 61, 144 * 1024, dtype=torch.int8).npu()
target_tensor = torch.zeros(33, 61, 144 * 1024, dtype=torch.int8).npu()

# 3. 注册缓冲区（必须先注册）
data_ptr = tensor.data_ptr()
addr = (data_ptr + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT
store.register_buffer(addr, 61 * 32 * 144 * 1024)

target_data_ptr = target_tensor.data_ptr()
remote_addr = (target_data_ptr + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT
store.register_buffer(remote_addr, 61 * 32 * 144 * 1024)

# 4. 批量上传 KV Cache
keys = []
addrs = []
sizes = []

for block_i in range(32):
    for layer in range(61):
        key = f"kv_cache_rank0_{block_i}_{layer}"
        keys.append(key)
        addrs.append(addr)
        sizes.append(144 * 1024)
        addr += 144 * 1024

# 执行批量上传
results = store.batch_put_from(keys, addrs, sizes)
print(f"Uploaded {len([r for r in results if r == 0])} KV Cache blocks")

# 5. 批量下载 KV Cache（从其他节点）
get_keys = [f"kv_cache_rank1_{block_i}_{layer}" 
            for block_i in range(32) for layer in range(61)]
remote_addrs = [remote_addr + i * 144 * 1024 for i in range(len(get_keys))]

# 执行批量下载
results = store.batch_get_into(get_keys, remote_addrs, sizes)
print(f"Downloaded {len([r for r in results if r > 0])} KV Cache blocks")

# 6. 关闭 Store
store.close()
```

---

### 4.2 分布式集群配置

**配置文件** (`config.yaml`):
```yaml
# 分布式集群配置
distributed: true
world_size: 4

# Master 地址
master_addr: "192.168.1.100"
master_port: "29500"

# Mooncake Store 配置
mooncake_store_ip: "192.168.1.1"
mooncake_store_port_start: 12345
metadata_url: "http://192.168.1.100:8080"
grpc_url: ""

# 传输配置
schema: "d2d"  # d2d, d2h, h2d, h2h
```

**启动 Master**:
```bash
mooncake_master \
  --enable_http_metadata_server=true \
  --http_metadata_server_host=0.0.0.0 \
  --http_metadata_server_port=8080
```

**运行示例**:
```bash
# 单机单卡
bash run.sh batch_put_get_sample.py --device_id=0 --schema="d2d"

# 单机多卡（使用配置文件）
bash run.sh batch_put_get_sample.py \
  --device_id=0 \
  --rank=0 \
  --world_size=4 \
  --distributed \
  --config=config.yaml
```

---

## 五、性能优化建议

### 5.1 传输类型选择

| 场景 | 推荐传输类型 | 原因 |
|------|------------|------|
| **同节点 NPU 间** | D2D + HCCS | 最高带宽（119 GB/s） |
| **跨节点 NPU 间** | D2D + RDMA | 高带宽（22 GB/s） |
| **KV Cache 卸载** | D2H | 卸载到 Host 内存 |
| **KV Cache 加载** | H2D | 从 Host 加载回 NPU |

---

### 5.2 批量传输优化

**建议**:
1. **批量大小**: 建议 32-64 个 block 一起传输
2. **缓冲区对齐**: 2MB 对齐，提高传输效率
3. **缓冲区注册**: 传输前必须注册，避免重复注册
4. **内存连续**: 尽量使用连续内存，减少传输次数

---

### 5.3 链路配置优化

**HCCS 优先**:
```bash
# 同节点内优先使用 HCCS
export HCCL_INTRA_ROCE_ENABLE=0
export HCCL_INTRA_PCIE_ENABLE=1
```

**RDMA 优化**:
```bash
# 跨节点使用 RDMA
export HCCL_INTRA_ROCE_ENABLE=1

# RDMA Traffic Class（QoS）
export HCCL_RDMA_TRAFFIC_CLASS=0
```

---

## 六、与标准 Mooncake 对比

### 6.1 功能对比

| 功能 | 标准 Mooncake | Mooncake + HIXL |
|------|--------------|----------------|
| **传输协议** | TCP, RDMA | TCP, RDMA, **HCCS, PCIe** |
| **零拷贝** | ✅ | ✅ |
| **单边传输** | ❌ | ✅ |
| **NPU 内存支持** | ❌ | ✅ |
| **D2D/D2H/H2D** | ❌ | ✅ |
| **最高带宽** | 22 GB/s (RDMA) | **119 GB/s (HCCS)** |

---

### 6.2 性能对比

**测试场景**: 传输 40GB 数据（LLaMA3-70B 128k tokens KV Cache）

| 方案 | 带宽 | 时间 | 提升 |
|------|------|------|------|
| **TCP** | ~9 GB/s | 4.4s | 1.0x |
| **RDMA** | ~22 GB/s | 1.8s | 2.4x |
| **HIXL (HCCS)** | **119 GB/s** | **0.34s** | **13x** |
| **HIXL (RDMA)** | **22 GB/s** | **1.8s** | **2.4x** |

---

## 七、故障排查

### 7.1 常见错误

#### **错误 1: Buffer 未注册**

```
RuntimeError: Buffer not registered
```

**解决方案**:
```python
# 必须先注册缓冲区
store.register_buffer(addr, buffer_size)
```

---

#### **错误 2: 链路配置冲突**

```
[Parse] [IntraLinkType]only set HCCL_INTRA_ROCE_ENABLE, and the val is zero, 
pls set HCCL_INTRA_PCIE_ENABLE
```

**解决方案**:
```bash
# 不要同时禁用 RDMA 和 PCIe
export HCCL_INTRA_ROCE_ENABLE=1  # 或
export HCCL_INTRA_PCIE_ENABLE=1
```

---

#### **错误 3: 内存对齐问题**

```
RuntimeError: Address not aligned
```

**解决方案**:
```python
# 2MB 对齐
ALIGNMENT = 2 * 1024 * 1024
aligned_addr = (addr + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT
```

---

## 八、总结

### 8.1 核心价值

**Mooncake + HIXL 集成为 Mooncake 带来**：
1. **昇腾 NPU 原生支持**: 直接传输 NPU 内存，无需 CPU 中转
2. **单边零拷贝**: 无需远端参与，降低延迟
3. **多链路支持**: HCCS、RDMA、PCIe 自动选择
4. **极致性能**: HCCS 119GB/s，是 RDMA 的 5.45 倍
5. **KV Cache 语义**: LLM-DataDist 提供专用接口

### 8.2 适用场景

| 场景 | 推荐使用 |
|------|---------|
| **昇腾 NPU 集群** | ✅ 强烈推荐 |
| **PD 分离推理** | ✅ 强烈推荐 |
| **KV Cache 分布式存储** | ✅ 强烈推荐 |
| **跨节点 KV Cache 传输** | ✅ 推荐 |
| **GPU 集群** | ❌ 使用标准 Mooncake |

---

**文档版本**: v2.0  
**创建时间**: 2026-06-27  
**基于源码**: `mooncake-transfer-engine/include/transport/ascend_transport/ascend_direct_transport/ascend_direct_transport.h`、`mooncake-transfer-engine/src/transport/ascend_transport/ascend_direct_transport/ascend_direct_transport.cpp`、`Mooncake 集成示例代码`
**维护者**: vLLM-Ascend 项目团队