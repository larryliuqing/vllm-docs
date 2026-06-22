# ROCE、HCCS、UB 三种传输协议详解

> 本文档详细分析 HIXL 中三种传输协议 (ROCE, HCCS, UB) 的原理、区别、选择机制和处理流程。

---

## 1. 协议概述与对比

### 1.1 昇腾硬件互联拓扑

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          昇腾硬件互联拓扑                                     │
└─────────────────────────────────────────────────────────────────────────────┘

                    ┌─────────────────────────────────────┐
                    │          SuperPod (集群)             │
                    │                                     │
   ┌────────────────┼─────────────────────────────────────┼────────────────┐
   │                │                                     │                │
   │   ┌────────────▼────────────┐       ┌───────────────▼────────────┐   │
   │   │   Supernode A (超节点)   │       │   Supernode B (超节点)     │   │
   │   │                         │       │                            │   │
   │   │  ┌─────┐  ┌─────┐       │       │      ┌─────┐  ┌─────┐     │   │
   │   │  │ NPU │──│ NPU │       │ ROCE  │      │ NPU │──│ NPU │     │   │
   │   │  │  0  │  │  1  │       │◄─────►│      │  4  │  │  5  │     │   │
   │   │  └──┬──┘  └──┬──┘       │ (网络)│      └──┬──┘  └──┬──┘     │   │
   │   │     │  HCCS  │          │       │         │  HCCS  │        │   │
   │   │     │(芯片间) │          │       │         │(芯片间) │        │   │
   │   │  ┌──▼──┐  ┌──▼──┐       │       │      ┌──▼──┐  ┌──▼──┐     │   │
   │   │  │ NPU │──│ NPU │       │       │      │ NPU │──│ NPU │     │   │
   │   │  │  2  │  │  3  │       │       │      │  6  │  │  7  │     │   │
   │   │  └─────┘  └─────┘       │       │      └─────┘  └─────┘     │   │
   │   │       UB (芯片内)        │       │          UB (芯片内)      │   │
   │   └─────────────────────────┘       └──────────────────────────┘   │
   │                                                                     │
   │   net_instance_id = "A"              net_instance_id = "B"          │
   └─────────────────────────────────────────────────────────────────────┘

   协议选择规则:
   ┌──────────────────────────────────────────────────────────────────────┐
   │  同一 NPU 内     → UB (Unified Bus)     最高性能 ~119 GB/s          │
   │  同一超节点内    → HCCS 或 UB            高性能 ~80-100 GB/s         │
   │  跨超节点       → ROCE                   中等性能 ~22 GB/s          │
   │  强制 ROCE      → 环境变量 HCCL_INTRA_ROCE_ENABLE=1                  │
   └──────────────────────────────────────────────────────────────────────┘
```

### 1.2 三种协议详细对比

| 特性 | **UB (Unified Bus)** | **HCCS** | **ROCE** |
|------|---------------------|----------|----------|
| **全称** | Unified Bus | Huawei Cluster Communication System | RDMA over Converged Ethernet |
| **适用范围** | 同芯片内 / 同超节点内 | 同节点内跨芯片 | 跨节点 / 跨超节点 |
| **物理介质** | 芯片内部总线 | 高速片间互联 | 以太网 (RDMA) |
| **传输引擎** | AICPU Kernel | HCCL 硬件引擎 | RDMA NIC |
| **带宽** | ~119 GB/s | ~80-100 GB/s | ~22 GB/s |
| **延迟** | ~1-5 μs | ~5-10 μs | ~10-50 μs |
| **内存类型** | D2D, D2H, H2D, H2H | D2D | Any (Host/Device) |
| **CPU参与** | 否 (AICPU 执行) | 否 (硬件) | 否 (RDMA offload) |
| **完成机制** | Notify + Kernel | 硬件完成队列 | RDMA CQ (完成队列) |
| **net_instance_id** | 必须相同 | 必须相同 | 可不同 |

### 1.3 CommType 枚举定义

```cpp
// src/hixl/engine/hixl_client.h
enum class CommType : uint32_t {
  COMM_TYPE_UB_D2D = 0U,    // Device-to-Device via UB
  COMM_TYPE_UB_H2D = 1U,    // Host-to-Device via UB
  COMM_TYPE_UB_D2H = 2U,    // Device-to-Host via UB
  COMM_TYPE_UB_H2H = 3U,    // Host-to-Host via UB
  COMM_TYPE_ROCE  = 4U,     // RDMA over Converged Ethernet
  COMM_TYPE_HCCS  = 5U      // Huawei Cluster Communication System
};
```

---

## 2. 协议选择机制

### 2.1 选择决策树

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           协议选择决策树                                     │
└─────────────────────────────────────────────────────────────────────────────┘

                              开始传输请求
                                  │
                                  ▼
                   ┌──────────────────────────────┐
                   │ HCCL_INTRA_ROCE_ENABLE == 1? │
                   └──────────────────────────────┘
                         │              │
                        Yes            No
                         │              │
                         ▼              ▼
                    ┌────────┐    ┌──────────────────────────────┐
                    │  ROCE  │    │ local.net_instance_id ==     │
                    │        │    │ remote.net_instance_id ?     │
                    └────────┘    └──────────────────────────────┘
                                       │              │
                                      No             Yes
                                       │              │
                                       ▼              ▼
                                  ┌────────┐    ┌─────────────────────┐
                                  │  ROCE  │    │ 使用 UB 端点匹配     │
                                  │(跨超节点)│   │                     │
                                  └────────┘    └─────────────────────┘
                                                       │
                                                       ▼
                                          ┌─────────────────────────┐
                                          │  根据 placement 分类     │
                                          ├─────────────────────────┤
                                          │ local=device, remote=device  → UB_D2D  │
                                          │ local=device, remote=host    → UB_D2H  │
                                          │ local=host,   remote=device  → UB_H2D  │
                                          │ local=host,   remote=host    → UB_H2H  │
                                          └─────────────────────────┘
```

### 2.2 核心选择代码

```cpp
// src/hixl/engine/hixl_client.cc

bool HixlClient::MustUseRoce(const std::vector<EndpointConfig> &local_endpoint_list,
                             const std::vector<EndpointConfig> &remote_endpoint_list) const {
  // 条件1: 检查环境变量 HCCL_INTRA_ROCE_ENABLE
  std::string env_roce_enable;
  const char *env_ret = std::getenv("HCCL_INTRA_ROCE_ENABLE");
  if (env_ret != nullptr) {
    env_roce_enable = env_ret;
  }
  const bool is_env_roce_enabled = (env_roce_enable == "1");
  
  // 条件2: 检查是否在同一超节点 (net_instance_id)
  const bool is_net_instance_different =
      local_endpoint_list[0].net_instance_id != remote_endpoint_list[0].net_instance_id;
  
  // 任一条件满足则必须使用 ROCE
  return is_env_roce_enabled || is_net_instance_different;
}
```

### 2.3 传输分类逻辑

```cpp
// src/hixl/engine/hixl_client.cc

Status HixlClient::ClassifyTransfers(const std::vector<TransferOpDesc> &op_descs,
                                     std::map<CommType, std::vector<TransferOpDesc>> &op_descs_table) {
  for (const auto &op_desc : op_descs) {
    // 1. 判断本地内存类型
    MemType local_mem_type;
    {
      std::lock_guard<std::mutex> lock(local_segments_mutex_);
      GetMemType(local_segments_, op_desc.local_addr, op_desc.len, local_mem_type);
    }
    
    // 2. 判断远程内存类型
    MemType remote_mem_type;
    {
      std::lock_guard<std::mutex> lock(remote_segments_mutex_);
      GetMemType(remote_segments_, op_desc.remote_addr, op_desc.len, remote_mem_type);
    }

    // 3. 如果 ROCE client 存在，直接使用 ROCE
    {
      std::lock_guard<std::mutex> lock(client_handles_mutex_);
      if (client_handles_.find(CommType::COMM_TYPE_ROCE) != client_handles_.end()) {
        op_descs_table[CommType::COMM_TYPE_ROCE].push_back(op_desc);
        continue;
      }
    }

    // 4. 否则根据内存位置选择 UB 类型
    CommType cur_type;
    if (local_mem_type == MEM_DEVICE) {
      cur_type = (remote_mem_type == MEM_DEVICE) ? 
          CommType::COMM_TYPE_UB_D2D : CommType::COMM_TYPE_UB_D2H;
    } else {
      cur_type = (remote_mem_type == MEM_DEVICE) ? 
          CommType::COMM_TYPE_UB_H2D : CommType::COMM_TYPE_UB_H2H;
    }
    op_descs_table[cur_type].push_back(op_desc);
  }
  return SUCCESS;
}
```

### 2.4 Endpoint 匹配规则

```cpp
// src/hixl/engine/hixl_client.h

struct MatchKey {
  std::string dst_eid;   // 目标 endpoint ID（空表示通配）
  std::string plane;     // 通信平面
  std::string placement; // 内存位置（"host" 或 "device"）
  
  bool Matches(const MatchKey &query) const {
    // 通配匹配：空 dst_eid 匹配任意
    if (!dst_eid.empty() && !query.dst_eid.empty() && dst_eid != query.dst_eid) {
      return false;
    }
    // plane 和 placement 必须精确匹配
    return plane == query.plane && placement == query.placement;
  }
};

// 解析通信类型
CommType ParseCommType(const std::string &local_placement, const std::string &remote_placement) const {
  if (local_placement == kPlacementDevice && remote_placement == kPlacementDevice) {
    return CommType::COMM_TYPE_UB_D2D;
  } else if (local_placement == kPlacementDevice && remote_placement == kPlacementHost) {
    return CommType::COMM_TYPE_UB_D2H;
  } else if (local_placement == kPlacementHost && remote_placement == kPlacementHost) {
    return CommType::COMM_TYPE_UB_H2H;
  } else {
    return CommType::COMM_TYPE_UB_H2D;
  }
}
```

---

## 3. ROCE (RDMA over Converged Ethernet) 详解

### 3.1 ROCE 工作原理

ROCE (RDMA over Converged Ethernet) 是一种允许在以太网上实现 RDMA (Remote Direct Memory Access) 的协议。

**核心特性**：
- **零拷贝**：数据直接从发送方内存传输到接收方内存，无需 CPU 参与
- **内核旁路**：绕过操作系统内核，直接访问网络硬件
- **CPU 卸载**：网络协议处理由 NIC 硬件完成

### 3.2 ROCE 连接建立流程

```
Server                                    Client
  │                                         │
  │  1. HixlCSServer::Initialize()          │
  │     ├─ EndpointCreate()                 │
  │     │   └─ HCCL 创建 ROCE endpoint      │
  │     └─ Listen() 等待连接                │
  │                                         │
  │                    ◄────────────────────│  2. TCP Connect
  │                                         │
  │  3. Accept + 交换 Endpoint 信息         │
  │     ├─ 发送 local_endpoint 配置         │
  │     └─ 接收 remote_endpoint 配置 ──────►│
  │                                         │
  │  4. CreateChannel()                     │
  │     └─ HCCL ChannelCreate()             │
  │        (创建 RDMA QP)                   │
  │                                         │
  │  5. MemExport() 导出内存描述符           │
  │     └─ 生成 remote_mem_desc ───────────►│  6. MemImport()
  │                                         │     └─ 获取远程内存访问权限
  │                                         │
  │  ✓ 连接就绪                             │  ✓ 连接就绪
```

### 3.3 ROCE 数据传输流程

```cpp
// src/hixl/cs/hixl_cs_client.cc

Status HixlCSClient::BatchTransferHost(bool is_get, const CommunicateMem &mem, void **handle) {
  // ============================================================
  // 阶段1: 提交 RDMA 批量操作
  // ============================================================
  BatchTransferTask(is_get, mem);
}

Status HixlCSClient::BatchTransferTask(bool is_get, const CommunicateMem &mem) {
  ThreadHandle thread = local_endpoint_->GetThread();
  
  if (is_get) {
    // 批量 RDMA Read: 从远程读取到本地
    for (uint32_t i = 0; i < mem.list_num; i++) {
      HcommProxy::ReadNbiOnThread(thread, channel, 
                                   mem.dst_buf_list[i],  // local destination
                                   mem.src_buf_list[i],  // remote source
                                   mem.len_list[i]);
    }
  } else {
    // 批量 RDMA Write: 从本地写入到远程
    for (uint32_t i = 0; i < mem.list_num; i++) {
      HcommProxy::WriteNbiOnThread(thread, channel,
                                    mem.dst_buf_list[i],  // remote destination
                                    mem.src_buf_list[i],  // local source
                                    mem.len_list[i]);
    }
  }
  
  // ChannelFence 确保顺序完成
  HcommProxy::ChannelFenceOnThread(thread, channel);
  
  return SUCCESS;
}
```

### 3.4 ROCE 完成机制

```
┌─────────┐                                      ┌─────────┐
│  CLIENT │                                      │  SERVER │
└────┬────┘                                      └────┬────┘
     │                                                │
     │  ┌─────────────────────────────────────────┐  │
     │  │ RDMA Write (批量)                        │  │
     │  │ local_src → remote_dst                  │  │
     │  └─────────────────────────────────────────┘  │
     │ ──────────────────────────────────────────────►│
     │        WriteNbiOnThread (非阻塞)              │ trans_flag = 0
     │                                                │ (数据正在写入...)
     │                                                │
     │  ┌─────────────────────────────────────────┐  │
     │  │ ChannelFence (内存屏障)                  │  │
     │  │ 确保所有 Write 完成后再读取 flag         │  │
     │  └─────────────────────────────────────────┘  │
     │ ──────────────────────────────────────────────►│
     │                                                │ trans_flag = 1
     │                                                │ (写入完成)
     │  ┌─────────────────────────────────────────┐  │
     │  │ RDMA Read (完成标志)                     │  │
     │  │ remote_flag → local_flag_queue          │  │
     │  └─────────────────────────────────────────┘  │
     │ ◄──────────────────────────────────────────────│
     │        ReadNbiOnThread                         │
     │                                                │
     ▼                                                ▼
  flag_queue_[index] = 1                         数据写入完成
     │
     ▼
  CheckStatus() → COMPLETE
```

**完成标志机制**：

```cpp
// flag_queue_ 是 Host 内存的 uint64_t 数组 (4096 个槽位)
uint64_t flag_queue_[4096];

// 获取可用槽位
int32_t AcquireFlagIndex() {
  std::lock_guard<std::mutex> lock(indices_mutex_);
  if (top_index_ > 0) {
    return available_indices_[--top_index_];  // 栈式分配
  }
  return -1;  // 无可用槽位
}

// 检查完成状态
Status CheckStatusHost(CompleteHandle &handle, HixlCompleteStatus &status) {
  if (*handle.flag_address == 1) {
    status = HIXL_COMPLETE_STATUS_COMPLETE;
    ReleaseFlagIndex(handle.flag_index);  // 释放槽位
  } else {
    status = HIXL_COMPLETE_STATUS_WAITING;
  }
  return SUCCESS;
}
```

### 3.5 ROCE 关键 API

| API | 功能 | 说明 |
|-----|------|------|
| `EndpointCreate` | 创建 ROCE 端点 | 分配 RDMA 资源 |
| `MemReg` | 注册内存 | 锁定物理页面，获取 RDMA 访问权限 |
| `MemExport` | 导出内存描述符 | 生成包含访问密钥的序列化描述符 |
| `MemImport` | 导入内存描述符 | 获取远程内存访问句柄 |
| `ChannelCreate` | 创建 RDMA 通道 | 建立 QP (Queue Pair) 连接 |
| `ReadNbiOnThread` | 非阻塞 RDMA Read | 从远程内存读取数据 |
| `WriteNbiOnThread` | 非阻塞 RDMA Write | 向远程内存写入数据 |
| `ChannelFence` | 内存屏障 | 确保操作顺序完成 |

---

## 4. UB (Unified Bus) 详解

### 4.1 UB 工作原理

UB (Unified Bus) 是昇腾芯片内部的高速互联总线，通过 AICPU Kernel 执行数据传输。

**核心特性**：
- **最高带宽**：D2D 可达 119 GB/s
- **最低延迟**：芯片内传输约 1 μs
- **AICPU 执行**：不占用主 CPU，由 AICPU Kernel 完成传输
- **Notify 机制**：使用硬件通知完成

### 4.2 UB 子类型

| 类型 | 本地内存 | 远程内存 | 典型场景 | 带宽 |
|------|---------|---------|---------|------|
| UB_D2D | Device | Device | KV Cache 传输 | ~119 GB/s |
| UB_D2H | Device | Host | 日志/结果回传 | ~80 GB/s |
| UB_H2D | Host | Device | 参数加载 | ~80 GB/s |
| UB_H2H | Host | Host | 控制面数据 | ~50 GB/s |

### 4.3 UB 数据传输流程

```cpp
// src/hixl/cs/hixl_cs_client.cc

Status HixlCSClient::BatchTransferDevice(bool is_get, const CommunicateMem &mem, void **handle) {
  // ============================================================
  // 阶段1: 从 CompletePool 获取 slot
  // ============================================================
  CompletePool::SlotHandle slot;
  AcquireUbSlot(slot);
  // slot 包含: stream, notify, thread, host_flag
  
  // ============================================================
  // 阶段2: 准备设备侧参数
  // ============================================================
  MemDev mem_dev;
  // 将 dst/src/len 数组拷贝到设备内存
  AllocAndCopyDeviceBuffer(&mem_dev.dst_buf_list_dev, mem.dst_buf_list, ...);
  AllocAndCopyDeviceBuffer(&mem_dev.src_buf_list_dev, mem.src_buf_list, ...);
  AllocAndCopyDeviceBuffer(&mem_dev.len_list_dev, mem.len_list, ...);
  
  // ============================================================
  // 阶段3: 准备远程 flag 和加载 AICPU Kernel
  // ============================================================
  void *remote_flag;
  PrepareUbRemoteFlagAndKernel(remote_flag);
  
  // ============================================================
  // 阶段4: 启动 AICPU Kernel
  // ============================================================
  LaunchUbAndStage(is_get, handle, remote_flag);
}
```

### 4.4 AICPU Kernel 执行流程

```
┌─────────────────────────────────────────────────────────────┐
│ AICPU Kernel (HixlBatchGet 或 HixlBatchPut)                 │
│                                                             │
│  1. 从 args 读取参数:                                        │
│     - dst_buf_list, src_buf_list, len_list                  │
│     - thread, channel                                       │
│     - remote_flag, local_flag                               │
│                                                             │
│  2. 批量 UB 传输:                                            │
│     for (i = 0; i < list_num; i++) {                        │
│       if (is_get) {                                         │
│         UB_Read(dst[i], src[i], len[i]);                    │
│       } else {                                              │
│         UB_Write(dst[i], src[i], len[i]);                   │
│       }                                                     │
│     }                                                       │
│                                                             │
│  3. 写入远程 flag (通知 Server):                             │
│     *remote_flag = 1;                                       │
│                                                             │
│  4. 触发本地 notify (通知 Client):                           │
│     rtNotify(local_flag);                                   │
└─────────────────────────────────────────────────────────────┘
```

### 4.5 UB 传输时序图

```
┌─────────┐                    ┌─────────┐                    ┌─────────┐
│  CLIENT │                    │  AICPU  │                    │  SERVER │
│  (Host) │                    │ (Device)│                    │ (Device)│
└────┬────┘                    └────┬────┘                    └────┬────┘
     │                              │                              │
     │  1. AcquireUbSlot()          │                              │
     │     (获取 stream/notify)     │                              │
     │                              │                              │
     │  2. 准备参数到设备内存        │                              │
     │     (dst/src/len arrays)     │                              │
     │ ─────────────────────────────►│                              │
     │                              │                              │
     │  3. aclrtLaunchKernel()      │                              │
     │     (HixlBatchGet/Put)       │                              │
     │ ─────────────────────────────►│                              │
     │                              │                              │
     │                              │  4. 执行批量 UB 传输          │
     │                              │     (UB_Read/Write)          │
     │                              │ ──────────────────────────────►│
     │                              │                              │
     │                              │  5. 写入 remote_flag = 1     │
     │                              │ ──────────────────────────────►│
     │                              │                              │
     │                              │  6. rtNotify(local_flag)     │
     │                              │ ────────────┐                │
     │                              │             │                │
     │  7. aclrtWaitAndResetNotify()│             │                │
     │     (等待 Notify 信号)       │◄────────────┘                │
     │                              │                              │
     │  8. aclrtMemcpyAsync()       │                              │
     │     (D2H: dev_const_one → host_flag)                        │
     │ ◄─────────────────────────────│                              │
     │                              │                              │
     ▼                              │                              ▼
  host_flag = 1                     │                          数据传输完成
     │                              │
     ▼                              │
  CheckStatus() → COMPLETE          │
```

### 4.6 CompletePool 设计

```cpp
// src/hixl/cs/complete_pool.h

class CompletePool {
 public:
  static constexpr uint32_t kMaxSlots = 128U;  // 最大 128 个并发 slot
  static CompletePool& GetInstance();          // 单例访问
  
  struct SlotHandle {
    uint32_t slot_index;
    aclrtContext ctx;        // ACL context
    aclrtStream stream;      // 异步执行流
    ThreadHandle thread;     // HCCL thread (用于 UB)
    aclrtNotify notify;      // 硬件通知对象
    void *host_flag;         // Host 可见的完成标志
    uint64_t notify_addr;    // notify 的设备地址
    uint32_t notify_len;
  };
  
 private:
  struct Slot {
    bool in_use;
    aclrtContext ctx;
    aclrtStream stream;
    ThreadHandle thread;
    aclrtNotify notify;
    uint64_t notify_addr;
    uint32_t notify_len;
    void *host_flag;
  };
  
  std::deque<uint32_t> free_list_;      // 可用 slot 索引栈
  std::array<Slot, kMaxSlots> slots_;   // 预分配的所有 slot
  uint32_t ref_cnt_;                    // 引用计数
};

// 初始化时预分配所有资源
Status InitAllSlotsLocked(int32_t device_id, CommEngine engine, 
                          uint32_t thread_num, uint32_t notify_num_per_thread) {
  for (uint32_t i = 0; i < kMaxSlots; i++) {
    Slot &slot = slots_[i];
    
    // 1. 创建 ACL context 和 stream
    aclrtCreateContext(&slot.ctx, device_id);
    aclrtCreateStream(&slot.stream);
    
    // 2. 分配 HCCL thread
    HcommProxy::ThreadAlloc(COMM_ENGINE_AICPU, 1, 1, &slot.thread);
    
    // 3. 创建 notify
    aclrtCreateNotify(slot.stream, &slot.notify);
    
    // 4. 获取 notify 的设备地址 (供 Kernel 使用)
    rtGetDevResAddress(notify_id, &slot.notify_addr, &slot.notify_len);
    
    // 5. 分配 pinned host memory
    aclrtMallocHost(&slot.host_flag, 8);
  }
}
```

### 4.7 AICPU Kernel 加载

```cpp
// src/hixl/cs/load_kernel.cc

Status LoadUbKernelAndGetHandles(const std::string &kernel_json_path,
                                   aclrtBinHandle &bin_handle,
                                   void *&func_get, void *&func_put) {
  // 1. 加载 kernel binary
  aclrtBinaryLoadFromFile(kernel_json_path.c_str(), 
                          ACL_BINARY_LOAD_FROM_FILE, 
                          &bin_handle);
  
  // 2. 获取 kernel 函数句柄
  aclrtGetKernelStubFunc(bin_handle, "HixlBatchGet", &func_get);
  aclrtGetKernelStubFunc(bin_handle, "HixlBatchPut", &func_put);
}

// Kernel 路径: $ASCEND_HOME_PATH/opp/built-in/op_impl/aicpu/config/libcann_hixl_kernel.json
```

---

## 5. HCCS (Huawei Cluster Communication System) 详解

### 5.1 HCCS 工作原理

HCCS 是华为自研的高速芯片间互联协议，用于同一服务器节点内多个 NPU 芯片之间的通信。

**核心特性**：
- **高带宽**：约 80-100 GB/s
- **低延迟**：约 5-10 μs
- **硬件实现**：由 HCCL 硬件引擎执行
- **同节点**：仅限同一物理服务器内的芯片

### 5.2 HCCS Endpoint 匹配

```cpp
// src/hixl/cs/endpoint_store.cc

// HCCS endpoint 通过 commAddr.id 进行匹配
if (lhs.protocol == COMM_PROTOCOL_HCCS) {
  return lhs.commAddr.id == rhs.commAddr.id;
}
```

### 5.3 HCCS vs ROCE vs UB 对比

| 协议 | 范围 | 传输方式 | 拓扑要求 |
|------|------|---------|---------|
| HCCS | 节点内跨芯片 | 高速片间互联 | 同一 `serverIdx` |
| UB | 芯片内/超节点内 | Unified Bus (AICPU) | 同一 `net_instance_id` |
| ROCE | 跨节点 | RDMA over Ethernet | 任意，或强制 |

### 5.4 HCCS 处理流程

HCCS 在 HIXL 中主要通过 HCCL 底层库处理：

1. **连接建立**：通过 HCCL 的 HCOMM 层创建 HCCS endpoint 和 channel
2. **内存注册**：调用 `HcommProxy::MemReg` 注册内存到 HCCS endpoint
3. **数据传输**：通过 `ReadNbiOnThread`/`WriteNbiOnThread` 执行 HCCS 传输
4. **完成检查**：使用 `ChannelFence` 或轮询完成标志

---

## 6. 完成机制对比

### 6.1 ROCE 完成机制

```
+----------------+      RDMA Write/Read      +----------------+
|    CLIENT      | ------------------------> |    SERVER      |
|                |                           |                |
|  flag_queue_   | <------- RDMA Read -------|  flag_addr    |
|  [flag_index]  |   (after ChannelFence)    |  (value=1)    |
+----------------+                           +----------------+

流程:
1. 提交 Nbi RDMA 操作 (Write/Read)
2. ChannelFence 确保顺序
3. 最后一次 RDMA Read 读取远程完成标志
4. 轮询本地 flag_queue_[flag_index] == 1
5. 完成后释放 flag_index
```

### 6.2 UB 完成机制

```
+----------------+     AICPU Kernel      +----------------+
|    CLIENT      | --------------------> |    SERVER      |
|                |                        |                |
|  CompletePool  |                        |  remote_flag  |
|  - notify      | <-- Notify Signal ---- |  (written by  |
|  - host_flag   |                        |   kernel)     |
|  - stream      |                        |                |
+----------------+                        +----------------+

流程:
1. 从 CompletePool 获取 slot (含 notify/host_flag)
2. 将参数拷贝到设备内存
3. 启动 AICPU kernel (HixlBatchGet/Put)
4. Kernel 执行 UB 传输
5. Kernel 写入 notify
6. aclrtWaitAndResetNotify 等待
7. 异步 D2H 拷贝完成值到 host_flag
8. 轮询 host_flag == 1
9. 释放 slot 回 CompletePool
```

### 6.3 完成机制差异对比

| 方面 | ROCE | UB |
|------|------|-----|
| **完成存储** | Host 内存 (`flag_queue_`) | Device notify + host flag |
| **传输引擎** | RDMA (HCCL CPU engine) | AICPU kernel |
| **排序机制** | `ChannelFence` | Stream ordering |
| **槽位获取** | 栈式索引池 (4096) | CompletePool (128 slots) |
| **标志大小** | 8 bytes | 8 bytes |
| **最大并发** | 4096 | 128 |
| **等待方式** | 轮询 host 标志 | Notify wait + async D2H |

---

## 7. 性能对比

### 7.1 带宽对比

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        传输性能对比 (128MB 数据)                             │
└─────────────────────────────────────────────────────────────────────────────┘

带宽
│
│  120 GB/s ─┬───────────────────────────────────────── UB_D2D (芯片内)
│            │
│  100 GB/s ─┼──────────────────────────────────── HCCS (节点内跨芯片)
│            │
│   80 GB/s ─┼─────────────────────────────── UB_H2D/D2H (Host-Device)
│            │
│   50 GB/s ─┼──────────────────────── UB_H2H (Host-Host via UB)
│            │
│   22 GB/s ─┼────────────────── ROCE (跨节点 RDMA)
│            │
│            └──────────────────────────────────────────────►
│              协议类型
```

### 7.2 延迟对比

```
延迟
│
│    1 μs ─┬─────── UB_D2D
│         │
│    5 μs ─┼─────── UB_H2D/D2H
│         │
│   10 μs ─┼─────── HCCS
│         │
│   20 μs ─┼─────── UB_H2H
│         │
│   50 μs ─┼─────── ROCE (同数据中心)
│         │
│          └──────────────────────────────────────────────►
│            协议类型
```

---

## 8. 最佳实践

### 8.1 协议选择建议

| 场景 | 推荐协议 | 原因 |
|------|---------|------|
| **同芯片 D2D** | UB_D2D | 最高带宽、最低延迟 |
| **同节点跨芯片** | HCCS 或 UB | 取决于硬件配置，优先 HCCS |
| **同超节点跨节点** | ROCE 或 UB | 需要配置 net_instance_id |
| **跨超节点** | ROCE (强制) | 硬件限制，必须走网络 |
| **KV Cache 传输** | UB_D2D | LLM 推理场景最优 |
| **参数同步** | ROCE | 大模型训练跨节点 |

### 8.2 配置技巧

```bash
# 强制使用 ROCE (即使同超节点)
export HCCL_INTRA_ROCE_ENABLE=1

# 配置 net_instance_id (Endpoint JSON)
{
  "endpoint_id": "ep0",
  "protocol": "ub_ctp",
  "placement": "device",
  "plane": "data",
  "net_instance_id": "supernode_A"  # 相同值才能使用 UB
}
```

### 8.3 性能调优建议

1. **批量传输**：尽量使用批量 API，减少调用开销
2. **内存注册**：提前注册大块内存，避免频繁注册/注销
3. **异步传输**：使用 `TransferAsync` + `GetTransferStatus`，实现计算与传输并行
4. **CompletePool 大小**：根据并发传输数量调整，默认 128 slots
5. **ROCE MTU**：配置网络 MTU 为 9000 (Jumbo Frame) 提升吞吐

---

## 9. 关键源文件索引

| 文件路径 | 主要内容 |
|---------|---------|
| `src/hixl/engine/hixl_client.h` | CommType 枚举，Endpoint 匹配 |
| `src/hixl/engine/hixl_client.cc` | MustUseRoce, ClassifyTransfers, BatchTransfer |
| `src/hixl/cs/hixl_cs_client.h` | CompleteHandle, UbBatchArgs, CommunicateMem |
| `src/hixl/cs/hixl_cs_client.cc` | BatchTransferHost, BatchTransferDevice, LaunchUbAndStage |
| `src/hixl/cs/load_kernel.cc` | AICPU kernel 加载 |
| `src/hixl/cs/complete_pool.h` | CompletePool 单例设计 |
| `src/hixl/proxy/hcomm_proxy.h` | HCCL API 封装 |
| `src/hixl/proxy/hcomm/hcomm_res_defs.h` | 协议枚举，Endpoint 拓扑结构 |

---

> 文档生成日期：2026-06-16
> 
> 基于 HIXL 项目源码分析