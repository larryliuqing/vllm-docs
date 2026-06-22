# PD 分离测试场景分析

> 本文档详细分析 Prefill-Decode 分离架构在不同场景下的测试配置、硬件要求和网络连通性。

---

## 1. PD 分离架构概述

### 1.1 基本概念

PD 分离（Prefill-Decode Separation）是 LLM 推理的一种优化架构：

- **Prefill 阶段（P节点）**: 处理用户输入的 prompt，生成初始 KV Cache，计算密集型
- **Decode 阶段（D节点）**: 持续生成 token，读取 KV Cache，内存带宽密集型

分离架构的优势：
- 不同阶段可以使用不同规模的硬件
- Prefill 和 Decode 可以并行处理不同请求
- KV Cache 通过高速网络传输，减少延迟

### 1.2 数据流图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          PD 分离数据流                                       │
└─────────────────────────────────────────────────────────────────────────────┘

   用户请求
       │
       ▼
┌──────────────┐
│   Prefill    │  ─────────────────────────────────────────────
│    Server    │  │                                           │
│   (P节点)    │  │                                           │
└──────┬───────┘  │                                           │
       │          │                                           │
       │ 生成     │                                           │
       │ KV Cache │                                           │
       │          │                                           │
       ▼          │                                           │
┌──────────────┐  │           高速传输网络                     │
│  KV Cache    │  │         (HCCS/RoCE/RDMA)                   │
│   Buffer     │  │                                           │
└──────┬───────┘  │                                           │
       │          │                                           │
       │          │                                           │
       │ ─────────┼────────────────────────────────────────────┼───►
       │          │                                           │    KV Cache
       │          │                                           │    传输
       │          │                                           │
       │          │                                           ▼
       │          │                                    ┌──────────────┐
       │          │                                    │  KV Cache    │
       │          │                                    │   Buffer     │
       │          │                                    └──────┬───────┘
       │          │                                           │
       │          │                                           │ 读取
       │          │                                           │ KV Cache
       │          │                                           │
       │          │                                    ┌──────┴───────┐
       │          │                                    │    Decode    │
       │          │                                    │    Server    │
       │          │                                    │    (D节点)   │
       │          │                                    └──────┬───────┘
       │          │                                           │
       │          │                                           │
       │          │                                           │ 生成
       │          │                                           │ Output Token
       │          │                                           │
       │          │                                           ▼
       │          │                                    ┌──────────────┐
       │          │                                    │   Response   │
       │          │                                    │   to User    │
       │          │                                    └──────────────┘
       │          │
       │◄─────────┼───────────────────────────────────── (可选: 接收新请求)
       │          │
       ▼
┌──────────────┐
│   处理下一个 │
│   Prefill请求│
└──────────────┘
```

---

## 2. HIXL Server-Server D2D 测试流程

### 2.1 测试程序分析

基于 `examples/cpp/server_server_d2d.cpp` 的测试流程：

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Server-Server D2D 测试流程                        │
└─────────────────────────────────────────────────────────────────────┘

Server A (Prefill)                         Server B (Decode)
     │                                           │
     │  1. aclrtSetDevice(device_id)             │
     │     设置计算设备                           │
     │                                           │
     │  2. Hixl.Initialize(local_engine)         │
     │     初始化 HIXL 引擎                       │
     │     options["BufferPool"] = "0:0"         │
     │                                           │
     │  3. aclrtMalloc (device内存)              │
     │     分配设备内存                           │
     │                                           │
     │  4. RegisterMem(buffer, MEM_DEVICE)       │
     │     注册内存到 HIXL                        │
     │                                           │
     │  5. 保存地址到文件 (local_engine)          │
     │     文件名 = local_engine                  │
     │     内容 = buffer_addr buffer2_addr       │
     │                                           │
     │  6. sleep(5s) 等待对端注册                 │
     │                                           │
     │                    ◄───────────────────────│  对端同样步骤 1-6
     │                                           │
     │  7. Connect(remote_engine)                │
     │     建立到对端的连接                       │
     │     - 从文件读取对端地址                   │
     │                                           │
     │  8. TransferSync(WRITE)                   │
     │     buffer → remote_addr                  │
     │     写入数据到对端                         │
     │                                           │
     │  9. sleep(5s) 等待对端读取                 │
     │                                           │
     │                    ◄───────────────────────│  8. TransferSync(READ)
     │                                           │     remote_addr2 → buffer2
     │                                           │     从对端读取数据
     │                                           │
     │  10. Disconnect                           │
     │     断开连接                               │
     │                                           │
     │  11. DeregisterMem                        │
     │     注销内存                               │
     │                                           │
     │  12. Finalize                             │
     │     释放 HIXL 引擎                         │
     │                                           │
```

### 2.2 运行命令

```bash
# Server A (启动在 Device 0)
./server_server_d2d 0 127.0.0.1:26000 127.0.0.1:26001

# Server B (启动在 Device 1)
./server_server_d2d 1 127.0.0.1:26001 127.0.0.1:26000
```

参数说明：
- `0/1`: 逻辑设备 ID
- `127.0.0.1:26000`: local_engine (本端监听地址)
- `127.0.0.1:26001`: remote_engine (对端地址)

---

## 3. 测试场景分类

### 3.1 场景分类总览

| 场景类型 | 描述 | 通信方式 | 适用情况 |
|---------|------|---------|---------|
| **场景1** | 同节点内多卡 Ascend | HCCS/UB | Atlas训练服务器，内部HCCS连接 |
| **场景2** | 跨节点 Ascend 同构 | RoCE | 多台Ascend服务器，光纤连接 |
| **场景3** | Ascend+NVIDIA 异构 | RDMA/TCP+Mooncake | 混合硬件环境 |

---

## 4. 场景1: 同节点内多卡 Ascend

### 4.1 硬件拓扑

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                   场景1: Ascend 同节点内多卡                                 │
└─────────────────────────────────────────────────────────────────────────────┘

硬件拓扑:
┌──────────────────────────────────────────────┐
│         物理服务器 (Atlas 训练服务器)          │
│                                              │
│   ┌─────┐  HCCS  ┌─────┐                    │
│   │ NPU │◄──────►│ NPU │                    │
│   │ P卡 │        │ D卡 │                    │
│   │(Prefill)│    │(Decode)│                  │
│   │ Device 4│    │ Device 5│                 │
│   └──┬──┘        └──┬──┘                    │
│      │              │                        │
│      │   PCIe       │                        │
│      ▼              ▼                        │
│   ┌──────────────────────┐                  │
│   │      Host CPU        │                  │
│   │   (Linux OS)         │                  │
│   └──────────────────────┘                  │
│                                              │
│   内部互联:                                  │
│   - HCCS: 芯片间高速通信 (~80-100 GB/s)      │
│   - UB: 芯片内通信 (~119 GB/s)               │
│   - PCIe: Host-Device通信                   │
└──────────────────────────────────────────────┘
```

### 4.2 通信方式选择

| 方式 | 带宽 | 延迟 | 适用范围 |
|------|------|------|---------|
| **HCCS** | ~80-100 GB/s | ~5-10 μs | 同节点内跨芯片 |
| **UB_D2D** | ~119 GB/s | ~1-5 μs | 同芯片内或超节点内 |
| **RoCE (强制)** | ~22 GB/s | ~10-50 μs | 需要光口连接 |

**推荐**: 同节点内使用 HCCS/UB，无需外部网络连线。

### 4.3 配置要求

#### 4.3.1 HCCS 链路检查

```bash
# 检查 HCCS 状态
source /usr/local/Ascend/cann/set_env.sh
npu-smi info -t hccs -i 4 -c 0

# 期望输出:
#   hccs health status             : OK        # 必须是 OK
#   hccs lane mode                 : [x x x x x x x]  # 非全0
#   hccs link lane list            : [xxxx xxxx xxxx ...]  # 有激活lane
#   hccs link speed                : [xxx xxx xxx ...]  # 有速度值
```

如果 `hccs health status: NOK`，说明 HCCS 链路不正常，无法进行同节点通信。

#### 4.3.2 Device IP 配置

```bash
# /etc/hccn.conf 配置
# 每个设备的 IP 必须唯一

cat /etc/hccn.conf
# 期望内容:
#   address_4=192.168.1.4
#   address_5=192.168.1.5
#   address_6=192.168.1.6
#   address_7=192.168.1.7

# 注意: IP 不需要实际可达，仅用于 rank table 生成
# 但必须唯一，不能重复
```

#### 4.3.3 环境变量配置

```bash
# 禁用 RoCE，使用 HCCS/PCIe
export HCCL_INTRA_ROCE_ENABLE=0
export HCCL_INTRA_PCIE_ENABLE=1

# 可选: 打印传输详细信息
export ASCEND_TRANSPORT_PRINT=1
```

#### 4.3.4 HIXL 初始化配置

```cpp
// Server A (Prefill, Device 0)
std::map<AscendString, AscendString> options;
options["BufferPool"] = "0:0";
Hixl hixl;
hixl.Initialize("127.0.0.1:26000", options);  // local_engine

// Server B (Decode, Device 1)
std::map<AscendString, AscendString> options;
options["BufferPool"] = "0:0";
Hixl hixl;
hixl.Initialize("127.0.0.1:26001", options);  // local_engine
```

### 4.4 网络连通性要求

| 要求项 | 状态 | 说明 |
|--------|------|------|
| **HCCS 链路** | 必须OK | 物理连接在服务器内部 |
| **外部网络** | 不需要 | 无需光纤/以太网连线 |
| **IP 地址** | 配置唯一 | `/etc/hccn.conf` 中每个设备IP唯一 |
| **Engine Address** | 可用127.0.0.1 | 同节点内可用本地回环地址 |
| **端口** | 不冲突 | 两端使用不同端口 (26000 vs 26001) |

### 4.5 启动示例

```bash
# 设置环境变量
export HCCL_INTRA_ROCE_ENABLE=0
export HCCL_INTRA_PCIE_ENABLE=1
source /usr/local/Ascend/cann/set_env.sh

# 编译测试程序
cd /home/bes/work/vllm-project/hixl/examples/cpp
bash compile_test.sh

# 启动 Server A (Prefill, Device 0)
./server_server_d2d 0 127.0.0.1:26000 127.0.0.1:26001 &
sleep 5

# 启动 Server B (Decode, Device 1)
./server_server_d2d 1 127.0.0.1:26001 127.0.0.1:26000 &

# 等待测试完成
wait
```

### 4.6 常见问题与排查

| 问题 | 可能原因 | 解决方案 |
|------|---------|---------|
| `HcclCommPrepare fail` | HCCS链路NOK | 检查 `npu-smi info -t hccs` |
| `IP repeated error` | `/etc/hccn.conf` IP重复 | 配置唯一IP |
| `RegisterMem fail` | BufferPool未配置 | 添加 `options["BufferPool"] = "0:0"` |
| `Connect timeout` | 端口冲突或对端未启动 | 检查端口和对端进程 |

---

## 5. 场景2: 跨节点 Ascend 同构

### 5.1 hardware拓扑

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                   场景2: Ascend 跨节点通信                                   │
└─────────────────────────────────────────────────────────────────────────────┘

硬件拓扑:
┌────────────────────┐          RoCE           ┌────────────────────┐
│   Server A         │◄───────────────────────►│   Server B         │
│   (Prefill节点)     │      光纤/以太网        │   (Decode节点)      │
│                    │                          │                    │
│   ┌─────┐          │                          │   ┌─────┐          │
│   │ NPU │          │                          │   │ NPU │          │
│   │Ascend│          │      RoCE Link          │   │Ascend│          │
│   │ Device 0│       │◄──────────────────────►│   │ Device 0│       │
│   └──┬──┘          │        (光纤)           │   └──┬──┘          │
│      │             │                          │      │             │
│      │ PCIe        │                          │      │ PCIe        │
│      ▼             │                          │      ▼             │
│   ┌──────┐         │                          │   ┌──────┐         │
│   │ Host │         │                          │   │ Host │         │
│   │ CPU  │         │                          │   │ CPU  │         │
│   └──────┘         │                          │   └──────┘         │
│                    │                          │                    │
│   光口IP:          │                          │   光口IP:          │
│   192.168.1.4      │                          │   192.168.2.4      │
│                    │                          │                    │
│   Host IP:         │                          │   Host IP:         │
│   192.168.1.10     │                          │   192.168.2.10     │
└────────────────────┘                          └────────────────────┘

         ┌────────────────────────────────────┐
         │        交换机/网络设备              │
         │                                    │
         │  支持 RoCEv2 或 IB 网络            │
         │  MTU 建议 9000 (Jumbo Frame)       │
         └────────────────────────────────────┘
```

### 5.2 通信方式选择

| 方式 | 带宽 | 延迟 | 适用范围 |
|------|------|------|---------|
| **RoCE** | ~22 GB/s | ~10-50 μs | 跨节点通信（必须） |
| **HCCS** | 不适用 | 不适用 | 仅同节点内 |

**跨节点必须使用 RoCE**，无法使用 HCCS/UB。

### 5.3 配置要求

#### 5.3.1 RoCE 光口检查

```bash
# 检查光口链路状态
source /usr/local/Ascend/cann/set_env.sh
hccn_tool -i 0 -link -g

# 期望输出:
#   link status: UP        # 必须是 UP，DOWN表示未接线

# 检查所有设备
for i in 0 1 2 3; do
    echo "=== Device $i ==="
    hccn_tool -i $i -link -g
done
```

#### 5.3.2 Device IP 配置

```bash
# Server A: /etc/hccn.conf
address_0=192.168.1.0    # 物理可达的IP
address_1=192.168.1.1
address_2=192.168.1.2
address_3=192.168.1.3

# Server B: /etc/hccn.conf
address_0=192.168.2.0    # 不同网段或不同IP
address_1=192.168.2.1
address_2=192.168.2.2
address_3=192.168.2.3

# IP 必须真实可达，可以通过 ping 测试
ping 192.168.2.0    # 从Server A ping Server B的NPU IP
```

#### 5.3.3 网络连通性测试

```bash
# 测试 IP 可达性
ping <peer_npu_ip>

# 测试端口可达性 (需要先启动对端)
nc -zv <peer_host_ip> 26000

# 检查 MTU 配置
ip link show | grep mtu
# 建议 MTU 9000 (Jumbo Frame)
```

#### 5.3.4 环境变量配置

```bash
# 强制使用 RoCE (跨节点默认使用RoCE，可显式设置)
export HCCL_INTRA_ROCE_ENABLE=1

# 打印传输详细信息
export ASCEND_TRANSPORT_PRINT=1

# 可选: 设置超时时间
export HCCL_CONNECT_TIMEOUT=600  # 秒
```

#### 5.3.5 HIXL 初始化配置

```cpp
// Server A (Prefill节点)
std::map<AscendString, AscendString> options;
options["BufferPool"] = "0:0";
Hixl hixl;
hixl.Initialize("192.168.1.10:26000", options);  // 本端 Host IP

// 连接到 Server B
hixl.Connect("192.168.2.10:26001");  // 对端 Host IP

// Server B (Decode节点)
std::map<AscendString, AscendString> options;
options["BufferPool"] = "0:0";
Hixl hixl;
hixl.Initialize("192.168.2.10:26001", options);  // 本端 Host IP

// 连接到 Server A
hixl.Connect("192.168.1.10:26000");  // 对端 Host IP
```

### 5.4 网络连通性要求

| 要求项 | 状态 | 说明 |
|--------|------|------|
| **RoCE 光口** | 必须UP | 光纤连接两个节点的光口 |
| **IP 可达** | 必须 | ping 测试通过 |
| **防火墙** | 开放端口 | 26000-26001 端口开放 |
| **MTU** | 建议9000 | Jumbo Frame 提升吞吐 |
| **交换机** | 支持RoCEv2 | 或 IB 网络 |

### 5.5 启动示例

```bash
# === Server A (Prefill节点) ===

# 设置环境变量
export HCCL_INTRA_ROCE_ENABLE=1
export ASCEND_TRANSPORT_PRINT=1
source /usr/local/Ascend/cann/set_env.sh

# 编译测试程序
cd /path/to/hixl/examples/cpp
bash compile_test.sh

# 启动 Server A
./server_server_d2d 0 192.168.1.10:26000 192.168.2.10:26001 &
sleep 10  # 等待对端启动


# === Server B (Decode节点) ===

# 设置环境变量
export HCCL_INTRA_ROCE_ENABLE=1
export ASCEND_TRANSPORT_PRINT=1
source /usr/local/Ascend/cann/set_env.sh

# 启动 Server B
./server_server_d2d 0 192.168.2.10:26001 192.168.1.10:26000 &
```

### 5.6 常见问题与排查

| 问题 | 可能原因 | 解决方案 |
|------|---------|---------|
| `link status: DOWN` | 光纤未接线 | 检查光纤连接 |
| `Connect timeout` | IP不可达或防火墙阻挡 | ping测试，检查防火墙 |
| `RoCE error` | MTU配置不当 | 设置MTU 9000 |
| `Channel create fail` | 网络不支持RoCEv2 | 检查交换机配置 |

---

## 6. 场景3: Ascend+NVIDIA 异构节点

### 6.1 硬件拓扑

```
┌─────────────────────────────────────────────────────────────────────────────┐
│           场景3: Ascend + NVIDIA 异构节点                                    │
└─────────────────────────────────────────────────────────────────────────────┘

硬件拓扑:
┌────────────────────┐          网络          ┌────────────────────┐
│   Server A         │◄─────────────────────►│   Server B         │
│   (Ascend Prefill) │      RDMA/TCP         │   (NVIDIA Decode)  │
│                    │                       │                    │
│   ┌─────┐          │                       │   ┌─────┐          │
│   │ NPU │          │                       │   │ GPU │          │
│   │Ascend│          │                       │   │NVIDIA│         │
│   │A2/A3 │          │                       │   │A100/H100│       │
│   └──┬──┘          │                       │   └──┬──┘          │
│      │             │                       │      │             │
│      │ PCIe        │                       │      │ PCIe        │
│      ▼             │                       │      ▼             │
│   ┌──────┐         │                       │   ┌──────┐         │
│   │ Host │         │                       │   │ Host │         │
│   │ CPU  │         │                       │   │ CPU  │         │
│   └──────┘         │                       │   └──────┘         │
│                    │                       │                    │
│   HIXL Engine      │                       │   Mooncake         │
│   KV Cache传输     │                       │   Transfer Engine │
│                    │                       │                    │
│   RoCE NIC         │◄─────────────────────►│   IB/RoCE NIC      │
│   或普通网卡        │      RDMA/TCP         │   或普通网卡        │
│                    │                       │                    │
│   IP: 10.0.0.1     │                       │   IP: 10.0.0.2     │
└────────────────────┘                       └────────────────────┘

通信协议栈:
┌────────────────────────────────────────────────────────────────────────────┐
│                                                                            │
│   Ascend 端                          NVIDIA 端                              │
│   ┌─────────────────┐               ┌─────────────────┐                   │
│   │ vLLM Prefill    │               │ vLLM Decode     │                   │
│   └────────┬────────┘               └────────┬────────┘                   │
│            │                                  │                            │
│            ▼                                  ▼                            │
│   ┌─────────────────┐               ┌─────────────────┐                   │
│   │ HIXL Engine     │               │ Mooncake Store  │                   │
│   │ (KV Transfer)   │               │ (KV Transfer)   │                   │
│   └────────┬────────┘               └────────┬────────┘                   │
│            │                                  │                            │
│            ▼                                  ▼                            │
│   ┌─────────────────┐               ┌─────────────────┐                   │
│   │ Mooncake        │◄─────────────►│ Mooncake        │                   │
│   │ Ascend Store    │               │ NVIDIA Store    │                   │
│   └────────┬────────┘               └────────┬────────┘                   │
│            │                                  │                            │
│            ▼                                  ▼                            │
│   ┌─────────────────┐               ┌─────────────────┐                   │
│   │ Transport       │               │ Transport       │                   │
│   │ (RDMA/TCP)      │               │ (RDMA/TCP)      │                   │
│   └────────┬────────┘               └────────┬────────┘                   │
│            │                                  │                            │
│            ▼                                  ▼                            │
│   ┌─────────────────┐               ┌─────────────────┐                   │
│   │ Network         │               │ Network         │                   │
│   │ (RoCE/IB/TCP)   │               │ (RoCE/IB/TCP)   │                   │
│   └─────────────────┘               └─────────────────┘                   │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

### 6.2 通信方式选择

| 方式 | 带宽 | 延迟 | 适用范围 | 要求 |
|------|------|------|---------|------|
| **RDMA (GDR)** | ~10-20 GB/s | ~10-50 μs | GPU-Direct RDMA | 两端都有RDMA NIC |
| **TCP** | ~1-5 GB/s | ~100-500 μs | 标准TCP网络 | 无特殊要求 |

**异构节点必须使用 Mooncake 或类似的传输框架**，HIXL 本身不支持直接与 NVIDIA GPU 通信。

### 6.3 Mooncake 方案配置

#### 6.3.1 架构说明

Mooncake 是一个异构 KV Cache 传输框架，支持：
- Ascend NPU ↔ NVIDIA GPU
- RDMA 传输 (高性能)
- TCP 传输 (通用性)

参考文件: `examples/third_parties/mooncake_store/`

#### 6.3.2 Ascend 端配置

```python
# Ascend Prefill 端配置
# config_ascend_prefill.yaml

server:
  host: "10.0.0.1"
  port: 26000

storage:
  type: "ascend"
  device_id: 0
  allocator: "aclrtMalloc"

transfer:
  type: "rdma"  # 或 "tcp"
  # RDMA 配置
  rdma_device: "roce0"
  rdma_port: 1
  # TCP 配置 (如果使用TCP)
  # tcp_port: 26001

buffer_pool:
  size: 1073741824  # 1GB
```

#### 6.3.3 NVIDIA 端配置

```python
# NVIDIA Decode 端配置
# config_nvidia_decode.yaml

server:
  host: "10.0.0.2"
  port: 26001

storage:
  type: "nvidia"
  device_id: 0
  allocator: "cudaMalloc"

transfer:
  type: "rdma"  # 或 "tcp"
  # RDMA 配置 (GPUDirect RDMA)
  rdma_device: "ib0"
  rdma_port: 1
  gdr_enable: true
  # TCP 配置
  # tcp_port: 26002

buffer_pool:
  size: 1073741824  # 1GB
```

#### 6.3.4 RDMA 网络要求

```bash
# Ascend 端: RoCE 配置
# 检查 RoCE 链路
hccn_tool -i 0 -link -g  # link status: UP

# NVIDIA 端: IB/RoCE 配置
# 检查 IB 设备
ibv_devinfo
# 期望输出:
#   hca_id: mlx5_0
#   link_layer: Ethernet (RoCE) 或 Infiniband
#   state: PORT_ACTIVE

# 检查 GDR 支持
cat /proc/driver/nvidia/gdr_support
# 或
nvidia-smi -q | grep -i "gpudirect"
```

#### 6.3.5 TCP 网络要求

```bash
# 测试 TCP 连通性
ping 10.0.0.2  # 从 Ascend 端 ping NVIDIA 端

# 测试端口可达性
nc -zv 10.0.0.2 26001

# 检查防火墙
iptables -L | grep 26001
# 或
firewall-cmd --list-ports
```

### 6.4 网络连通性要求对比

| 方式 | Ascend端要求 | NVIDIA端要求 | 网络要求 |
|------|-------------|-------------|---------|
| **RDMA** | RoCE NIC, 光口UP | IB/RoCE NIC, GDR支持 | RDMA网络，同一网段或路由 |
| **TCP** | 普通网卡 | 普通网卡 | TCP/IP网络，端口开放 |

### 6.5 启动示例

```bash
# === Ascend 端 (Prefill) ===

# 设置环境变量
source /usr/local/Ascend/cann/set_env.sh
export MOONCAKE_ASCEND_ENABLE=1

# 启动 Mooncake Ascend Store
python mooncake_ascend_store.py --config config_ascend_prefill.yaml &
sleep 5

# 启动 vLLM Prefill
python -m vllm.entrypoints.api_server \
    --model /path/to/model \
    --device ascend \
    --kv-transfer mooncake \
    --mooncake-config config_ascend_prefill.yaml


# === NVIDIA 端 (Decode) ===

# 设置环境变量
export MOONCAKE_NVIDIA_ENABLE=1
export CUDA_VISIBLE_DEVICES=0

# 启动 Mooncake NVIDIA Store
python mooncake_nvidia_store.py --config config_nvidia_decode.yaml &
sleep 5

# 启动 vLLM Decode
python -m vllm.entrypoints.api_server \
    --model /path/to/model \
    --device cuda \
    --kv-transfer mooncake \
    --mooncake-config config_nvidia_decode.yaml \
    --decode-only
```

### 6.6 常见问题与排查

| 问题 | 可能原因 | 解决方案 |
|------|---------|---------|
| `GDR not supported` | NVIDIA驱动不支持GDR | 更新驱动，启用GDR |
| `RDMA connect fail` | RDMA NIC配置不当 | 检查IB/RoCE设备状态 |
| `Transfer slow` | 使用TCP而非RDMA | 配置RDMA网络 |
| `Ascend Store init fail` | ACL初始化失败 | 检查CANN环境 |

---

## 7. 三种场景对比总结

### 7.1 硬件要求对比

| 场景 | P节点硬件 | D节点硬件 | 内部互联 | 跨节点网络 |
|------|---------|---------|---------|-----------|
| **场景1** | Ascend NPU | Ascend NPU | HCCS (必须OK) | 不需要 |
| **场景2** | Ascend NPU | Ascend NPU | 不适用 | RoCE光纤 (必须UP) |
| **场景3** | Ascend NPU | NVIDIA GPU | 不适用 | RDMA或TCP |

### 7.2 网络连通性对比

| 场景 | 链路状态要求 | IP要求 | 端口要求 | 特殊要求 |
|------|-------------|--------|---------|---------|
| **场景1** | HCCS OK | `/etc/hccn.conf`唯一IP | 不同端口 | 无外部网络 |
| **场景2** | RoCE UP | 真实可达IP | 防火墙开放 | MTU 9000 |
| **场景3 (RDMA)** | 两端RDMA UP | 真实可达IP | RDMA端口 | GDR支持 |
| **场景3 (TCP)** | TCP可达 | 真实可达IP | TCP端口 | 无特殊 |

### 7.3 性能对比

| 场景 | 通信方式 | 带宽 | 延迟 | KV Cache传输时间 (1GB) |
|------|---------|------|------|----------------------|
| **场景1** | HCCS | 80-100 GB/s | 5-10 μs | ~10-12 ms |
| **场景1** | UB_D2D | 119 GB/s | 1-5 μs | ~8 ms |
| **场景2** | RoCE | 22 GB/s | 10-50 μs | ~45 ms |
| **场景3 (RDMA)** | GDR | 10-20 GB/s | 10-50 μs | ~50-100 ms |
| **场景3 (TCP)** | TCP | 1-5 GB/s | 100-500 μs | ~200-1000 ms |

### 7.4 配置复杂度对比

| 场景 | 配置项数量 | 配置难度 | 排查难度 |
|------|---------|---------|---------|
| **场景1** | 少 | 低 | 低 |
| **场景2** | 中 | 中 | 中 |
| **场景3** | 多 | 高 | 高 |

---

## 8. 测试前检查清单

### 8.1 场景1 检查清单

```bash
# 1. 检查 HCCS 状态
npu-smi info -t hccs -i 4 -c 0 | grep "hccs health status"
# 期望: hccs health status : OK

# 2. 检查设备 IP 配置
cat /etc/hccn.conf
# 期望: 每个设备 IP 不同

# 3. 检查 CANN 环境
source /usr/local/Ascend/cann/set_env.sh
npu-smi info
# 期望: 设备列表正常

# 4. 设置环境变量
export HCCL_INTRA_ROCE_ENABLE=0
export HCCL_INTRA_PCIE_ENABLE=1

# 5. 编译测试程序
cd /path/to/hixl/examples/cpp
bash compile_test.sh
# 期望: 编译成功，生成 server_server_d2d
```

### 8.2 场景2 检查清单

```bash
# 1. 检查 RoCE 光口状态
hccn_tool -i 0 -link -g
# 期望: link status: UP

# 2. 检查 IP 可达性
ping <peer_npu_ip>
# 期望: ping 成功

# 3. 检查防火墙
iptables -L | grep 26000
# 或开放端口
firewall-cmd --add-port=26000-26001/tcp

# 4. 检查 MTU
ip link show | grep mtu
# 建议: mtu 9000

# 5. 设置环境变量
export HCCL_INTRA_ROCE_ENABLE=1

# 6. 两端都编译测试程序
bash compile_test.sh
```

### 8.3 场景3 检查清单

```bash
# === Ascend 端 ===

# 1. 检查 RoCE 状态
hccn_tool -i 0 -link -g  # UP

# 2. 检查 Mooncake 环境
pip show mooncake-transfer-engine

# === NVIDIA 端 ===

# 1. 检查 CUDA 环境
nvidia-smi

# 2. 检查 IB/RoCE 设备
ibv_devinfo  # state: PORT_ACTIVE

# 3. 检查 GDR 支持
nvidia-smi -q | grep -i "gpudirect"

# 4. 检查 Mooncake 环境
pip show mooncake-transfer-engine

# === 网络 ===

# 1. 测试连通性
ping <peer_ip>

# 2. 测试 RDMA (如果使用RDMA)
rping -s -a <peer_ip> -v  # Server端
rping -c -a <peer_ip> -v  # Client端
```

---

## 9. 故障排查指南

### 9.1 HIXL 相关错误

| 错误信息 | 可能原因 | 排查步骤 |
|---------|---------|---------|
| `Initialize failed` | ACL初始化失败 | 检查CANN环境，`aclrtSetDevice` |
| `RegisterMem failed` | 内存注册失败 | 检查`BufferPool`配置，内存地址 |
| `Connect failed` | 建链失败 | 检查网络，端口，对端状态 |
| `TransferSync failed` | 传输失败 | 检查内存地址，Channel状态 |

### 9.2 HCCL 相关错误

| 错误码 | 含义 | 排查步骤 |
|--------|------|---------|
| `103900` | IP重复 | 检查`/etc/hccn.conf`，确保IP唯一 |
| `103905` | HcclCommPrepare失败 | 检查HCCS状态或RoCE链路 |
| `507899` | 内存注册失败 | 检查BufferPool配置 |

### 9.3 网络相关错误

| 现象 | 可能原因 | 排查步骤 |
|------|---------|---------|
| `link status: DOWN` | 光纤未接线 | 检查光纤物理连接 |
| `ping不通` | IP不可达 | 检查IP配置，路由，防火墙 |
| `端口不通` | 防火墙阻挡 | `iptables -L`，开放端口 |
| `RDMA连接失败` | RDMA配置不当 | `ibv_devinfo`，检查RDMA设备 |

---

## 10. 附录

### 10.1 关键命令速查

```bash
# HCCS 状态检查
npu-smi info -t hccs -i <device_id> -c 0

# RoCE 链路检查
hccn_tool -i <device_id> -link -g

# 设备列表
npu-smi info -l

# IP 配置查看
cat /etc/hccn.conf

# 网络连通测试
ping <ip>
nc -zv <ip> <port>

# RDMA 设备检查 (NVIDIA端)
ibv_devinfo

# GPU 状态检查 (NVIDIA端)
nvidia-smi
```

### 10.2 环境变量速查

```bash
# 同节点内通信 (HCCS/PCIe)
export HCCL_INTRA_ROCE_ENABLE=0
export HCCL_INTRA_PCIE_ENABLE=1

# 跨节点通信 (RoCE)
export HCCL_INTRA_ROCE_ENABLE=1

# 详细日志
export ASCEND_TRANSPORT_PRINT=1

# 超时设置
export HCCL_CONNECT_TIMEOUT=600
```

### 10.3 参考文档

| 文档 | 路径 |
|------|------|
| HIXL Architecture | `/home/bes/work/vllm-project/hixl/docs/design/HIXL_Engine_Architecture.md` |
| Transport Protocols | `/home/bes/work/vllm-project/hixl/docs/design/Transport_Protocols_Detailed.md` |
| Server-Server D2D Example | `/home/bes/work/vllm-project/hixl/examples/cpp/server_server_d2d.cpp` |
| Mooncake Example | `/home/bes/work/vllm-project/hixl/examples/third_parties/mooncake_store/` |

### 10.4 当前测试环境状态

> **重要**：以下是基于实际硬件检查的结果，用于确定可行的测试场景。

| 检查项 | 状态 | 检查命令 |
|--------|------|----------|
| **运行环境** | 虚拟机 (VM) | `npu-smi info -t topo` 返回 "cannot be executed on a VM" |
| **HCCS Health** | NOK (不健康) | `npu-smi info -t hccs -i 4 -c 0` → "hccs health status: NOK" |
| **HCCS Lane Mode** | 全0 (无激活链路) | 同上 → "hccs lane mode: [0 0 0 0 0 0 0]" |
| **RoCE Link Status** | DOWN (光口未接线) | `hccn_tool -i 4 -link -g` → "link status: DOWN" |
| **Device IP 配置** | 已设置 | `/etc/hccn.conf` → address_4=192.168.1.4 |

**当前环境限制：**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     当前测试环境限制分析                                      │
└─────────────────────────────────────────────────────────────────────────────┘

硬件检查结果:
┌──────────────────────────────────────────────┐
│         虚拟机环境 (VM)                       │
│                                              │
│   ┌─────┐   HCCS    ┌─────┐                 │
│   │ NPU │  ───────► │ NPU │                 │
│   │  4  │   NOK     │  5  │                 │
│   └─────┘           └─────┘                 │
│                                              │
│   RoCE 光口状态:                              │
│   ┌─────┐   光纤    ┌─────┐                 │
│   │ NPU │  ───────► │ NPU │                 │
│   │  4  │   DOWN    │  5  │                 │
│   │     │  (未接线)  │     │                 │
│   └─────┘           └─────┘                 │
│                                              │
│   结论: 无法进行实际NPU间通信测试              │
└──────────────────────────────────────────────┘
```

**可行的测试场景（基于当前状态）：**

| 场景 | 可行性 | 说明 |
|------|--------|------|
| 场景1: 同节点内 HCCS | ❌ 不可行 | HCCS health NOK，需要物理服务器 |
| 场景1: 同节点内 RoCE | ❌ 不可行 | 光口 DOWN，需要插光纤 |
| 场景2: 跨节点 RoCE | ❌ 不可行 | 光口 DOWN，需要光纤连接 |
| 场景3: Ascend+NVIDIA | ❌ 不可行 | 无可用网络链路 |

**要使测试可行，需要：**

1. **插好 RoCE 光纤**：连接 NPU 光口，使链路 UP
   ```bash
   # 检查命令
   hccn_tool -i 4 -link -g  # 应返回 "link status: UP"
   ```

2. **或者使用物理 Atlas 训练服务器**：确保 HCCS 链路正常
   ```bash
   # 检查命令
   npu-smi info -t hccs -i 4 -c 0  # 应返回 "hccs health status: OK"
   ```

---

> 文档生成日期：2026-06-18
>
> 基于 HIXL 项目和测试环境分析