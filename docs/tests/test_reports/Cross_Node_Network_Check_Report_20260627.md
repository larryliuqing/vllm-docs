# 跨节点 PD 分离测试 — 双节点网络与硬件检查报告

**检查日期**: 2026-06-27
**检查节点**: node01 (192.168.0.190) / node02 (192.168.0.193)
**操作者**: la (普通用户) → 通过 `ssh root@node02` 访问 node02

---

## 1. 检查方法说明

所有检查命令均从 node01（当前工作节点）发起，通过 SSH 远程执行：

| 检查项 | 命令 | 用途 |
|--------|------|------|
| NPU 状态 | `npu-smi info` | 确认 8 卡 Health=OK, HBM 正常 |
| 片内互联拓扑 | `npu-smi info -t topo -i 0` | 确认 NPU 间直连方式 (HCCS / PCIe) |
| HCCS 链路详情 | `npu-smi info -t hccs -i 0 -c 0` | 确认 lane 数、速率、健康状态 |
| HCCS IP | `hccn_tool -i <N> -ip -g` | 确认每个 NPU 的 HCCS 通信 IP |
| 片内互联带宽 | `npu-smi info -t hccs-bw -i 0 -c 0 -time 200` | 测试 HCCS 链路实时带宽 |
| 物理网卡 | `ip addr show` | 确认管理网卡 IP 和状态 |
| RDMA 链路 | `rdma link` | 确认 RoCE 链路 UP/DOWN |
| 跨节点 ping | `ping <对方管理IP>` | 确认跨节点网络连通性和延迟 |
| **NPU ping NPU** | `hccn_tool -i <src> -ping -g address <dst_ip> pkt 64` | 从 NPU 侧测试 HCCS 连通性和延迟 |
| **ARP 表** | `hccn_tool -i <N> -arp -g` | 查看 NPU 的 ARP 表，确认可达的 HCCS IP |
| **路由表** | `hccn_tool -i <N> -route -g` | 查看 NPU 内部路由 |
| HCCL 配置 | `cat /etc/hccn.conf` | 确认 HCCS 网络参数 |
| Docker 镜像 | `docker images vllm-ascend` | 确认推理镜像存在 |
| Docker Runtime | `cat /etc/docker/daemon.json` | 确认 Ascend Docker Runtime 配置 |
| 系统资源 | `free -h`, `nproc`, `npu-smi info -m` | 确认 CPU/内存/NPU 映射 |
| Meta 接口 | `ip addr show Meta` | 检查 HCCL 运行时 TUN 设备 |

### 关键命令说明

```bash
# NPU 之间 ping（节点内或跨节点）
hccn_tool -i 0 -ping -g address 10.0.0.21 pkt 64

# 参数解释：
#   -i 0          从 NPU0 发起 ping
#   -ping -g      获取 ping 结果（-g = get）
#   address X.X.X.X   目标 IP
#   pkt 64        包大小（字节）

# 查看 NPU 的 ARP 表
hccn_tool -i 0 -arp -g

# 查看 NPU 的路由表
hccn_tool -i 0 -route -g
```

---

## 2. 硬件配置概览

| 项目 | node01 | node02 |
|------|--------|--------|
| **主机名** | master | ascend |
| **OS** | openEuler 22.03 (LTS-SP4) | openEuler 22.03 (LTS-SP4) |
| **内核** | 5.10.0-216.0.0.115 | 5.10.0-318.0.0.221 |
| **CPU 核数** | 192 | 192 |
| **总内存** | 2.0 TiB | 2.0 TiB |
| **NPU 型号** | Ascend 910B4 × 8 | Ascend 910B4 × 8 |
| **NPU HBM** | 32 GiB / card | 32 GiB / card |
| **NPU 拓扑** | V1 芯片 | V1 芯片 |
| **Docker Runtime** | ascend-docker-runtime | ascend-docker-runtime |
| **推理镜像** | vllm-ascend:v0.20.2rc (18GB) | vllm-ascend:v0.20.2rc (18GB) |

> 注：CANN 9.0.0 和驱动 25.5.2 均在容器内，宿主机侧无 CANN 安装。

---

## 3. 节点内 NPU 互联检查

### 3.1 NPU 拓扑

两节点拓扑完全一致，8 NPU **全互联 (HCCS)**：

```
       NPU0  NPU1  NPU2  NPU3  NPU4  NPU5  NPU6  NPU7  CPU Affinity
NPU0    X    HCCS  HCCS  HCCS  HCCS  HCCS  HCCS  HCCS   144-167
NPU1  HCCS    X    HCCS  HCCS  HCCS  HCCS  HCCS  HCCS   144-167
NPU2  HCCS  HCCS    X    HCCS  HCCS  HCCS  HCCS  HCCS    96-119
NPU3  HCCS  HCCS  HCCS    X    HCCS  HCCS  HCCS  HCCS    96-119
NPU4  HCCS  HCCS  HCCS  HCCS    X    HCCS  HCCS  HCCS     0-23
NPU5  HCCS  HCCS  HCCS  HCCS  HCCS    X    HCCS  HCCS     0-23
NPU6  HCCS  HCCS  HCCS  HCCS  HCCS  HCCS    X    HCCS    48-71
NPU7  HCCS  HCCS  HCCS  HCCS  HCCS  HCCS  HCCS    X      48-71
```

**结论**: 无 PCIe/SMP/PIX/PXB 路径，8 NPU 之间 **全部通过 HCCS 直连**，无 PCIe 瓶颈。

### 3.2 HCCS 链路参数

| 参数 | node01 | node02 |
|------|--------|--------|
| **健康状态** | OK | OK |
| **Lane 模式** | 4 (每链路) | 4 (每链路) |
| **Lane 列表** | 1111 (全部 UP) | 1111 (全部 UP) |
| **链路速率** | 224 Gbps / lane | 224 Gbps / lane |
| **单对带宽** | ~896 Gbps 双向 | ~896 Gbps 双向 |
| **重传计数** | 0 | 0 |
| **错误计数** | 0 | 0 |

### 3.3 HCCS IP 配置

| NPU | node01 | node02 |
|-----|--------|--------|
| 0 | 10.0.0.11/24 | 10.0.0.21/24 |
| 1 | 10.0.0.12/24 | 10.0.0.22/24 |
| 2 | 10.0.0.13/24 | 10.0.0.23/24 |
| 3 | 10.0.0.14/24 | 10.0.0.24/24 |
| 4 | 10.0.0.15/24 | 10.0.0.25/24 |
| 5 | 10.0.0.16/24 | 10.0.0.26/24 |
| 6 | 10.0.0.17/24 | 10.0.0.27/24 |
| 7 | 10.0.0.18/24 | 10.0.0.28/24 |
| **子网** | 10.0.0.0/24 | 10.0.0.0/24 |
| **网关** | 10.0.0.1 | 10.0.0.1 |

> 两节点 HCCS IP 在同一子网（10.0.0.0/24），共用网关 10.0.0.1，这为跨节点直接互联提供了条件。

### 3.4 NPU 路由表

两节点路由表一致（从 `hccn_tool -i 0 -route -g` 获取）：

```
Destination     Gateway         Genmask         Flags Iface
default         10.0.0.1        0.0.0.0         UG    eth0
10.0.0.0        *               255.255.255.0   U     eth0
127.0.0.1       *               255.255.255.255 UH    lo
192.168.1.0     *               255.255.255.0   U     end0v0
192.168.2.0     *               255.255.255.0   U     end0v0
```

**说明**: `eth0` 是 NPU 内部的 HCCS 网络接口，`end0v0` 是 NPU 内部的另一个逻辑接口。

---

## 4. 节点内 NPU 间连通性测试（核心测试）

在 NPU 层面通过 `hccn_tool -ping` 测试 HCCS 网络连通性。**与宿主机的 `ping` 不同，`hccn_tool -ping` 是 NPU 内部的 ICMP 测试，走的是 HCCS 直连通道。**

### 4.1 node01：NPU0 → 本节点所有 NPU

```bash
hccn_tool -i 0 -ping -g address 10.0.0.12 pkt 64   # → NPU1
hccn_tool -i 0 -ping -g address 10.0.0.18 pkt 64   # → NPU7
```

| 源 → 目标 | 结果 | 平均延迟 |
|-----------|------|---------|
| NPU0 → 10.0.0.11 (自己) | ✅ 3 received, 0% loss | ~0.09 ms |
| NPU0 → 10.0.0.12 (NPU1) | ✅ 3 received, 0% loss | ~0.10 ms |
| NPU0 → 10.0.0.13 (NPU2) | ✅ 3 received, 0% loss | ~0.08 ms |
| NPU0 → 10.0.0.14 (NPU3) | ✅ 3 received, 0% loss | ~0.08 ms |
| NPU0 → 10.0.0.15 (NPU4) | ✅ 3 received, 0% loss | ~0.09 ms |
| NPU0 → 10.0.0.16 (NPU5) | ✅ 3 received, 0% loss | ~0.08 ms |
| NPU0 → 10.0.0.17 (NPU6) | ✅ 3 received, 0% loss | ~0.09 ms |
| NPU0 → 10.0.0.18 (NPU7) | ✅ 3 received, 0% loss | ~0.09 ms |

**节点内 HCCS 延迟**: **0.046~0.190 ms**，首包稍高（~0.18ms），后续稳定在 **~0.05ms**。

### 4.2 node01：NPU7 → 本节点所有 NPU（反向验证）

```bash
hccn_tool -i 7 -ping -g address 10.0.0.11 pkt 64   # NPU7 → NPU0
```

| 源 → 目标 | 结果 |
|-----------|------|
| NPU7 → 10.0.0.11 (NPU0) | ✅ 3 received, 0% loss |
| NPU7 → 10.0.0.12 (NPU1) | ✅ 3 received, 0% loss |
| NPU7 → 10.0.0.13 (NPU2) | ✅ 3 received, 0% loss |
| NPU7 → 10.0.0.14 (NPU3) | ✅ 3 received, 0% loss |
| NPU7 → 10.0.0.15 (NPU4) | ✅ 3 received, 0% loss |
| NPU7 → 10.0.0.16 (NPU5) | ✅ 3 received, 0% loss |
| NPU7 → 10.0.0.17 (NPU6) | ✅ 3 received, 0% loss |
| NPU7 → 10.0.0.18 (自己) | ✅ 3 received, 0% loss |

### 4.3 node02：NPU0 → 本节点所有 NPU

```bash
hccn_tool -i 0 -ping -g address 10.0.0.22 pkt 64   # → NPU1
hccn_tool -i 0 -ping -g address 10.0.0.28 pkt 64   # → NPU7
```

| 源 → 目标 | 结果 |
|-----------|------|
| NPU0 → 10.0.0.21 (自己) | ✅ 3 received, 0% loss |
| NPU0 → 10.0.0.22 (NPU1) | ✅ 3 received, 0% loss |
| NPU0 → 10.0.0.23 (NPU2) | ✅ 3 received, 0% loss |
| NPU0 → 10.0.0.24 (NPU3) | ✅ 3 received, 0% loss |
| NPU0 → 10.0.0.25 (NPU4) | ✅ 3 received, 0% loss |
| NPU0 → 10.0.0.26 (NPU5) | ✅ 3 received, 0% loss |
| NPU0 → 10.0.0.27 (NPU6) | ✅ 3 received, 0% loss |
| NPU0 → 10.0.0.28 (NPU7) | ✅ 3 received, 0% loss |

### 4.4 节点内小结

| 维度 | 结果 |
|------|------|
| 节点内 NPU 全连通 | ✅ 所有组合 0% loss |
| HCCS 延迟 (稳定态) | ~0.05 ms |
| HCCS 首包延迟 | ~0.18 ms |
| 链路健康 | ✅ OK, retry=0, error=0 |

---

## 5. 跨节点连通性测试（核心测试）

### 5.1 管理网络 Ping

```bash
# node01 → node02
ping 192.168.0.193
# node02 → node01（从 node02 SSH 执行）
ping 192.168.0.190
```

| 方向 | 最小 | 平均 | 最大 |
|------|------|------|------|
| node01 → node02 | 0.064 ms | 0.080 ms | 0.097 ms |
| node02 → node01 | 0.097 ms | 0.110 ms | 0.124 ms |

### 5.2 RDMA / RoCE 链路

```bash
# node01
rdma link         → link hns_0/1 state ACTIVE netdev enp189s0f0
# node02
rdma link         → link rocep189s0f0/1 state ACTIVE netdev enp189s0f0
```

| 参数 | node01 | node02 |
|------|--------|--------|
| **RDMA 设备** | hns_0/1 | rocep189s0f0/1 |
| **链路状态** | ACTIVE, LINK_UP | ACTIVE, LINK_UP |
| **关联网卡** | enp189s0f0 (管理口, 192.168.0.190) | enp189s0f0 (管理口, 192.168.0.193) |
| **MTU** | 9000 (Meta 接口) | 1500 (管理口) |

> 两节点 RDMA 设备名称不同（`hns_0` vs `rocep189s0f0`），但均关联 enp189s0f0，不影响跨节点 HCCL。

### 5.3 HCCS 跨节点 Ping

**node01 NPU → node02 NPU**，走 HCCS 网络（10.0.0.0/24 子网直连）：

```bash
# node01 NPU0 → node02 NPU0
hccn_tool -i 0 -ping -g address 10.0.0.21 pkt 64
```

| 源 | 目标 | 结果 | 平均延迟 |
|----|------|------|---------|
| node01 NPU0 → node02 NPU0 (10.0.0.21) | ✅ 3 received, 0% loss | ~0.09 ms |
| node01 NPU0 → node02 NPU1 (10.0.0.22) | ✅ 3 received, 0% loss | ~0.09 ms |
| node01 NPU0 → node02 NPU2 (10.0.0.23) | ✅ 3 received, 0% loss | ~0.09 ms |
| node01 NPU0 → node02 NPU3 (10.0.0.24) | ✅ 3 received, 0% loss | ~0.09 ms |
| node01 NPU0 → node02 NPU4 (10.0.0.25) | ✅ 3 received, 0% loss | ~0.09 ms |
| node01 NPU0 → node02 NPU5 (10.0.0.26) | ✅ 3 received, 0% loss | ~0.09 ms |
| node01 NPU0 → node02 NPU6 (10.0.0.27) | ✅ 3 received, 0% loss | ~0.09 ms |
| node01 NPU0 → node02 NPU7 (10.0.0.28) | ✅ 3 received, 0% loss | ~0.09 ms |

**node02 NPU → node01 NPU**（反向验证）：

```bash
# node02 NPU0 → node01 NPU0
hccn_tool -i 0 -ping -g address 10.0.0.11 pkt 64
```

| 源 | 目标 | 结果 |
|----|------|------|
| node02 NPU0 → node01 NPU0 (10.0.0.11) | ✅ 3 received, 0% loss |
| node02 NPU0 → node01 NPU1 (10.0.0.12) | ✅ 3 received, 0% loss |
| node02 NPU0 → node01 NPU2 (10.0.0.13) | ✅ 3 received, 0% loss |
| node02 NPU0 → node01 NPU3 (10.0.0.14) | ✅ 3 received, 0% loss |
| node02 NPU0 → node01 NPU4 (10.0.0.15) | ✅ 3 received, 0% loss |
| node02 NPU0 → node01 NPU5 (10.0.0.16) | ✅ 3 received, 0% loss |
| node02 NPU0 → node01 NPU6 (10.0.0.17) | ✅ 3 received, 0% loss |
| node02 NPU0 → node01 NPU7 (10.0.0.18) | ✅ 3 received, 0% loss |

**跨节点 HCCS 延迟**: **0.025~0.161 ms**，首包约 0.14ms，稳定态 **~0.03ms**。

### 5.4 ARP 表分析

从 `hccn_tool -i 0 -arp -g` 获取的 NPU0 ARP 表：

**node01 NPU0 ARP 表**（16 条记录）：
| IP | MAC | 归属 |
|----|-----|------|
| 10.0.0.1 | 7c:33:f9:04:6f:23 | 网关 |
| 10.0.0.12 | 0c:4f:9b:b5:ca:75 | 本节点 NPU1 ✅ |
| 10.0.0.13 | 0c:4f:9b:b5:ca:56 | 本节点 NPU2 ✅ |
| 10.0.0.14 | 0c:4f:9b:b5:ca:33 | 本节点 NPU3 ✅ |
| 10.0.0.15 | 0c:4f:9b:b5:ca:28 | 本节点 NPU4 ✅ |
| 10.0.0.16 | 0c:4f:9b:b5:ca:7b | 本节点 NPU5 ✅ |
| 10.0.0.17 | 0c:4f:9b:b5:c1:5c | 本节点 NPU6 ✅ |
| 10.0.0.18 | 0c:4f:9b:b5:ca:2c | 本节点 NPU7 ✅ |
| 10.0.0.21 | 0c:4f:9b:b5:ca:4d | node02 NPU0 ✅ |
| 10.0.0.22 | 0c:4f:9b:b5:ca:55 | node02 NPU1 ✅ |
| 10.0.0.23 | 0c:4f:9b:b5:ca:32 | node02 NPU2 ✅ |
| 10.0.0.24 | 0c:4f:9b:b5:ca:30 | node02 NPU3 ✅ |
| 10.0.0.25 | e4:82:10:50:0d:57 | node02 NPU4 ✅ |
| 10.0.0.26 | e4:82:10:50:0d:4b | node02 NPU5 ✅ |
| 10.0.0.27 | e4:82:10:50:0d:76 | node02 NPU6 ✅ |
| 10.0.0.28 | e4:82:10:50:0d:7a | node02 NPU7 ✅ |

**node02 NPU0 ARP 表**（16 条记录）：
同样包含本节点所有 NPU（10.0.0.21-28）和 node01 所有 NPU（10.0.0.11-18）。

**关键发现**: 两节点 NPU 的 ARP 表都包含了对方节点所有 8 个 NPU 的 MAC 地址，说明 **HCCS 网络在二层（L2）层面就已经互联互通**，跨节点通信不需要三层路由。

### 5.5 MAC 地址归属分析

| MAC 前缀 | 设备类型 |
|----------|---------|
| `0c:4f:9b:b5:ca:*` | node01 NPU（内部 HCCS 网口） |
| `0c:4f:9b:b5:c1:*` | node01 NPU6（特殊 MAC） |
| `e4:82:10:50:0d:*` | node02 NPU（内部 HCCS 网口） |
| `0c:4f:9b:b5:ca:74` | node01 NPU0 自身 MAC |
| `7c:33:f9:04:6f:23` | 交换机/网关 |
| `7c:33:f9:04:6f:21` | 上联交换机（LLDP 信息） |

> 两节点 NPU 的 MAC 分别来自不同的 OUI 段：
> - node01: `0c:4f:9b` → Huawei
> - node02: `e4:82:10` → Huawei
> 这是同一设备内不同 HCCS 网口的正常差异。

---

## 6. 跨节点网络拓扑总图

```
┌─────────────────────────────────────────────────────────────────┐
│  交换机 (Switch) 10.0.0.1 / 192.168.0.1                        │
│  ┌─ LLDP: Huawei XH9110-24BQ8DQ, 200GE1/0/8                  ─┘
│  │
│  ├── 10.0.0.0/24 ── HCCS 网络 (NPU 间内部通信)
│  └── 192.168.0.0/24 ── 管理网络 (RoCE + TCP)
│
├──────────────────────────────┬──────────────────────────────┐
│  node01 (192.168.0.190)     │  node02 (192.168.0.193)      │
│  master                     │  ascend                      │
│                              │                              │
│  ┌─ HCCS ──────────────┐    │  ┌─ HCCS ──────────────┐    │
│  │ NPU0  10.0.0.11     │    │  │ NPU0  10.0.0.21     │    │
│  │ NPU1  10.0.0.12     │    │  │ NPU1  10.0.0.22     │    │
│  │ NPU2  10.0.0.13     │←━━━→│  │ NPU2  10.0.0.23     │    │
│  │ NPU3  10.0.0.14     │     │  │ NPU3  10.0.0.24     │    │
│  │ NPU4  10.0.0.15     │0.03ms│  │ NPU4  10.0.0.25     │    │
│  │ NPU5  10.0.0.16     │L2直连│  │ NPU5  10.0.0.26     │    │
│  │ NPU6  10.0.0.17     │     │  │ NPU6  10.0.0.27     │    │
│  │ NPU7  10.0.0.18     │     │  │ NPU7  10.0.0.28     │    │
│  └─────────────────────┘    │  └─────────────────────┘    │
│                              │                              │
│  enp189s0f0 (RoCE RDMA)     │  enp189s0f0 (RoCE RDMA)     │
└──────────────────────────────┴──────────────────────────────┘
```

**关键总结：**

| 通信路径 | 协议 | 延迟 | 带宽 |
|----------|------|------|------|
| 节点内 NPU↔NPU | HCCS (直连) | ~0.05 ms | 896 Gbps / 链路 |
| 跨节点 NPU↔NPU | HCCS (L2 桥接) | ~0.03~0.09 ms | 未知（HCCL 自动选择） |
| 跨节点 管理口 | RoCE | ~0.08 ms | 1 Gbps (管理口) |
| 跨节点 管理口 | TCP | ~0.08 ms | 1 Gbps (管理口) |

---

## 7. 综合结论

### ✅ 双节点跨节点 PD 分离部署条件验证通过

| 检查项 | 结果 | 影响 |
|--------|------|------|
| 8 NPU 健康状态 | ✅ 全部 OK | 可用 16 卡 |
| NPU 拓扑 (HCCS 全互联) | ✅ 无 PCIe 瓶颈 | 节点内通信最优 |
| HCCS 链路 (4 lane × 224 Gbps) | ✅ 全部 UP | 节点内带宽充足 |
| node01 HCCS IP 配置 | ✅ 10.0.0.11-18 | 正常 |
| node02 HCCS IP 配置 | ✅ 10.0.0.21-28 | ✅ **已配置完成** |
| 节点内 NPU↔NPU ping | ✅ 0% loss, ~0.05ms | 节点内 HCCS 正常 |
| 跨节点 NPU↔NPU ping | ✅ 0% loss, ~0.03-0.09ms | **跨节点 HCCS 二层互通** |
| ARP 表互相可见 | ✅ 16/16 完整 | HCCS L2 桥接正常 |
| RDMA RoCE | ✅ ACTIVE | 跨节点 HCCL 可用 |
| 管理网络 | ✅ 0.08ms | 控制面通信正常 |
| Docker 镜像 | ✅ vllm-ascend:v0.20.2rc | 两节点一致 |
| Ascend Docker Runtime | ✅ 已配置 | 容器可挂载 NPU |

### ⚠️ 注意事项

1. **管理口带宽限制**: enp189s0f0 协商速度为 **1000Mb/s (1 Gbps)**，对于大规模 KV cache 跨节点传输可能是瓶颈。HCCS 跨节点（10.0.0.x）走的是独立通道，不受此限制。
2. **RoCE 与 HCCS 双路径**: Ascend HCCL 会**自动选择最优路径**，跨节点通信会优先使用 HCCS（10.0.0.x 直连）而非 RoCE 管理口。
3. **推荐部署方案**:
   - node01 (Prefill): NPU 0-3, TP=4
   - node02 (Decode): NPU 4-7, TP=4
   - 环境变量: `HCCL_SOCKET_IFNAME=enp189s0f0`, `HCCL_INTRA_ROCE_ENABLE=1`

---

## 8. 原始数据

### 8.1 关键命令输出

```bash
# ─── NPU IP 查看 ───
# node01
$ hccn_tool -i 0 -ip -g
ipaddr:10.0.0.11 / netmask:255.255.255.0
# node02
$ hccn_tool -i 0 -ip -g
ipaddr:10.0.0.21 / netmask:255.255.255.0

# ─── NPU 间 ping ───
# node01 NPU0 → node02 NPU0
$ hccn_tool -i 0 -ping -g address 10.0.0.21 pkt 64
device 0 PING 10.0.0.21
recv seq=0,time=0.181000ms
recv seq=1,time=0.055000ms
recv seq=2,time=0.048000ms
3 packets transmitted, 3 received, 0.00% packet loss

# node01 NPU0 → 本节点 NPU1
$ hccn_tool -i 0 -ping -g address 10.0.0.12 pkt 64
device 0 PING 10.0.0.12
recv seq=0,time=0.190000ms
recv seq=1,time=0.053000ms
recv seq=2,time=0.046000ms
3 packets transmitted, 3 received, 0.00% packet loss

# node02 NPU0 → node01 NPU0（反向验证）
$ hccn_tool -i 0 -ping -g address 10.0.0.11 pkt 64
recv seq=0,time=0.161000ms
recv seq=1,time=0.099000ms
recv seq=2,time=0.036000ms

# ─── NPU 路由表 ───
$ hccn_tool -i 0 -route -g
Destination     Gateway         Genmask         Flags Iface
default         10.0.0.1        0.0.0.0         UG    eth0
10.0.0.0        *               255.255.255.0   U     eth0

# ─── NPU ARP 表 ───
$ hccn_tool -i 0 -arp -g
(10.0.0.12) at 0c:4f:9b:b5:ca:75 [ether]  on eth0  # node01 NPU1
(10.0.0.21) at 0c:4f:9b:b5:ca:4d [ether]  on eth0  # node02 NPU0
...

# ─── RDMA 链路 ───
# node01
$ rdma link
link hns_0/1 state ACTIVE physical_state LINK_UP netdev enp189s0f0
# node02
$ rdma link
link rocep189s0f0/1 state ACTIVE physical_state LINK_UP netdev enp189s0f0

# ─── 管理口网速 ───
$ ethtool enp189s0f0 | grep Speed
Speed: 1000Mb/s
```

---

**报告生成时间**: 2026-06-27  
**基于工具**: `npu-smi`, `hccn_tool`, `hccn_tool -ping`, `hccn_tool -arp`, `hccn_tool -route`, `rdma`, `ping`, `ip`, `ethtool`, `docker`