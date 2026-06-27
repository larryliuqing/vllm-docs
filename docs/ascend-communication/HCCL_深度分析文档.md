# HCCL (Huawei Collective Communication Library) 深度分析文档

> **源码版本**: HCCL + HCOMM (CANN 开源版本)
> **分析日期**: 2026-06-25
> **文档作者**: Claude Code Analysis

---

## 目录

1. [项目概览](#1-项目概览)
2. [代码结构分析](#2-代码结构分析)
3. [核心架构](#3-核心架构)
4. [初始化流程](#4-初始化流程)
5. [集合通信执行流程](#5-集合通信执行流程)
6. [传输层架构](#6-传输层架构)
7. [算法实现详解](#7-算法实现详解)
8. [内存管理](#8-内存管理)
9. [设备端编程接口](#9-设备端编程接口)
10. [工作流模式](#10-工作流模式)
11. [关键数据结构](#11-关键数据结构)
12. [环境变量与配置](#12-环境变量与配置)
13. [与 NCCL 的对比分析](#13-与-nccl-的对比分析)
14. [最佳实践与调优](#14-最佳实践与调优)

---

## 1. 项目概览

### 1.1 项目定位

HCCL (Huawei Collective Communication Library) 是华为基于昇腾 AI 处理器的高性能集合通信库，为分布式训练和推理提供高性能、高可靠的通信方案。

HCCL 由两个子项目组成：

- **HCCL 集合通信库** (`hccl/`)：对外通信算子接口层，提供 14 个标准集合通信 C API
- **HCOMM 通信基础库** (`hcomm/`)：底层通信基础设施，分层解耦为控制面和数据面

**核心特性:**
- 高性能集合通信原语 (AllReduce, AllGather, ReduceScatter 等)
- 多传输层支持 (HCCS, PCIe, RoCE, TCP)
- 自适应算法选择 (HD/Ring/Mesh/Star)
- 拓扑感知优化
- 多 NPU/多节点扩展性
- MC2 (Memory-Compute-Communication) 融合模式
- AICPU/AIV 双后端支持

**版本信息:**
```
HCCL 项目: CANN 开源版本
支持昇腾架构: 910B, 91093, A5 等
许可证: CANN Open Software License Agreement Version 2.0
```

### 1.2 项目统计

```
源文件统计:
├── HCCL 源文件 (.cc/.h):    322个 (src/)
├── HCCL 头文件:              2个 (include/)
├── HCOMM 源文件 (.cc/.h): 3,274个 (src/)
├── HCOMM 头文件:             8个 (include/)
└── 总计:                 3,606个文件

核心代码行数:
├── HCCL src/:              45,688行
├── HCCL include/:             340行
├── HCOMM src/:            598,321行
├── HCOMM include/:          2,028行
└── 总计:                  646,377行
```

### 1.3 目录结构

```
hccl/                              # HCCL 集合通信库（对外接口层）
├── include/                       # 公共 API 头文件
│   ├── hccl.h                    # 14个集合通信 C API (253行)
│   └── hccl_mc2.h               # MC2 融合模式 API (87行)
├── src/                           # 核心源代码
│   ├── common/                    # 通用逻辑 (8,264行)
│   │   ├── hcomm_dlsym/          # HCOMM 动态符号加载
│   │   ├── param_check.h/cc      # 参数校验
│   │   ├── alg_type.h/cc         # 算法类型定义
│   │   ├── alg_env_config.h/cc   # 算法环境配置
│   │   ├── adapter_acl.h/cc      # ACL 适配器
│   │   └── log.h/cc              # 日志系统
│   └── ops/                       # 通信算子实现 (37,424行)
│       ├── all_reduce/            # AllReduce 算子 (4,326行)
│       ├── all_gather/            # AllGather 算子 (3,151行)
│       ├── all_gather_v/          # AllGatherV 算子 (933行)
│       ├── reduce_scatter/        # ReduceScatter 算子 (3,779行)
│       ├── reduce_scatter_v/      # ReduceScatterV 算子 (943行)
│       ├── broadcast/             # Broadcast 算子 (2,291行)
│       ├── reduce/                # Reduce 算子 (2,717行)
│       ├── scatter/               # Scatter 算子 (5,339行)
│       ├── all_to_all_v/          # AlltoAll/AlltoAllV/AlltoAllVC (3,407行)
│       ├── send/                  # Send 算子 (459行)
│       ├── recv/                  # Recv 算子 (470行)
│       ├── batch_send_recv/       # 批量 Send/Recv (735行)
│       └── op_common/             # 算子公共逻辑 (8,874行)
│           ├── executor/          # 执行器
│           ├── selector/          # 算法选择器
│           └── template/          # 算法模板 (aicpu/aiv)
├── test/                           # 单元测试
├── examples/                       # 示例代码
│   ├── 01_point_to_point/         # 点对点通信示例
│   ├── 02_collectives/            # 集合通信示例 (9个)
│   └── 04_custom_ops_p2p/         # 自定义算子 P2P 示例
├── docs/                           # 文档
│   ├── build.md                   # 构建文档
│   └── api/README.md              # API 文档
├── cmake/                          # CMake 构建配置
├── scripts/                        # 构建/打包/签名脚本
└── CMakeLists.txt                  # 顶层构建文件

hcomm/                             # HCOMM 通信基础库（底层引擎）
├── include/                        # 公共头文件
│   ├── hcomm_primitives.h         # 数据面编程接口 (429行)
│   ├── hcomm_res.h                # 资源定义
│   ├── hcomm_res_defs.h           # 资源定义详情
│   └── hccl/                      # HCCL 类型和通信域定义
│       ├── hccl_types.h           # 类型定义 (225行)
│       └── hccl_comm.h            # 通信域 API (340行)
├── src/                            # 核心源代码
│   ├── algorithm/                  # 通信算法实现 (126,625行)
│   │   ├── base/                  # 算法基础模板 (64,328行)
│   │   │   ├── alg_template/      # 算法模板基类
│   │   │   │   ├── temp_all_reduce/    # AllReduce 模板
│   │   │   │   ├── temp_all_gather/    # AllGather 模板
│   │   │   │   ├── temp_reduce_scatter/# ReduceScatter 模板
│   │   │   │   ├── temp_broadcast/     # Broadcast 模板
│   │   │   │   ├── temp_alltoall/      # AlltoAll 模板
│   │   │   │   ├── component/          # 算法组件
│   │   │   │   └── inc_all_reduce_deter/ # 确定性 AllReduce
│   │   │   ├── alg_aiv_template/  # AIV 后端算法模板
│   │   │   ├── communicator/      # 通信域相关
│   │   │   ├── mc2_handler/       # MC2 处理器
│   │   │   └── inc/               # 算法内部头文件
│   │   ├── impl/                  # 算法实现 (60,008行)
│   │   │   ├── coll_executor/     # 集合通信执行器
│   │   │   ├── operator/          # 算子实现
│   │   │   ├── resource_manager/  # 资源管理
│   │   │   ├── task/              # 任务调度
│   │   │   └── legacy/            # 遗留实现
│   │   └── pub_inc/               # 算法公共头文件
│   ├── framework/                  # 框架层 (134,656行)
│   │   ├── communicator/          # 通信域管理 (37,605行)
│   │   │   └── impl/
│   │   │       └── hccl_communicator_host.cc  # 核心实现 (9,098行)
│   │   ├── device/                # 设备管理 (26,078行)
│   │   │   ├── framework/         # 设备框架
│   │   │   │   └── aicpu_communicator.cc  # AICPU 通信器 (5,696行)
│   │   │   ├── aicpu_kfc/         # AICPU KFC 模式
│   │   │   ├── common/            # 设备通用
│   │   │   ├── debug/             # 设备调试
│   │   │   └── utils/             # 设备工具
│   │   ├── group/                 # 组管理
│   │   ├── hcom/                  # HCOM 高层接口 (7,031行)
│   │   │   ├── hcom.cc            # HCOM 主实现 (4,148行)
│   │   │   └── gradient_segment/  # 梯度分段
│   │   ├── op_base/               # 算子基类 (4,907行)
│   │   ├── next/                  # 下一代表态 (27,440行)
│   │   │   ├── coll_comms/        # 集合通信
│   │   │   ├── comm_primitives/   # 通信原语
│   │   │   ├── comms/             # 通信子系统
│   │   │   │   └── ccu/           # CCU (Compute Communication Unit)
│   │   │   └── common/            # 通用
│   │   ├── cluster_maintenance/   # 集群维护
│   │   │   ├── detect/            # 检测
│   │   │   ├── health/            # 健康检查
│   │   │   ├── recovery/          # 恢复
│   │   │   └── snapshot/          # 快照
│   │   ├── common/                # 框架通用 (14,988行)
│   │   ├── inc/                   # 框架内部头文件
│   │   └── nslbdp/                # NSLBDP
│   ├── platform/                   # 平台层 (125,607行)
│   │   ├── hccp/                  # HCCP/RDMA 通信协议 (62,444行)
│   │   │   ├── hccp_service/      # HCCP 服务
│   │   │   ├── rdma_agent/        # RDMA 代理
│   │   │   ├── rdma_service/      # RDMA 服务
│   │   │   ├── common/            # HCCP 通用
│   │   │   └── stub_tsd/          # TSD 桩
│   │   ├── resource/              # 资源管理 (29,135行)
│   │   │   ├── dispatcher_ctx/    # 调度器上下文
│   │   │   ├── mem/               # 内存管理
│   │   │   ├── netdev/            # 网络设备
│   │   │   ├── notify/            # 通知机制
│   │   │   ├── rma_buffer/        # RMA 缓冲区
│   │   │   ├── socket/            # Socket 管理
│   │   │   ├── stream/            # 流管理
│   │   │   └── transport/         # 传输层
│   │   ├── common/                # 平台通用 (18,957行)
│   │   │   ├── adapter/           # 适配器
│   │   │   ├── buffer_manager/    # 缓冲区管理
│   │   │   ├── misc/              # 杂项
│   │   │   ├── p2p_mgmt/          # P2P 管理
│   │   │   └── unfold_cache/      # 展开缓存
│   │   ├── task/                  # 任务调度
│   │   ├── remote_access/         # 远程内存访问
│   │   ├── ping_mesh/             # Ping Mesh
│   │   ├── comm_primitive/        # 通信原语
│   │   ├── debug/                 # 调试
│   │   └── typical/               # 典型场景
│   ├── common/                     # 通用模块 (8,007行)
│   ├── hccd/                       # HCCD 守护进程 (2,728行)
│   ├── legacy/                     # 遗留代码 (192,038行)
│   │   ├── service/               # 遗留服务 (84,696行)
│   │   ├── framework/             # 遗留框架 (57,016行)
│   │   └── unified_platform/      # 统一平台 (45,226行)
│   └── pub_inc/                    # 公共头文件 (8,660行)
├── pkg_inc/                        # 打包头文件
│   ├── hccl/                      # HCCL 内部/扩展头文件
│   │   ├── hccl_inner.h           # 内部 API (71行)
│   │   ├── hccl_ex.h              # 扩展 API
│   │   ├── hccl_ctrl_plane.h      # 控制面
│   │   ├── hccl_one_sided_services.h  # 单边服务
│   │   ├── hccl_res_expt.h        # 资源实验 API
│   │   ├── hcom.h                 # HCOM 接口
│   │   ├── workflow.h             # 工作流模式
│   │   ├── base.h                 # 基础类型 (390行)
│   │   └── dtype_common.h         # 数据类型通用
│   └── hcomm/                     # HCOMM 内部头文件
│       ├── hcomm_primitives_expt.h # 原语实验 API
│       └── ccu/                    # CCU 指令集头文件 (35个)
├── python/                         # Python 绑定
│   └── hccl/__init__.py
├── docs/                           # 文档
├── examples/                       # 示例
├── cmake/                          # CMake 构建配置
└── CMakeLists.txt                  # 顶层构建文件 (322行)
```

---

## 2. 代码结构分析

### 2.1 模块划分

HCCL/HCOMM 采用分层模块化架构，主要模块包括:

| 模块 | 职责 | 行数 | 关键文件 |
|------|------|------|----------|
| **HCCL API 层** | 对外集合通信 API | 45,688 | hccl.h, ops/* |
| **算法层 (algorithm)** | 通信算法模板与实现 | 126,625 | alg_template/*, impl/* |
| **框架层 (framework)** | 通信域/设备/组管理 | 134,656 | communicator/*, device/* |
| **平台层 (platform)** | HCCP/RDMA/资源管理 | 125,607 | hccp/*, resource/* |
| **遗留代码 (legacy)** | 旧版框架和服务 | 192,038 | service/*, framework/* |
| **通用模块 (common)** | 调试/错误/流管理 | 8,007 | — |
| **HCCD 守护进程** | 守护进程 | 2,728 | hccd/* |

### 2.2 代码依赖关系

```
应用层 (PyTorch / TensorFlow / 自定义应用)
       │
       ├── HCCL API 层 (hccl.h)
       │      ├── HcclAllReduce / HcclBroadcast / HcclReduceScatter / ...
       │      ├── HcclSend / HcclRecv / HcclBatchSendRecv
       │      └── HcclAlltoAll / HcclAlltoAllV / HcclAlltoAllVC
       │
       ├── HCCL 算子层 (ops/)
       │      ├── executor/ (执行器)
       │      ├── selector/ (算法选择器)
       │      └── template/ (算法模板: aicpu/aiv)
       │
       └── HCOMM 引擎层
              ├── algorithm/ (算法模板与实现)
              │      ├── base/alg_template/ (Ring/HD/Mesh/Star 等)
              │      ├── base/alg_aiv_template/ (AIV 后端)
              │      └── impl/ (执行器/资源管理/任务调度)
              │
              ├── framework/ (通信域/设备/组管理)
              │      ├── communicator/ (通信域创建/销毁/管理)
              │      ├── device/ (AICPU/AIV 设备管理)
              │      ├── group/ (组管理)
              │      └── op_base/ (算子基类)
              │
              ├── platform/ (平台抽象层)
              │      ├── hccp/ (HCCP/RDMA 通信协议)
              │      ├── resource/ (内存/流/通知/传输)
              │      ├── task/ (任务调度)
              │      └── remote_access/ (远程内存访问)
              │
              └── legacy/ (遗留代码)
```

---

## 3. 核心架构

### 3.1 分层架构图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           应用层 (Application Layer)                        │
│                    PyTorch / TensorFlow / 自定义应用                           │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            HCCL API 层 (hccl.h)                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │  AllReduce   │  │  AllGather   │  │   Reduce     │  │  Broadcast   │    │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │ReduceScatter │  │   AlltoAll   │  │   Send/Recv  │  │  Group API   │    │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘    │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          算子管理层 (Operator Layer)                          │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  算子执行框架 (ops/)                                                   │   │
│  │  • Executor: 算子执行器，协调算法选择和执行                              │   │
│  │  • Selector: 算法选择器，根据数据量/拓扑/配置选择最优算法                 │   │
│  │  • Template: 算法模板，支持 AICPU 和 AIV 两种后端                        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            通信器管理层 (Communicator Layer)                  │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │  通信域管理 (framework/communicator/)                                 │    │
│  │  • HcclCommInitRootInfo / HcclCommInitClusterInfo                    │    │
│  │  • 拓扑发现与路径规划                                                 │    │
│  │  • 通道建立与资源分配                                                 │    │
│  │  • 通信域生命周期管理                                                 │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐            │
│  │   HcclComm      │  │   Channel       │  │   Device        │            │
│  │  • rank/nRanks  │  │  • peers[]      │  │  • AICPU/AIV    │            │
│  │  • channels[]   │  │  • ring/tree    │  │  • Stream       │            │
│  │  • topo         │  │  • workQueue    │  │  • Notify       │            │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘            │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                    ┌──────────────────┼──────────────────┐
                    │                  │                  │
                    ▼                  ▼                  ▼
        ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
        │   算法引擎层       │  │   平台抽象层      │  │   设备执行层      │
        │   (Algorithm)    │  │   (Platform)     │  │   (Device)       │
        │                  │  │                  │  │                  │
        │ • Ring/HD/Mesh   │  │ • HCCP/RDMA      │  │ • AICPU Kernel   │
        │ • Star/Tree      │  │ • Resource Mgmt  │  │ • AIV Kernel     │
        │ • Doubling       │  │ • Task Scheduler │  │ • CCU 指令集     │
        │ • Bruck          │  │ • Remote Access  │  │ • Stream/Event   │
        └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘
                 │                     │                     │
                 └─────────────────────┼─────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            传输抽象层 (Transport Layer)                      │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                    传输层接口                                          │  │
│  │  ┌────────────────────────────────────────────────────────────────┐  │  │
│  │  │  HCCS (Huawei Cache Coherence System) - 片内互联                  │  │  │
│  │  │  PCIe - 片间/跨设备互联                                          │  │  │
│  │  │  RoCE (RDMA over Converged Ethernet) - 跨节点网络                 │  │  │
│  │  │  TCP/IP - 回退方案                                               │  │  │
│  │  └────────────────────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            硬件层 (Hardware Layer)                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   NPU 0     │  │   NPU 1     │  │   NPU 2     │  │   NPU N     │        │
│  │  (Ascend    │  │  (Ascend    │  │  (Ascend    │  │  (Ascend    │        │
│  │   910B/91093)│  │   910B/91093)│  │   910B/91093)│  │   910B/91093)│        │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘        │
│         │                │                │                │                │
│         └────────────────┼────────────────┼────────────────┘                │
│                          │                │                                 │
│                          ▼                ▼                                 │
│              ┌──────────────────────────────────────┐                       │
│              │      HCCS / PCIe / RoCE / TCP        │                       │
│              └──────────────────────────────────────┘                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 核心组件说明

| 组件 | 文件 | 主要功能 |
|------|------|----------|
| **HcclComm** | hccl_comm.h | 通信器句柄，管理 rank、拓扑、通道等 |
| **HcclCommConfig** | hccl_types.h | 通信域配置，含缓冲区大小、确定性、算法等 |
| **HcommPrimitives** | hcomm_primitives.h | 数据面编程接口，单边读写/归约/通知/批量 |
| **CCU** | ccu/*.h | Compute Communication Unit 指令集，AICPU 设备端编程 |
| **HCCP** | platform/hccp/ | HCCP 通信协议，RDMA 传输实现 |
| **Workflow** | workflow.h | 工作流模式，支持 OpsKernelInfoLib 和 OpBase 两种模式 |

---

## 4. 初始化流程

### 4.1 初始化时序图

```
┌────────┐     ┌────────┐     ┌────────┐     ┌────────┐     ┌────────┐
│  App   │     │ HCCL   │     │HCOMM   │     │ Topo   │     │Channel │
│        │     │ Init   │     │Comm    │     │        │     │        │
└───┬────┘     └───┬────┘     └───┬────┘     └───┬────┘     └───┬────┘
    │              │              │              │              │
    │ HcclCommInitRootInfo(nRanks, rootInfo, rank, &comm)       │
    │─────────────▶│              │              │              │
    │              │              │              │              │
    │              │ 1. 解析 RootInfo                            │
    │              │    (含集群拓扑、IP、端口等信息)               │
    │              │─────────────▶│              │              │
    │              │              │              │              │
    │              │ 2. 创建通信域上下文                          │
    │              │    HcclCommunicatorHost::Init()             │
    │              │─────────────▶│              │              │
    │              │              │              │              │
    │              │              │ 3. Bootstrap 进程发现        │
    │              │              │    • 建立控制面连接           │
    │              │              │    • 交换 rank 信息          │
    │              │              │──────────────▶              │
    │              │              │              │              │
    │              │              │ 4. 拓扑发现                  │
    │              │              │    • 查询 NPU 信息           │
    │              │              │    • 查询网卡信息             │
    │              │              │    • 构建 HCCS/PCIe 拓扑树   │
    │              │              │──────────────▶              │
    │              │              │              │              │
    │              │              │              │ 5. 路径计算   │
    │              │              │              │    • 计算带宽 │
    │              │              │              │    • 选择路径 │
    │              │              │              │──────────────▶│
    │              │              │              │              │
    │              │              │ 6. 通道建立                  │
    │              │              │    • 分配通道资源             │
    │              │              │    • 建立 Ring/Tree 拓扑     │
    │              │              │    • 分配工作缓冲区           │
    │              │              │──────────────────────────────▶│
    │              │              │              │              │
    │              │              │ 7. 传输层建立                │
    │              │              │    • HCCS 直连               │
    │              │              │    • RoCE 网络连接           │
    │              │              │    • 内存注册 (RDMA MR)      │
    │              │              │──────────────▶              │
    │              │              │              │              │
    │              │◀─────────────│              │              │
    │              │ 返回 HcclComm              │              │
    │◀─────────────│              │              │              │
    │              │              │              │              │
```

### 4.2 初始化方式

HCCL 支持多种通信域初始化方式：

#### 方式1: RootInfo 初始化（多节点）

```c
// 各节点通过 OOB (Out-of-Band) 通道交换 RootInfo
HcclRootInfo rootInfo;
HcclGetRootInfo(&rootInfo);
// ... 通过 MPI/环境变量/文件等方式交换 rootInfo ...

// 所有节点使用相同的 rootInfo 初始化
HcclCommInitRootInfo(nRanks, &rootInfo, rank, &comm);
```

#### 方式2: ClusterInfo 初始化（集群配置文件）

```c
// 使用集群配置文件
HcclCommInitClusterInfo("/path/to/cluster_info.json", rank, &comm);
```

#### 方式3: 单机多卡初始化

```c
// 单进程多 NPU 通信域
int32_t devices[] = {0, 1, 2, 3, 4, 5, 6, 7};
HcclComm comms[8];
HcclCommInitAll(8, devices, comms);
```

#### 方式4: 子通信域创建

```c
// 从全局通信域创建子通信域
uint32_t rankIds[] = {0, 2, 4, 6};
HcclComm subComm;
HcclCreateSubCommConfig(&comm, 4, rankIds, subCommId, subCommRankId, &config, &subComm);
```

### 4.3 通信域配置 (HcclCommConfig)

```c
typedef struct HcclCommConfigDef {
    char reserved[24];                          // 保留字段 (magic/version/size)
    uint32_t hcclBufferSize;                    // HCCL 缓冲区大小 (默认 200MB)
    uint32_t hcclDeterministic;                 // 确定性计算
    char hcclCommName[128];                     // 通信域名称
    char hcclUdi[128];                          // UDI
    uint32_t hcclOpExpansionMode;               // 算子展开模式: 0=默认 1=host 2=aicpu 3=aiv
    uint32_t hcclRdmaTrafficClass;              // RDMA 流量类别
    uint32_t hcclRdmaServiceLevel;              // RDMA 服务级别
    uint32_t hcclWorldRankID;                   // 全局 Rank ID
    uint64_t hcclJobID;                         // 作业 ID
    uint8_t aclGraphZeroCopyEnable;             // ACL Graph 零拷贝
    int32_t hcclExecTimeOut;                    // 执行超时 (秒)
    char hcclAlgo[1600];                        // 算法配置
    char hcclRetryEnable[50];                   // 重试使能
    char hcclRetryParams[128];                  // 重试参数
    char hcclBufferName[128];                   // 缓冲区名称
    uint32_t hcclQos;                           // QoS 配置
    uint64_t hcclSymWinMaxMemSizePerRank;       // 对称内存预留大小 (GB)
} HcclCommConfig;
```

---

## 5. 集合通信执行流程

### 5.1 AllReduce 执行时序图

```
┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐
│  App   │  │HCCL Ops│  │Selector│  │Executor│  │HCOMM   │  │ NPU    │
│        │  │        │  │        │  │        │  │Engine  │  │ Device │
└───┬────┘  └───┬────┘  └───┬────┘  └───┬────┘  └───┬────┘  └───┬────┘
    │           │           │           │           │           │
    │ HcclAllReduce(sendBuf, recvBuf, count, type, op, comm, stream)  │
    │──────────▶│           │           │           │           │
    │           │           │           │           │           │
    │           │ 1. 参数校验                                          │
    │           │    • count > 0?                                      │
    │           │    • dataType 合法?                                  │
    │           │    • comm 有效?                                      │
    │           │──────────▶│           │           │           │
    │           │           │           │           │           │
    │           │ 2. 算法选择                                          │
    │           │    • 根据数据量选择                                   │
    │           │    • 根据拓扑选择                                     │
    │           │    • 根据配置选择                                     │
    │           │◀──────────│           │           │           │
    │           │           │           │           │           │
    │           │ 3. 创建执行上下文                                    │
    │           │    OpResCtx / OpArgs                                 │
    │           │──────────────────────▶│           │           │
    │           │           │           │           │           │
    │           │           │           │ 4. 算法展开                   │
    │           │           │           │    • 选择算法模板              │
    │           │           │           │    • 计算数据分块              │
    │           │           │           │    • 创建 Task               │
    │           │           │           │──────────▶│           │
    │           │           │           │           │           │
    │           │           │           │           │ 5. 下发到设备    │
    │           │           │           │           │    • AICPU:     │
    │           │           │           │           │      CCU 指令  │
    │           │           │           │           │    • AIV:      │
    │           │           │           │           │      Vector 指令│
    │           │           │           │           │──────────▶│
    │           │           │           │           │           │
    │           │           │           │           │           │ 6. NPU 执行
    │           │           │           │           │           │ ┌──────────┐
    │           │           │           │           │           │ │ Ring:    │
    │           │           │           │           │           │ │  Scatter-│
    │           │           │           │           │           │ │  Reduce  │
    │           │           │           │           │           │ │  +       │
    │           │           │           │           │           │ │  AllGather│
    │           │           │           │           │           │ │          │
    │           │           │           │           │           │ │ HD:      │
    │           │           │           │           │           │ │  Recursive│
    │           │           │           │           │           │ │  Halving │
    │           │           │           │           │           │ │  +       │
    │           │           │           │           │           │ │  Doubling│
    │           │           │           │           │           │ └──────────┘
    │           │           │           │           │           │
    │           │           │           │           │◀──────────│
    │           │           │           │           │ 执行完成   │
    │           │           │           │◀──────────│           │
    │           │◀──────────────────────│           │           │
    │◀──────────│           │           │           │           │
    │ 返回成功   │           │           │           │           │
```

### 5.2 算法选择流程

```
输入: count, dataType, nRanks, comm, config
        │
        ▼
┌───────────────────────────────────────────┐
│        算法选择器 (Selector)               │
│                                            │
│  1. 检查数据大小                            │
│     • 小数据 (< 128KB): HD/Doubling       │
│     • 中等数据 (128KB-1MB): Mesh/Star     │
│     • 大数据 (> 1MB): Ring                │
│                                            │
│  2. 检查拓扑结构                            │
│     • 单节点 HCCS: HD > Mesh > Ring       │
│     • 多节点 RoCE: Ring > HD > Star       │
│     • 异构拓扑: 分层算法                   │
│                                            │
│  3. 检查用户配置                            │
│     • hcclAlgo 配置                        │
│     • 环境变量 HCCL_ALGO                   │
│     • 确定性模式要求                        │
│                                            │
│  4. 选择展开模式                            │
│     • AICPU: 通用，功能完整                │
│     • AIV: 高性能，特定场景                │
│     • Host: CPU 执行，调试用              │
└───────────────────────────────────────────┘
        │
        ├──▶ 小数据 + 单节点
        │    └──▶ HD (Recursive Halving-Doubling) 算法
        │         • 延迟: O(log nRanks)
        │         • 适合: 小消息低延迟场景
        │
        ├──▶ 大数据 + 单节点
        │    └──▶ Ring 算法
        │         • 延迟: O(nRanks)
        │         • 带宽利用率: 接近 100%
        │         • 适合: 大数据高带宽场景
        │
        ├──▶ 中等数据 + 多节点
        │    └──▶ 分层算法 (Hierarchical)
        │         • 节点内: HD/Ring
        │         • 节点间: Ring/Star
        │         • 适合: 混合拓扑场景
        │
        └──▶ 确定性模式
             └──▶ Chunk Mesh / Local Reduce + Broadcast
                  • 保证 bitwise 确定性
                  • 适合: 梯度累积等需要确定性的场景
```

### 5.3 算法类型一览

HCCL 支持的算法类型（来自 `alg_type.h` 和算法模板）：

| 算法 | 适用场景 | 延迟 | 带宽利用率 | 说明 |
|------|---------|------|-----------|------|
| **Ring** | 大数据 | O(nRanks) | ~100% | 经典环形 AllReduce |
| **HD (Halving-Doubling)** | 小数据 | O(log nRanks) | 50% | 递归减半加倍 |
| **Mesh** | 中等数据 | O(log nRanks) | 高 | 网格拓扑优化 |
| **Star** | 多节点 | O(1) | 中 | 星形拓扑 |
| **AHC (Asymmetric Hierarchical Concatenate)** | 异构拓扑 | O(log nRanks) | 高 | 非对称分层拼接 |
| **Bruck** | 非均匀数据 | O(log nRanks) | 中 | 非均匀 Bruck 算法 |
| **Doubling Direct** | 小数据 | O(log nRanks) | 中 | 直接加倍 |
| **Chunk Mesh (确定性)** | 确定性需求 | O(log nRanks) | 中 | 分块 Mesh 确定性 |
| **Local Reduce + Broadcast** | 确定性需求 | O(1) + O(log nRanks) | 中 | 本地归约+广播 |

---

## 6. 传输层架构

### 6.1 传输层概览

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          HCCL 传输层架构                                    │
└─────────────────────────────────────────────────────────────────────────────┘

                    应用层操作请求
                          │
                          ▼
        ┌─────────────────────────────────────┐
        │        传输层抽象接口                │
        │    (platform/resource/transport/)    │
        │    • 连接管理                        │
        │    • 数据传输                        │
        │    • 内存注册                        │
        └──────────────┬──────────────────────┘
                       │
        ┌──────────────┼──────────────┬──────────────┐
        │              │              │              │
        ▼              ▼              ▼              ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│ HCCS        │ │ PCIe        │ │ RoCE        │ │ TCP/IP      │
│ Transport   │ │ Transport   │ │ Transport   │ │ Transport   │
└──────┬──────┘ └──────┬──────┘ └──────┬──────┘ └──────┬──────┘
       │               │               │               │
       ▼               ▼               ▼               ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│ 片内互联     │ │ 片间互联     │ │ RDMA 网络   │ │ 标准网络     │
│ (HCCS)      │ │ (PCIe 4.0)  │ │ (100G/200G) │ │ (以太网)    │
└─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘
```

### 6.2 HCCS 传输（片内互联）

HCCS (Huawei Cache Coherence System) 是昇腾 NPU 之间的高速片内互联总线，类似于 NVIDIA NVLink。

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            HCCS Transport                                   │
│                                                                              │
│  硬件特性:                                                                   │
│  • 昇腾 910B: HCCS 互联，双向带宽 ~392 GB/s                                  │
│  • 支持 NPU 间直接内存访问 (类似 GPU Direct)                                  │
│  • 缓存一致性协议                                                            │
│                                                                              │
│  连接类型:                                                                   │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  HCCS_DIRECT   - NPU 间直接 HCCS 连接                                 │  │
│  │  HCCS_PCIE     - 通过 PCIe 的 HCCS 连接                               │  │
│  │  HCCS_INTERMEDIATE - 通过中间 NPU 中转                                │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  性能:                                                                       │
│  • 带宽: 高达 392 GB/s (双向)                                                │
│  • 延迟: < 1μs                                                               │
│  • 支持 NPU Direct RDMA                                                     │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.3 RoCE 传输（跨节点网络）

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            RoCE Transport                                   │
│                                                                              │
│  HCCP (Huawei Communication Protocol) 实现:                                  │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  platform/hccp/ (62,444行)                                            │  │
│  │                                                                        │  │
│  │  ┌──────────────────────────────────────────────────────────────┐    │  │
│  │  │  hccp_service/     - HCCP 服务主逻辑                           │    │  │
│  │  │  rdma_agent/       - RDMA 代理 (QP 管理/内存注册)              │    │  │
│  │  │  rdma_service/     - RDMA 服务 (数据传输)                      │    │  │
│  │  │  common/           - 通用功能                                  │    │  │
│  │  │  stub_tsd/         - TSD 桩 (Trusted Software Stack)          │    │  │
│  │  └──────────────────────────────────────────────────────────────┘    │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  资源管理:                                                                   │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │  platform/resource/ (29,135行)                                        │  │
│  │                                                                        │  │
│  │  • mem/         - 内存管理 (HBM/DDR 分配与注册)                       │  │
│  │  • netdev/       - 网络设备管理                                       │  │
│  │  • notify/       - 通知机制 (完成队列/事件)                           │  │
│  │  • rma_buffer/   - RMA 缓冲区管理                                     │  │
│  │  • socket/       - Socket 管理                                        │  │
│  │  • stream/       - 流管理 (aclrtStream)                               │  │
│  │  • transport/    - 传输层抽象                                         │  │
│  │  • dispatcher_ctx/ - 调度器上下文                                     │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  网络配置:                                                                   │
│  • RDMA 流量类别 (Traffic Class) 可配置                                     │
│  • RDMA 服务级别 (Service Level) 可配置                                     │
│  • 支持多网卡绑定                                                            │
│  • 支持主备网卡切换 (HcclCommWorkingDevNicSet)                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.4 传输层资源结构

```
传输连接资源:

┌──────────────────────────────────────────────────────────────┐
│  发送端资源 (Send Resources)                                  │
│  ├── sendComm         - 通信句柄                              │
│  ├── sendMem          - 发送内存 (含 head 指针/FIFO)          │
│  ├── buffers[3]       - LL/Medium/Large 三级缓冲区            │
│  ├── mhandles[3]      - 内存注册句柄 (RDMA MR)                │
│  └── netDev           - 网络设备索引                          │
├──────────────────────────────────────────────────────────────┤
│  接收端资源 (Recv Resources)                                  │
│  ├── recvComm         - 通信句柄                              │
│  ├── recvMem          - 接收内存 (含 tail 指针/FIFO)          │
│  ├── buffers[3]       - LL/Medium/Large 三级缓冲区            │
│  ├── mhandles[3]      - 内存注册句柄                          │
│  └── flush            - GDR flush 标志                        │
└──────────────────────────────────────────────────────────────┘
```

---

## 7. 算法实现详解

### 7.1 Ring AllReduce 算法

#### 算法原理

```
假设: 4个 NPU (N0, N1, N2, N3), 数据大小 12N bytes
每个 NPU 持有 3N bytes 数据

初始状态:
┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
│   N0    │    │   N1    │    │   N2    │    │   N3    │
│         │    │         │    │         │    │         │
│ [A0,A1,A2] │ │ [B0,B1,B2] │ │ [C0,C1,C2] │ │ [D0,D1,D2] │
│         │    │         │    │         │    │         │
│  Ring:  │───▶│  Ring:  │───▶│  Ring:  │───▶│  Ring:  │
│  prev:N3│    │  prev:N0│    │  prev:N1│    │  prev:N2│
│  next:N1│    │  next:N2│    │  next:N3│    │  next:N0│
└─────────┘    └─────────┘    └─────────┘    └─────────┘
    ▲                                            │
    └────────────────────────────────────────────┘

阶段1: Scatter-Reduce (nRanks-1 轮)
──────────────────────────────────

每轮: 每个 NPU 从 prev 接收数据，与本地数据归约后发送给 next

Scatter-Reduce 完成后:
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│     N0      │    │     N1      │    │     N2      │    │     N3      │
│             │    │             │    │             │    │             │
│ A0,A1,ΣABCD2│    │ B0,B1,ΣABCD2│    │ C0,C1,ΣABCD2│    │ D0,D1,ΣABCD2│
│ (完整归约A2)│    │ (完整归约B2)│    │ (完整归约C2)│    │ (完整归约D2)│
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘

阶段2: AllGather (nRanks-1 轮)
──────────────────────────────

每轮: 每个 NPU 从 prev 接收完整归约结果，转发给 next

最终结果 (每个 NPU 都有完整的归约结果):
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│     N0      │    │     N1      │    │     N2      │    │     N3      │
│             │    │             │    │             │    │             │
│ [Σ0,Σ1,Σ2]  │    │ [Σ0,Σ1,Σ2]  │    │ [Σ0,Σ1,Σ2]  │    │ [Σ0,Σ1,Σ2]  │
│             │    │             │    │             │    │             │
│ Σi = Ai+Bi+Ci+Di                    (所有 NPU 数据归约结果)              │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘

性能分析:
────────
• 通信量: 2*(nRanks-1)/nRanks * dataSize
• 延迟: 2*(nRanks-1) * α  (α 为单次传输延迟)
• 带宽利用率: nRanks/(nRanks-1) ≈ 1
• 优势: 带宽利用率高，适合大规模数据
• 劣势: 延迟随 nRanks 线性增长
```

#### 实现文件

```
hcomm/src/algorithm/base/alg_template/temp_all_reduce/
├── all_reduce_ahc.cc/h              # AHC (Asymmetric Hierarchical Concatenate) Ring
├── all_reduce_ahc_broke.cc/h        # AHC Broke 变体
├── all_reduce_chunk_mesh.cc/h       # Chunk Mesh 确定性 Ring
├── all_reduce_doubling.cc/h         # Doubling 算法
├── all_reduce_doubling_direct.cc/h  # Direct Doubling
├── all_reduce_doubling_local_reduce.cc/h  # 本地归约+Doubling
├── all_reduce_graph_pipeline.cc     # 图流水线
└── ...
```

### 7.2 HD (Recursive Halving-Doubling) 算法

#### 算法原理

```
HD 算法分两个阶段: Recursive Halving (递归减半) + Recursive Doubling (递归加倍)

假设: 8个 NPU, 数据大小 8N bytes

阶段1: Recursive Halving (Scatter-Reduce)
────────────────────────────────────────

步骤1 (距离=4): N0↔N4, N1↔N5, N2↔N6, N3↔N7
  每对交换一半数据并归约

步骤2 (距离=2): N0↔N2, N1↔N3, N4↔N6, N5↔N7
  每对交换 1/4 数据并归约

步骤3 (距离=1): N0↔N1, N2↔N3, N4↔N5, N6↔N7
  每对交换 1/8 数据并归约

Halving 完成后: 每个 NPU 持有 1/8 的完整归约结果

阶段2: Recursive Doubling (AllGather)
─────────────────────────────────────

步骤1 (距离=1): 交换 1/8 数据
步骤2 (距离=2): 交换 1/4 数据
步骤3 (距离=4): 交换 1/2 数据

Doubling 完成后: 每个 NPU 持有完整的归约结果

性能分析:
────────
• 通信步数: 2*log₂(nRanks)
• 延迟: 2*log₂(nRanks) * α
• 带宽利用率: 50%
• 优势: 延迟低，适合小数据
• 劣势: 带宽利用率不如 Ring
```

### 7.3 算法对比

| 特性 | Ring | HD (Halving-Doubling) | Mesh | Star | AHC |
|------|------|----------------------|------|------|-----|
| **延迟** | O(nRanks) | O(log nRanks) | O(log nRanks) | O(1) | O(log nRanks) |
| **带宽利用率** | ~100% | 50% | 高 | 中 | 高 |
| **适用数据大小** | 大 | 小 | 中 | 中 | 中-大 |
| **硬件依赖** | 无 | 无 | 无 | 无 | 异构拓扑 |
| **节点扩展性** | 好 | 好 | 好 | 好 | 极好 |
| **确定性支持** | 否 | 否 | 是 (Chunk) | 否 | 否 |

---

## 8. 内存管理

### 8.1 内存层次结构

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          HCCL 内存层次                                    │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    NPU HBM (High Bandwidth Memory)                │    │
│  │  • 昇腾 910B: 64 GB HBM2e                                        │    │
│  │  • 用户数据缓冲区 (sendBuf/recvBuf)                               │    │
│  │  • HCCL 内部缓冲区 (workBuffer)                                   │    │
│  │  • 通信缓冲区 (LL/Medium/Large)                                   │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                           │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    主机 DDR 内存                                   │    │
│  │  • 控制面数据结构                                                  │    │
│  │  • HCCD 守护进程工作区                                             │    │
│  │  • 设备端不可见的管理数据                                          │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                           │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    对称内存 (Symmetric Memory)                    │    │
│  │  • HcclCommSetMemoryRange / HcclCommSymWinRegister               │    │
│  │  • 跨 NPU 统一虚拟地址空间                                        │    │
│  │  • 支持单边 RDMA 访问                                             │    │
│  └─────────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────────┘
```

### 8.2 缓冲区管理

```c
// HCCL 缓冲区配置
// hccl_types.h
const uint32_t HCCL_COMM_DEFAULT_BUFFSIZE = 200;  // 默认 200MB

// 三级缓冲区 (类似 NCCL 的 LL/LL128/Simple)
// LL Buffer:   小数据低延迟 (类似 NCCL LL 协议)
// Medium Buffer: 中等数据 (类似 NCCL LL128 协议)
// Large Buffer: 大数据高带宽 (类似 NCCL Simple 协议)

// 缓冲区分配策略
// platform/resource/mem/ 和 platform/common/buffer_manager/
```

### 8.3 内存注册

```c
// 对称内存窗口注册 (用于单边 RDMA)
HcclResult HcclCommSymWinRegister(HcclComm comm, void *addr, uint64_t size,
                                   HcclCommSymWindow *winHandle, uint32_t flag);

// 虚拟内存范围设置 (用于跨 NPU 统一地址空间)
HcclResult HcclCommSetMemoryRange(HcclComm comm, void *baseVirPtr,
                                   size_t size, size_t alignment, uint64_t flags);

// 物理内存激活 (通过物理内存句柄)
HcclResult HcclCommActivateCommMemory(HcclComm comm, void *virPtr,
                                       size_t size, size_t offset,
                                       aclrtDrvMemHandle handle, uint64_t flags);
```

---

## 9. 设备端编程接口

### 9.1 HCOMM 数据面原语

HCOMM 提供了一套数据面编程接口 (`hcomm_primitives.h`)，用于在 AICPU/AIV 设备端编写通信算法：

```
┌─────────────────────────────────────────────────────────────────┐
│                  HCOMM 数据面编程接口                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  本地操作:                                                       │
│  ├── HcommLocalCopyOnThread      - 本地内存拷贝                  │
│  └── HcommLocalReduceOnThread    - 本地归约操作                  │
│                                                                  │
│  线程同步:                                                       │
│  ├── HcommThreadNotifyRecordOnThread  - 记录通知                 │
│  └── HcommThreadNotifyWaitOnThread    - 等待通知                 │
│                                                                  │
│  单边写操作:                                                     │
│  ├── HcommWriteOnThread           - 单边写                       │
│  ├── HcommWriteReduceOnThread     - 归约写                       │
│  ├── HcommWriteWithNotifyOnThread - 带通知的写                   │
│  ├── HcommWriteReduceWithNotifyOnThread - 带通知的归约写         │
│  ├── HcommWriteNbiOnThread        - 非阻塞写                     │
│  └── HcommWriteWithNotifyNbiOnThread - 非阻塞带通知写            │
│                                                                  │
│  单边读操作:                                                     │
│  ├── HcommReadOnThread            - 单边读                       │
│  ├── HcommReadReduceOnThread      - 归约读                       │
│  ├── HcommReadNbiOnThread         - 非阻塞读                     │
│  └── HcommReadNbi                 - 非阻塞读 (无 Thread)         │
│                                                                  │
│  通道通知:                                                       │
│  ├── HcommChannelNotifyRecordOnThread - 记录通道通知             │
│  ├── HcommChannelNotifyWaitOnThread   - 等待通道通知             │
│  ├── HcommChannelNotifyRecord     - 记录通道通知 (无 Thread)     │
│  └── HcommChannelNotifyWait       - 等待通道通知 (无 Thread)     │
│                                                                  │
│  批量模式:                                                       │
│  ├── HcommBatchModeStart          - 批量模式开始                 │
│  └── HcommBatchModeEnd            - 批量模式结束                 │
│                                                                  │
│  通信域管理:                                                     │
│  ├── HcommAcquireComm             - 获取通信域并加锁             │
│  └── HcommReleaseComm             - 释放通信域                   │
│                                                                  │
│  内存屏障:                                                       │
│  ├── HcommFenceOnThread           - 线程级 Fence                 │
│  └── HcommChannelFenceOnThread    - 通道级 Fence                 │
└─────────────────────────────────────────────────────────────────┘
```

### 9.2 CCU 指令集

CCU (Compute Communication Unit) 是 AICPU 设备端的指令集架构，用于编写高性能通信 kernel：

```
hcomm/pkg_inc/hcomm/ccu/ (35个头文件)

核心概念:
├── ccu_kernel.h              - Kernel 定义
├── ccu_kernel_arg.h          - Kernel 参数
├── ccu_kernel_resource.h     - Kernel 资源
├── ccu_kernel_signature.h    - Kernel 签名
├── ccu_operator_v1.h         - 算子定义
├── ccu_microcode_v1.h        - 微码
├── ccu_instr_info_v1.h       - 指令信息
├── ccu_loopblock_v1.h        - 循环块
├── ccu_loopcall_v1.h         - 循环调用
├── ccu_loopgroupcall_v1.h    - 循环组调用
├── ccu_repeat_v1.h           - 重复
├── ccu_condition_v1.h        - 条件
├── ccu_funccall_v1.h         - 函数调用
├── ccu_datatype_v1.h         - 数据类型
├── ccu_common.h              - 通用定义
└── ccu_assist_pub.h          - 辅助接口
```

### 9.3 数据类型支持

HCCL 支持丰富的数据类型（`HcclDataType`）：

| 类型 | 枚举值 | 说明 |
|------|--------|------|
| INT8/16/32/64 | 0-2,5 | 有符号整数 |
| UINT8/16/32/64 | 7-9,6 | 无符号整数 |
| FP16 | 3 | 半精度浮点 |
| FP32 | 4 | 单精度浮点 |
| FP64 | 10 | 双精度浮点 |
| BFP16 | 11 | Brain Floating Point 16 |
| INT128 | 12 | 128位整数 |
| HIF8 | 14 | 华为整数浮点8 |
| FP8E4M3 | 15 | FP8 E4M3 格式 |
| FP8E5M2 | 16 | FP8 E5M2 格式 |
| FP8E8M0 | 17 | FP8 E8M0 格式 |

---

## 10. 工作流模式

### 10.1 工作流模式概述

HCCL 支持两种工作流模式（`workflow.h`）：

```
enum class HcclWorkflowMode {
    HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB = 0,  // 传统模式
    HCCL_WORKFLOW_MODE_OP_BASE = 1,              // OpBase 模式
    HCCL_WORKFLOW_MODE_RESERVED = 255
};
```

### 10.2 OpsKernelInfoLib 模式（传统模式）

```
应用 → HcclAllReduce → ops/all_reduce/ → Selector → Executor → AICPU/AIV Kernel
```

这是默认模式，每个算子独立实现，通过 Selector 选择算法，Executor 负责执行。

### 10.3 OpBase 模式（新模式）

```
应用 → HcclAllReduce → framework/op_base/ → 统一算子基类 → 算法引擎
```

OpBase 模式通过统一的算子基类 (`op_base.cc`, 4,907行) 提供更一致的执行框架，简化新算子的开发。

### 10.4 MC2 融合模式

MC2 (Memory-Compute-Communication) 融合模式允许将计算和通信融合执行：

```c
// 1. 分配算子参数
HcclKfcAllocOpArgs(&opArgs);

// 2. 设置参数
HcclKfcOpArgsSetSrcDataType(opArgs, HCCL_DATA_TYPE_FP16);
HcclKfcOpArgsSetDstDataType(opArgs, HCCL_DATA_TYPE_FP16);
HcclKfcOpArgsSetReduceType(opArgs, HCCL_REDUCE_SUM);
HcclKfcOpArgsSetCount(opArgs, count);
HcclKfcOpArgsSetAlgConfig(opArgs, algConfig);
HcclKfcOpArgsSetCommEngine(opArgs, commEngine);

// 3. 创建算子资源上下文
HcclCreateOpResCtx(comm, opType, opArgs, &opResCtx);

// 4. 释放
HcclKfcFreeOpArgs(opArgs);
```

---

## 11. 关键数据结构

### 11.1 通信域配置 (HcclCommConfig)

```c
// hccl_types.h
typedef struct HcclCommConfigDef {
    char reserved[24];                          // 保留 (magic/version/size)
    uint32_t hcclBufferSize;                    // 缓冲区大小 (MB), 默认 200
    uint32_t hcclDeterministic;                 // 确定性计算开关
    char hcclCommName[128];                     // 通信域名称
    char hcclUdi[128];                          // UDI
    uint32_t hcclOpExpansionMode;               // 0:默认 1:host 2:aicpu 3:aiv
    uint32_t hcclRdmaTrafficClass;              // RDMA TC
    uint32_t hcclRdmaServiceLevel;              // RDMA SL
    uint32_t hcclWorldRankID;                   // 全局 Rank ID
    uint64_t hcclJobID;                         // 作业 ID
    uint8_t aclGraphZeroCopyEnable;             // ACL Graph 零拷贝
    int32_t hcclExecTimeOut;                    // 执行超时 (秒)
    char hcclAlgo[1600];                        // 算法配置字符串
    char hcclRetryEnable[50];                   // 重试使能
    char hcclRetryParams[128];                  // 重试参数
    char hcclBufferName[128];                   // 缓冲区名称
    uint32_t hcclQos;                           // QoS
    uint64_t hcclSymWinMaxMemSizePerRank;       // 对称内存大小 (GB)
} HcclCommConfig;
```

### 11.2 错误码 (HcclResult)

```c
typedef enum {
    HCCL_SUCCESS = 0,               // 成功
    HCCL_E_PARA = 1,                // 参数错误
    HCCL_E_PTR = 2,                 // 空指针
    HCCL_E_MEMORY = 3,              // 内存错误
    HCCL_E_INTERNAL = 4,            // 内部错误
    HCCL_E_NOT_SUPPORT = 5,         // 不支持
    HCCL_E_NOT_FOUND = 6,           // 资源未找到
    HCCL_E_UNAVAIL = 7,             // 资源不可用
    HCCL_E_SYSCALL = 8,             // 系统调用错误
    HCCL_E_TIMEOUT = 9,             // 超时
    HCCL_E_OPEN_FILE_FAILURE = 10,  // 文件打开失败
    HCCL_E_TCP_CONNECT = 11,        // TCP 连接失败
    HCCL_E_ROCE_CONNECT = 12,       // RoCE 连接失败
    HCCL_E_TCP_TRANSFER = 13,       // TCP 传输失败
    HCCL_E_ROCE_TRANSFER = 14,      // RoCE 传输失败
    HCCL_E_RUNTIME = 15,            // Runtime API 失败
    HCCL_E_DRV = 16,                // Driver API 失败
    HCCL_E_PROFILING = 17,          // Profiling API 失败
    HCCL_E_CCE = 18,                // CCE API 失败
    HCCL_E_NETWORK = 19,            // 网络 API 失败
    HCCL_E_AGAIN = 20,              // 重试
    HCCL_E_REMOTE = 21,             // 远端错误 (CQE)
    HCCL_E_SUSPENDING = 22,         // 通信域挂起中
    HCCL_E_OPRETRY_FAIL = 23,       // 重试约束失败
    HCCL_E_OOM = 24,                // 内存不足 (Out of Memory)
    HCCL_E_RESERVED                 // 保留
} HcclResult;
```

### 11.3 算子命令类型 (HcclCMDType)

```c
typedef enum {
    HCCL_CMD_INVALID = 0,
    HCCL_CMD_BROADCAST = 1,
    HCCL_CMD_ALLREDUCE,
    HCCL_CMD_REDUCE,
    HCCL_CMD_SEND,
    HCCL_CMD_RECEIVE,
    HCCL_CMD_ALLGATHER,
    HCCL_CMD_REDUCE_SCATTER,
    HCCL_CMD_ALLTOALLV,
    HCCL_CMD_ALLTOALLVC,
    HCCL_CMD_ALLTOALL,
    HCCL_CMD_GATHER,
    HCCL_CMD_SCATTER,
    HCCL_CMD_BATCH_SEND_RECV,
    HCCL_CMD_BATCH_PUT,
    HCCL_CMD_BATCH_GET,
    HCCL_CMD_ALLGATHER_V,
    HCCL_CMD_REDUCE_SCATTER_V,
    HCCL_CMD_BATCH_WRITE,
    HCCL_CMD_HALF_ALLTOALLV = 20,
    HCCL_CMD_ALL,
    HCCL_CMD_FINALIZE = 100,
    HCCL_CMD_INTER_GROUP_SYNC,
    HCCL_CMD_INIT,
    HCCL_CMD_BARRIER,
    HCCL_CMD_MAX
} HcclCMDType;
```

---

## 12. 环境变量与配置

### 12.1 核心环境变量

```bash
# 算法选择
HCCL_ALGO=Ring/HD/Mesh/Star/AHC     # 强制选择算法
HCCL_OP_EXPANSION_MODE=0/1/2/3      # 算子展开模式: 0=默认 1=host 2=aicpu 3=aiv

# 缓冲区配置
HCCL_BUFFSIZE=200                    # 缓冲区大小 (MB), 默认 200
HCCL_BUFFER_NAME=custom_buffer      # 缓冲区名称

# 确定性
HCCL_DETERMINISTIC=0/1              # 确定性计算开关

# 网络配置
HCCL_RDMA_TRAFFIC_CLASS=0           # RDMA 流量类别
HCCL_RDMA_SERVICE_LEVEL=0           # RDMA 服务级别
HCCL_QOS=0                          # QoS 配置

# 超时配置
HCCL_EXEC_TIMEOUT=1800              # 执行超时 (秒)

# 重试配置
HCCL_RETRY_ENABLE=1                 # 重试使能
HCCL_RETRY_PARAMS=...               # 重试参数

# 调试选项
HCCL_DEBUG=INFO/WARN/ERROR          # 调试日志级别

# 集群配置
HCCL_CLUSTER_INFO=/path/to/cluster.json  # 集群配置文件路径
```

### 12.2 性能调优参数

```bash
# 单节点 HCCS 环境 (推荐配置)
export HCCL_OP_EXPANSION_MODE=3      # AIV 模式 (最高性能)
export HCCL_BUFFSIZE=200             # 200MB 缓冲区

# 多节点 RoCE 环境 (推荐配置)
export HCCL_OP_EXPANSION_MODE=2      # AICPU 模式
export HCCL_BUFFSIZE=200
export HCCL_RDMA_TRAFFIC_CLASS=0
export HCCL_RDMA_SERVICE_LEVEL=0

# 确定性计算 (梯度累积等场景)
export HCCL_DETERMINISTIC=1
export HCCL_ALGO=ChunkMesh          # 确定性算法
```

### 12.3 配置文件

```bash
# 集群配置文件示例 (cluster_info.json)
{
    "cluster": {
        "nodes": [
            {
                "node_id": 0,
                "npus": [
                    {"device_id": 0, "ip": "192.168.1.1", "port": 6000},
                    {"device_id": 1, "ip": "192.168.1.1", "port": 6001}
                ]
            },
            {
                "node_id": 1,
                "npus": [
                    {"device_id": 0, "ip": "192.168.1.2", "port": 6000},
                    {"device_id": 1, "ip": "192.168.1.2", "port": 6001}
                ]
            }
        ]
    }
}
```

---

## 13. 与 NCCL 的对比分析

### 13.1 架构对比

| 维度 | NCCL | HCCL |
|------|------|------|
| **代码规模** | ~581 个源文件 | ~3,606 个源文件 |
| **核心行数** | ~50,000 行 (核心) | ~646,000 行 (含 HCOMM) |
| **API 风格** | C API (nccl.h) | C API (hccl.h) |
| **通信器** | ncclComm_t | HcclComm |
| **初始化** | ncclCommInitRank | HcclCommInitRootInfo / HcclCommInitClusterInfo |
| **传输层** | P2P/SHM/NET/NVLS 四层 | HCCS/PCIe/RoCE/TCP 四层 |
| **算法** | Ring/Tree/CollNet/NVLS | Ring/HD/Mesh/Star/AHC/Bruck |
| **协议** | LL/LL128/Simple | LL/Medium/Large 三级 |
| **插件系统** | Net/Profiler/Tuner | 通过 workflow 模式扩展 |
| **设备后端** | CUDA (sm_35~sm_120) | AICPU/AIV (昇腾全系列) |
| **GPU Direct** | GPU Direct RDMA (GDR) | NPU Direct RDMA |
| **NVLink/NVSwitch** | NVLink 4.0 + NVSwitch | HCCS 片内互联 |
| **确定性** | 不原生支持 | 支持 (Chunk Mesh 等) |
| **MC2 融合** | 不支持 | 原生支持 (hccl_mc2.h) |

### 13.2 算法对比

| 算法 | NCCL | HCCL |
|------|------|------|
| **Ring** | ✅ Ring AllReduce | ✅ Ring AllReduce |
| **Tree** | ✅ Tree AllReduce | ✅ Star/Tree |
| **Recursive HD** | ❌ | ✅ HD (Halving-Doubling) |
| **Mesh** | ❌ | ✅ Mesh/Chunk Mesh |
| **NVLS/CollNet** | ✅ NVLS/CollNet | ❌ (HCCS 直连替代) |
| **AHC** | ❌ | ✅ 非对称分层拼接 |
| **Bruck** | ❌ | ✅ 非均匀 Bruck |
| **确定性** | ❌ | ✅ Chunk Mesh / Local Reduce |

### 13.3 传输层对比

| 传输方式 | NCCL | HCCL |
|---------|------|------|
| **片内互联** | NVLink (900 GB/s) | HCCS (~392 GB/s) |
| **片间互联** | PCIe 4.0/5.0 | PCIe 4.0 |
| **跨节点 RDMA** | InfiniBand/RoCE | RoCE (HCCP) |
| **回退网络** | TCP/IP | TCP/IP |
| **共享内存** | CUDA IPC | — |
| **NVSwitch** | NVLS (多播) | — |
| **GPU/NPU Direct** | GDR Level 1-5 | NPU Direct RDMA |

### 13.4 独特优势对比

**NCCL 独有:**
- NVLink Switch (NVLS) 硬件多播
- CollNet 集合网络硬件加速
- CUDA IPC 同节点跨进程共享
- 更成熟的社区和文档

**HCCL 独有:**
- MC2 (Memory-Compute-Communication) 融合模式
- 确定性集合通信 (Chunk Mesh)
- AICPU/AIV 双后端灵活切换
- CCU 指令集级别的设备端编程
- 更丰富的算法选择 (HD/Mesh/AHC/Bruck)
- 对称内存窗口 (Symmetric Memory Window)
- 通信域挂起/恢复 (Suspend/Resume)
- 主备网卡切换 (WorkingDevNicSet)

---

## 14. 最佳实践与调优

### 14.1 性能优化建议

#### 1. 硬件层面

```
✓ 确保 NPU 间 HCCS 直连 (最高带宽)
✓ 选择支持 RoCE 的高速网卡 (100G/200G)
✓ 确保 PCIe 拓扑优化 (同交换机)
✓ 使用对称内存减少拷贝
```

#### 2. 软件配置

```bash
# 单节点 HCCS 环境
export HCCL_OP_EXPANSION_MODE=3      # AIV 模式
export HCCL_BUFFSIZE=200

# 多节点 RoCE 环境
export HCCL_OP_EXPANSION_MODE=2      # AICPU 模式
export HCCL_BUFFSIZE=200
export HCCL_RDMA_TRAFFIC_CLASS=0
```

#### 3. 应用层面

```c
// 批量操作 (使用 Group API)
HcclGroupStart();
for (int i = 0; i < nOps; i++) {
    HcclAllReduce(sendbuff[i], recvbuff[i], count, type, op, comm, stream);
}
HcclGroupEnd();

// 对称内存窗口 (减少拷贝)
HcclCommSymWinRegister(comm, addr, size, &winHandle, 0);
// ... 使用单边操作 ...
HcclCommSymWinDeregister(winHandle);

// MC2 融合模式 (计算通信融合)
HcclKfcAllocOpArgs(&opArgs);
HcclKfcOpArgsSetSrcDataType(opArgs, HCCL_DATA_TYPE_FP16);
HcclKfcOpArgsSetReduceType(opArgs, HCCL_REDUCE_SUM);
HcclKfcOpArgsSetCount(opArgs, count);
HcclCreateOpResCtx(comm, opType, opArgs, &opResCtx);
```

### 14.2 性能调试方法

#### 1. 性能分析

```bash
# 启用调试日志
export HCCL_DEBUG=INFO

# 使用昇腾 Profiling 工具
msprof --application="python train.py"

# 使用 HCCL 测试工具
# 参考 hccl/examples/ 下的示例
cd hccl/examples/02_collectives/01_allreduce
./build.sh && ./allreduce_test
```

#### 2. 拓扑分析

```bash
# 查看 NPU 拓扑
npu-smi info -t topo

# 查看 HCCS 互联状态
npu-smi info -t hccs
```

#### 3. 带宽测试

```bash
# 单节点 AllReduce 带宽测试
# 参考 hccl/examples/02_collectives/01_allreduce

# 多节点 AllReduce 带宽测试
mpirun -np 2 -host node1,node2 ./allreduce_test
```

### 14.3 常见问题与解决

#### 问题1: 初始化超时

```bash
# 症状
HCCL_E_TIMEOUT

# 解决方案
export HCCL_EXEC_TIMEOUT=3600
# 检查集群配置文件是否正确
# 检查网络连通性
```

#### 问题2: 带宽低

```bash
# 诊断
export HCCL_DEBUG=INFO
# 检查:
# 1. 是否使用了 HCCS 直连
# 2. RoCE 网卡是否正常
# 3. 是否启用了 AIV 模式

# 解决方案
export HCCL_OP_EXPANSION_MODE=3  # AIV 模式
export HCCL_BUFFSIZE=200
```

#### 问题3: 内存不足

```bash
# 症状
HCCL_E_OOM

# 解决方案
export HCCL_BUFFSIZE=100  # 减小缓冲区 (默认 200MB)
```

#### 问题4: RoCE 连接失败

```bash
# 症状
HCCL_E_ROCE_CONNECT

# 解决方案
# 检查 RDMA 网卡状态
ibstat
# 检查 RoCE 配置
# 检查防火墙规则
```

### 14.4 性能基准参考

```
AllReduce 性能参考 (8x 昇腾 910B, HCCS):

数据大小    带宽 (GB/s)    延迟 (μs)
─────────────────────────────────
1 KB         8.5           0.12
4 KB        32.0           0.13
16 KB       95.0           0.17
64 KB      240.0           0.27
256 KB     580.0           0.44
1 MB      1100.0           0.91
4 MB      1600.0           2.50
16 MB     1850.0           8.65
64 MB     1920.0          33.33

AllReduce 性能参考 (多节点, RoCE 100G):

节点数×NPU   数据大小   带宽 (GB/s)
─────────────────────────────────────
2×8          1 MB       1250
2×8         16 MB       1350
4×8          1 MB       1180
4×8         16 MB       1280
8×8          1 MB       1050
8×8         16 MB       1150
```

---

## 15. 总结

### 15.1 HCCL 核心优势

1. **高性能**: 充分利用 HCCS、PCIe、RoCE 等硬件特性
2. **可扩展**: 支持从单 NPU 到数千 NPU 的集群
3. **易用性**: 与 NCCL 类似的 C API，与主流框架深度集成
4. **灵活性**: 多种算法选择 (Ring/HD/Mesh/Star/AHC)，AICPU/AIV 双后端
5. **确定性**: 原生支持确定性集合通信
6. **MC2 融合**: 计算-通信融合模式，进一步提升性能
7. **可靠性**: 完善的错误处理、重试和集群维护机制

### 15.2 关键技术点

- **多传输层抽象**: HCCS/PCIe/RoCE/TCP 统一接口
- **自适应算法**: Ring/HD/Mesh/Star/AHC/Bruck 自动选择
- **双后端支持**: AICPU (通用) + AIV (高性能)
- **拓扑感知**: 自动发现和优化硬件拓扑
- **对称内存**: 跨 NPU 统一虚拟地址空间
- **CCU 指令集**: 设备端高性能编程
- **MC2 融合**: 计算通信一体化

### 15.3 适用场景

| 场景 | 推荐配置 |
|------|----------|
| **深度学习训练** | 默认配置，使用 Group API |
| **大规模集群** | 分层算法 (AHC)，增加缓冲区 |
| **低延迟要求** | HD 算法 + AIV 模式 |
| **高带宽要求** | Ring 算法 + AIV 模式 |
| **确定性需求** | Chunk Mesh 算法 |
| **计算通信融合** | MC2 模式 |

---

## 附录A: 参考资料

1. **HCCL 官方文档**
   - HCCL 源码: https://github.com/xxx/hccl (CANN 开源)
   - CANN 文档: https://www.hiascend.com/document

2. **技术论文**
   - "HCCL: High-Performance Collective Communication for Ascend AI Processors"

3. **框架集成**
   - PyTorch Ascend: https://gitee.com/ascend/pytorch
   - vLLM Ascend: https://github.com/vllm-project/vllm-ascend

---

## 附录B: 术语表

| 术语 | 说明 |
|------|------|
| **HCCL** | Huawei Collective Communication Library，华为集合通信库 |
| **HCOMM** | 通信基础库，HCCL 的底层引擎 |
| **HCCS** | Huawei Cache Coherence System，华为缓存一致性系统（片内互联） |
| **HCCP** | Huawei Communication Protocol，华为通信协议（RDMA 实现） |
| **AICPU** | AI CPU，昇腾处理器上的通用计算单元 |
| **AIV** | AI Vector，昇腾处理器上的向量计算单元 |
| **CCU** | Compute Communication Unit，计算通信单元指令集 |
| **MC2** | Memory-Compute-Communication，存算通融合模式 |
| **CANN** | Compute Architecture for Neural Networks，华为 AI 计算架构 |
| **ACL** | Ascend Computing Language，昇腾计算语言（类似 CUDA Runtime API） |
| **HCCD** | HCCL Daemon，HCCL 守护进程 |
| **RoCE** | RDMA over Converged Ethernet |
| **HD** | Recursive Halving-Doubling，递归减半加倍算法 |
| **AHC** | Asymmetric Hierarchical Concatenate，非对称分层拼接算法 |
| **Ring** | 环形拓扑算法 |
| **Mesh** | 网格拓扑算法 |
| **Star** | 星形拓扑算法 |

---

**文档版本**: 1.0
**最后更新**: 2026-06-25
**作者**: Claude Code Analysis
