# vLLM 多进程架构设计文档

> 文档版本：v1.0
> 创建日期：2026-05-16
> 适用版本：vLLM 0.20.x

---

## 目录

1. [概述](#概述)
2. [核心组件](#核心组件)
3. [CPU 单进程模式](#cpu-单进程模式)
4. [单机多卡模式](#单机多卡模式)
   - [详细启动过程](#详细启动过程)
   - [关键对象说明](#关键对象说明)
   - [启动时序图](#启动时序图)
   - [启动日志示例](#启动日志示例)
5. [多机多卡模式](#多机多卡模式)
6. [进程间通信机制](#进程间通信机制)
7. [配置参数说明](#配置参数说明)
8. [设计优势与局限](#设计优势与局限)

---

## 概述

vLLM 采用**层次化 + 模块化**的进程架构设计，支持从单 CPU 进程到多机多卡的多种部署场景。

### 支持的场景

| 场景 | 进程模型 | 并行策略 | 适用环境 |
|------|----------|----------|----------|
| CPU 单进程 | 1 进程 | 无 | 开发调试、小规模部署 |
| 单机多卡 | N+2 进程 | 数据并行 + 流水线并行 | 单服务器多 GPU |
| 多机多卡 | M×N+2 进程 | 数据并行 + 张量并行 + 流水线并行 | 分布式集群 |

### 核心设计原则

1. **层次化**：API Server → Engine Core → Worker 三层架构
2. **模块化**：各组件职责明确，可独立扩展
3. **透明性**：多进程调用与单进程调用方式一致
4. **高性能**：通过共享内存、RPC 优化减少通信开销

---

## 核心组件

### 1. LLMEngine（引擎封装层）

**位置**：API Server 进程

**职责**：
- 提供统一的推理接口
- 管理请求生命周期
- 协调输入/输出处理
- 提供向后兼容的 API

**关键代码**：
```python
class LLMEngine:
    def __init__(self, vllm_config, executor_class, log_stats):
        self.input_processor = InputProcessor(vllm_config, renderer)
        self.output_processor = OutputProcessor(tokenizer, ...)
        self.engine_core = EngineCoreClient.make_client(...)
```

### 2. EngineCore（推理核心层）

**位置**：Engine Core 进程

**职责**：
- 管理 KV 缓存（PagedAttention）
- 调度推理任务到执行器
- 管理请求队列
- 处理动态批处理
- 跨 Worker 结果汇总

**关键代码**：
```python
class EngineCore:
    def __init__(self, vllm_config, executor_class):
        self.model_executor = executor_class(vllm_config)
        self.scheduler = Scheduler(vllm_config)
        self.kv_cache_manager = KVCacheManager(vllm_config)
```

### 3. ModelExecutor（模型执行层）

**位置**：Worker 进程

**职责**：
- 加载模型权重到内存/显存
- 执行模型前向传播
- 处理 token 生成
- 支持不同设备（CPU/GPU）

**关键代码**：
```python
class CPUExecutor(Executor):
    def execute(self, requests):
        outputs = self.model.forward(requests)
        return outputs
```

### 4. 组件关系图

```
┌─────────────────────────────────────────────────────────────┐
│                      用户请求                                │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  LLMEngine（接口层）                                        │
│  - 输入处理                                                 │
│  - 输出处理                                                 │
│  - 生命周期管理                                             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  EngineCore（核心层）                                       │
│  - KV 缓存管理                                              │
│  - 任务调度                                                 │
│  - 动态批处理                                               │
│  - 结果汇总                                                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  ModelExecutor（执行层）                                   │
│  - 模型加载                                                 │
│  - 前向传播                                                 │
│  - Token 生成                                               │
└─────────────────────────────────────────────────────────────┘
```

---

## CPU 单进程模式

### 架构概述

CPU 单进程模式是最简单的部署方式，所有组件运行在**单一进程**内。

### 进程结构

```
┌─────────────────────────────────────────────────────────────┐
│                    单进程（API Server）                     │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  LLMEngine                                          │    │
│  │      ├── InputProcessor                             │    │
│  │      ├── OutputProcessor                            │    │
│  │      └── EngineCore                                 │    │
│  │              ├── Scheduler                          │    │
│  │              ├── KVCacheManager                      │    │
│  │              └── CPUExecutor                         │    │
│  │                      └── Model                      │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### 进程数量

```
总进程数 = 1
```

### 启动命令

```bash
VLLM_TARGET_DEVICE=cpu python -m vllm.entrypoints.cli.main serve distilgpt2 \
  --gpu-memory-utilization 0.2 \
  --max-num-seqs 1 \
  --port 8000
```

### 通信方式

- **组件间通信**：直接函数调用（无 IPC 开销）
- **优势**：简单、低延迟
- **劣势**：无法利用多核/多卡资源

### 适用场景

| 场景 | 推荐度 | 说明 |
|------|--------|------|
| 开发调试 | ⭐⭐⭐⭐⭐ | 简单易用，调试方便 |
| 小规模推理 | ⭐⭐⭐⭐ | 适合 QPS < 10 的场景 |
| 资源受限环境 | ⭐⭐⭐⭐ | 最低资源需求 |
| 生产环境 | ⭐⭐ | 不推荐，无容错能力 |

### 资源使用

| 资源 | 使用情况 |
|------|----------|
| CPU | 单核或指定核数 |
| 内存 | 模型大小 + KV 缓存 |
| GPU | 无 |

---

## 单机多卡模式

### 架构概述

单机多卡模式利用**数据并行**策略，将推理任务分发到多个 GPU Worker 进程。

### 进程结构

```
┌─────────────────────────────────────────────────────────────────────┐
│                    API Server 进程                                   │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  FastAPI Server ←→ LLMEngine (客户端)                       │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                           │                                         │
│                           ▼                                         │
│                    RPC 通信通道                                      │
│                           │                                         │
│                           ▼                                         │
┌─────────────────────────────────────────────────────────────────────┐
│                    Engine Core 进程                                 │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  KVCacheManager (跨卡共享)                                 │    │
│  │  Scheduler (任务调度)                                       │    │
│  │  OutputProcessor (结果汇总)                                 │    │
│  └─────────────────────────────────────────────────────────────┘    │
│           │                    │                    │              │
│           ▼                    ▼                    ▼              │
│  ┌───────────────┐    ┌───────────────┐    ┌───────────────┐       │
│  │   Worker 0   │    │   Worker 1   │    │  Worker N-1  │       │
│  │   (GPU 0)    │    │   (GPU 1)    │    │  (GPU N-1)  │       │
│  │               │    │               │    │               │       │
│  │  ModelShard   │    │  ModelShard   │    │  ModelShard   │       │
│  │  - Layer 0~K  │    │  - Layer K+1~ │    │  - Layer ...  │       │
│  │               │    │               │    │               │       │
│  │  CUDA Stream  │    │  CUDA Stream  │    │  CUDA Stream  │       │
│  └───────────────┘    └───────────────┘    └───────────────┘       │
└─────────────────────────────────────────────────────────────────────┘
```

### 进程数量

```
总进程数 = 1 (API Server) + 1 (Engine Core) + N (Workers)
        = N + 2

其中 N = GPU 数量
```

**示例**：4 卡机器
- 总进程数 = 1 + 1 + 4 = **6 个进程**

### 启动命令

```bash
# 启动 4 卡推理服务
VLLM_TARGET_DEVICE=cuda python -m vllm.entrypoints.cli.main serve distilgpt2 \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.9 \
  --max-num-seqs 16 \
  --port 8000
```

### 详细启动过程

单机多卡模式下，服务启动涉及**多个关键对象**的创建和初始化，完整的启动流程如下：

#### 启动流程总览

```
用户启动命令
    │
    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 阶段 1：命令行解析与配置创建                                         │
├─────────────────────────────────────────────────────────────────────┤
│  1.1 EngineArgs.from_cli_args()  - 解析命令行参数                   │
│  1.2 VllmConfig.create_engine_config()  - 创建 VLLM 配置对象        │
│  1.3 ParallelConfig 创建  - 配置并行策略（tensor-parallel-size）     │
└─────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 阶段 2：进程创建与初始化                                             │
├─────────────────────────────────────────────────────────────────────┤
│  2.1 EngineCoreClient.make_client()  - 创建引擎客户端               │
│       ├── multiprocess_mode=True  - 启用多进程模式                  │
│       └── asyncio_mode=False  - 同步引擎                            │
│                                                                     │
│  2.2 创建 API Server 进程                                          │
│       └── LLMEngine (客户端代理)  - 运行在 API Server 进程          │
│                                                                     │
│  2.3 创建 Engine Core 进程                                          │
│       └── EngineCore  - 运行在独立进程                              │
│                                                                     │
│  2.4 创建 N 个 Worker 进程（每个 GPU 一个）                        │
│       └── WorkerProcess_i  - 运行在各 GPU 上                        │
└─────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 阶段 3：模型加载与初始化                                             │
├─────────────────────────────────────────────────────────────────────┤
│  3.1 GPU 初始化                                                     │
│       ├── CUDA Context 创建                                        │
│       ├── NCCL 通信组初始化                                         │
│       └── 分布式存储初始化                                          │
│                                                                     │
│  3.2 模型权重加载                                                   │
│       ├── 从 HuggingFace 加载模型权重                               │
│       ├── 模型分片（Tensor Parallel）                               │
│       └── 权重广播到各 Worker                                       │
│                                                                     │
│  3.3 KV Cache 初始化                                               │
│       ├── PagedAttention 页表分配                                  │
│       └── 跨 Worker 共享内存设置                                   │
└─────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 阶段 4：服务启动与就绪                                               │
├─────────────────────────────────────────────────────────────────────┤
│  4.1 FastAPI Server 绑定端口                                       │
│  4.2 健康检查端点注册                                               │
│  4.3 API 路由注册 (/v1/completions, /v1/models, etc.)              │
│  4.4 服务就绪，等待请求                                             │
└─────────────────────────────────────────────────────────────────────┘
```

#### 关键对象说明

##### 1. EngineArgs（引擎参数）

**类路径**：`vllm/engine/arg_utils.py`

**职责**：解析和存储命令行参数

**关键属性**：
```python
class EngineArgs:
    model: str                      # 模型名称或路径
    tensor_parallel_size: int       # 张量并行大小（GPU 数量）
    pipeline_parallel_size: int     # 流水线并行大小
    gpu_memory_utilization: float  # GPU 内存利用率
    max_num_seqs: int              # 最大并发序列数
    dtype: str                     # 数据类型（float16/bfloat16）
    host: str                      # 监听地址
    port: int                      # 监听端口
```

**创建时机**：命令行解析时

---

##### 2. VllmConfig（VLLM 配置）

**类路径**：`vllm/config.py`

**职责**：统一的配置对象，整合所有配置子模块

**关键属性**：
```python
class VllmConfig:
    model_config: ModelConfig       # 模型配置
    cache_config: CacheConfig       # KV 缓存配置
    parallel_config: ParallelConfig  # 并行策略配置
    scheduler_config: SchedulerConfig  # 调度器配置
    device_config: DeviceConfig     # 设备配置
```

**创建时机**：调用 `engine_args.create_engine_config()`

---

##### 3. ParallelConfig（并行配置）

**类路径**：`vllm/config.py`

**职责**：配置并行策略参数

**关键属性**：
```python
class ParallelConfig:
    tensor_parallel_size: int      # 张量并行大小
    pipeline_parallel_size: int    # 流水线并行大小
    data_parallel_size: int        # 数据并行大小
    world_size: int               # 总进程数
    rank: int                     # 当前进程 rank
```

---

##### 4. EngineCoreClient（引擎客户端）

**类路径**：`vllm/v1/engine/core_client.py`

**职责**：管理 Engine Core 进程的生命周期，提供透明的 RPC 调用

**关键方法**：
```python
class EngineCoreClient:
    @classmethod
    def make_client(cls, multiprocess_mode, asyncio_mode, ...):
        """根据模式创建客户端"""
        if multiprocess_mode:
            # 创建独立进程
            return cls._create_multiprocess_client(...)
        else:
            # 单进程模式
            return cls._create_inprocess_client(...)

    def add_request(self, request):
        """添加推理请求（RPC 调用）"""

    def get_output(self, request_id):
        """获取推理结果（RPC 调用）"""
```

**创建时机**：在 `LLMEngine.__init__()` 中创建

---

##### 5. EngineCore（引擎核心）

**类路径**：`vllm/v1/engine/core.py`

**职责**：推理引擎的核心实现，负责任务调度和 KV 缓存管理

**关键组件**：
```python
class EngineCore:
    def __init__(self, vllm_config, executor_class):
        # 模型执行器
        self.model_executor: ModelExecutor

        # 调度器
        self.scheduler: Scheduler

        # KV 缓存管理器
        self.kv_cache_manager: KVCacheManager

        # 输入处理器
        self.input_processor: InputProcessor

        # 输出处理器
        self.output_processor: OutputProcessor

    def execute_model(self, requests):
        """执行模型推理"""
        # 1. 调度任务
        scheduled = self.scheduler.schedule(requests)

        # 2. 分配 KV 缓存
        self.kv_cache_manager.allocate(scheduled)

        # 3. 执行推理
        outputs = self.model_executor.execute(scheduled)

        # 4. 处理输出
        return self.output_processor.process(outputs)
```

**创建时机**：在 Worker 进程初始化时

---

##### 6. ModelExecutor（模型执行器）

**类路径**：`vllm/executor/`

**职责**：在指定设备上执行模型推理

**子类**：
```python
# CPU 执行器
class CPUExecutor(ModelExecutor):
    def __init__(self, vllm_config):
        self.model = load_model_to_cpu(vllm_config)

    def execute(self, requests):
        return self.model.forward(requests)

# GPU 执行器（数据并行）
class GPUExecutor(ModelExecutor):
    def __init__(self, vllm_config):
        # 初始化 GPU
        init_gpu_devices(vllm_config)

        # 初始化 NCCL
        init_nccl(vllm_config)

        # 加载模型
        self.model = load_model_to_gpu(vllm_config)

    def execute(self, requests):
        # 数据并行分片
        shard = shard_requests(requests, self.rank, self.world_size)

        # 执行推理
        outputs = self.model.forward(shard)

        # NCCL All-Reduce
        outputs = self.nccl_all_reduce(outputs)

        return outputs
```

**创建时机**：在 `EngineCore.__init__()` 中创建

---

##### 7. Scheduler（调度器）

**类路径**：`vllm/v1/engine/core.py`

**职责**：管理请求队列，决定何时以及如何执行推理

**关键方法**：
```python
class Scheduler:
    def schedule(self, requests):
        """调度请求批次"""
        # 1. 接收新请求
        new_requests = [r for r in requests if r.is_new]

        # 2. 合并到等待队列
        self.waiting_queue.extend(new_requests)

        # 3. 动态批处理
        batch = self._batch_requests(self.waiting_queue)

        # 4. 分配 GPU 时间片
        return ScheduledBatch(
            requests=batch,
            num_tokens=sum(r.num_tokens for r in batch)
        )

    def _batch_requests(self, requests):
        """合并符合条件的请求为批次"""
        # 根据 max_num_seqs 和 max_num_batched_tokens 合并
        batch = []
        total_tokens = 0

        for req in requests:
            if (len(batch) < max_num_seqs and
                total_tokens + req.num_tokens <= max_num_batched_tokens):
                batch.append(req)
                total_tokens += req.num_tokens

        return batch
```

---

##### 8. KVCacheManager（KV 缓存管理器）

**类路径**：`vllm/v1/kv_cache/`

**职责**：管理 KV 缓存的分配和回收，使用 PagedAttention

**关键方法**：
```python
class KVCacheManager:
    def __init__(self, vllm_config):
        self.page_size = vllm_config.cache_config.block_size
        self.num_blocks = vllm_config.cache_config.gpu_memory_utilization

    def allocate(self, scheduled_batch):
        """为批次分配 KV 缓存块"""
        for req in scheduled_batch.requests:
            num_blocks_needed = ceil(req.num_tokens / self.page_size)

            # 查找空闲块
            free_blocks = self._find_free_blocks(num_blocks_needed)

            # 分配块
            req.cache_blocks = free_blocks

            # 更新块状态
            self._update_block_status(free_blocks, req.request_id)

    def free_completed(self, completed_requests):
        """回收已完成请求的缓存块"""
        for req in completed_requests:
            self._update_block_status(req.cache_blocks, None)
```

---

##### 9. Worker 进程

**类路径**：`vllm/worker/`

**职责**：在特定 GPU 上执行模型推理的独立进程

**进程创建流程**：
```python
# EngineCoreClient._create_multiprocess_client()
def _create_multiprocess_client(self, ...):
    # 1. 创建 Worker 进程
    worker_processes = []
    for i in range(num_gpus):
        p = multiprocessing.Process(
            target=worker_main,
            args=(gpu_id, vllm_config, ...),
            name=f"Worker-{i}"
        )
        p.start()
        worker_processes.append(p)

    # 2. 创建 NCCL 通信组
    nccl_group = NCCLGroup(worker_processes)

    # 3. 返回 RPC 客户端
    return EngineCoreClient(rpc_server_address)
```

**Worker 主函数**：
```python
# vllm/worker/worker.py
def worker_main(gpu_id, vllm_config, ...):
    # 1. 设置 GPU
    torch.cuda.set_device(gpu_id)

    # 2. 创建 EngineCore
    engine_core = EngineCore(vllm_config, GPUExecutor)

    # 3. 进入工作循环
    while not shutdown_signal:
        # 等待任务
        task = rpc_server.receive_task()

        # 执行推理
        output = engine_core.execute_model(task)

        # 返回结果
        rpc_server.send_result(output)
```

---

##### 10. LLMEngine（引擎封装器）

**类路径**：`vllm/v1/engine/llm_engine.py`

**职责**：提供高层次的推理接口，协调输入输出处理

**关键方法**：
```python
class LLMEngine:
    def __init__(self, vllm_config, executor_class, ...):
        # 创建输入处理器
        self.input_processor = InputProcessor(vllm_config)

        # 创建输出处理器
        self.output_processor = OutputProcessor(vllm_config)

        # 创建引擎核心客户端
        self.engine_core = EngineCoreClient.make_client(...)

    def add_request(self, request):
        """添加推理请求"""
        # 1. 处理输入
        engine_input = self.input_processor.process(request)

        # 2. 发送到 Engine Core
        self.engine_core.add_request(engine_input)

    async def generate(self, prompts, sampling_params):
        """生成文本"""
        # 1. 创建请求
        request = Request(prompts, sampling_params)

        # 2. 添加请求
        self.add_request(request)

        # 3. 等待结果
        while not request.is_finished:
            output = await self.engine_core.get_output(request.id)
            yield output

        # 4. 后处理输出
        return self.output_processor.process(request.outputs)
```

---

#### 启动时序图

```
用户                API Server          Engine Core         Worker 0-N
 │                     │                    │                   │
 │  启动命令            │                    │                   │
 │────────────────────>│                    │                   │
 │                     │                    │                   │
 │  解析参数            │                    │                   │
 │                     │                    │                   │
 │  创建 VllmConfig     │                    │                   │
 │────────────────────>│                    │                   │
 │                     │                    │                   │
 │                     │  创建多进程          │                   │
 │                     │───────────────────> │                   │
 │                     │                    │    创建 Worker    │
 │                     │                    │─────────────────>│
 │                     │                    │                   │
 │                     │                    │  初始化 NCCL      │
 │                     │                    │───────────────────│
 │                     │                    │                   │
 │                     │                    │  加载模型权重      │
 │                     │                    │───────────────────│
 │                     │                    │                   │
 │                     │                    │  初始化 KV Cache  │
 │                     │                    │───────────────────│
 │                     │                    │                   │
 │                     │  服务就绪           │                   │
 │<────────────────────│                    │                   │
 │                     │                    │                   │
 │  HTTP 请求           │                    │                   │
 │────────────────────>│                    │                   │
 │                     │                    │                   │
 │                     │  调度任务           │                   │
 │                     │───────────────────>│                   │
 │                     │                    │                   │
 │                     │                    │  执行推理          │
 │                     │                    │<──────────────────│
 │                     │                    │                   │
 │                     │  返回结果           │                   │
 │<────────────────────│                    │                   │
 │                     │                    │                   │
```

#### 启动日志示例

```bash
$ VLLM_TARGET_DEVICE=cuda python -m vllm.entrypoints.cli.main serve distilgpt2 \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.9 \
    --port 8000

# 预期输出日志
INFO 05-16 10:30:00 [arg_utils.py:123] Parsing engine arguments...
INFO 05-16 10:30:00 [config.py:456] Creating VllmConfig...
INFO 05-16 10:30:00 [config.py:789]   - Model: distilgpt2
INFO 05-16 10:30:01 [config.py:789]   - Tensor Parallel Size: 4
INFO 05-16 10:30:01 [config.py:789]   - GPU Memory Utilization: 0.9
INFO 05-16 10:30:01 [core_client.py:111] Creating multiprocess engine client...
INFO 05-16 10:30:01 [core_client.py:222] Starting Engine Core process (PID: 12345)
INFO 05-16 10:30:02 [worker.py:333] Initializing GPU 0...
INFO 05-16 10:30:02 [worker.py:333] Initializing GPU 1...
INFO 05-16 10:30:02 [worker.py:333] Initializing GPU 2...
INFO 05-16 10:30:02 [worker.py:333] Initializing GPU 3...
INFO 05-16 10:30:03 [nccl.py:444] NCCL initialized (World Size: 4)
INFO 05-16 10:30:05 [model_loader.py:555] Loading model weights...
INFO 05-16 10:30:10 [model_loader.py:555]   - Loading layer 0-12 to GPU 0
INFO 05-16 10:30:10 [model_loader.py:555]   - Loading layer 13-24 to GPU 1
INFO 05-16 10:30:10 [model_loader.py:555]   - Loading layer 25-36 to GPU 2
INFO 05-16 10:30:10 [model_loader.py:555]   - Loading layer 37-48 to GPU 3
INFO 05-16 10:30:12 [kv_cache.py:666] Initializing KV cache (total blocks: 8192)
INFO 05-16 10:30:12 [api_server.py:777] Starting FastAPI server on 0.0.0.0:8000
INFO 05-16 10:30:12 [api_server.py:888] Server ready, accepting requests
```

---

### 并行策略

#### 1. 数据并行（Data Parallelism）

- 每个 Worker 持有完整的模型副本
- 输入数据分片到不同 Worker
- 结果通过 All-Reduce 汇总

```python
# 数据并行示意
class DataParallelExecutor:
    def execute(self, requests, worker_id, num_workers):
        # 分片请求
        shard = requests[worker_id::num_workers]
        # 各自推理
        outputs = self.model.forward(shard)
        # All-Reduce 汇总
        outputs = self.nccl_group.all_reduce(outputs)
        return outputs
```

#### 2. 流水线并行（Pipeline Parallelism，可选）

- 模型按层分割到不同 GPU
- 减少单卡显存占用
- 支持超大模型

```python
# 流水线并行示意
Worker 0: Layer 0~12  ──┐
Worker 1: Layer 13~24  ──┼──► Forward Pass
Worker 2: Layer 25~36  ──┤
Worker 3: Layer 37~48  ──┘
```

### 通信方式

| 通信方向 | 通信方式 | 带宽需求 | 用途 |
|----------|----------|----------|------|
| API Server ↔ Engine Core | RPC（TCP/Unix Socket） | 低 | 请求转发 |
| Engine Core ↔ Workers | 共享内存 + Queue | 中 | 任务分发 |
| Workers ↔ Workers | NCCL | 高 | 数据并行同步 |
| Workers 内部 | CUDA IPC | 极高 | GPU 间数据传输 |

### 适用场景

| 场景 | 推荐度 | 说明 |
|------|--------|------|
| 中等规模推理 | ⭐⭐⭐⭐⭐ | 适合 QPS 10-100 |
| 大模型部署 | ⭐⭐⭐⭐⭐ | 单卡无法容纳的模型 |
| 高吞吐需求 | ⭐⭐⭐⭐⭐ | 多卡并行加速 |
| 生产环境 | ⭐⭐⭐⭐⭐ | 推荐的主要部署方式 |

### 资源使用

| 资源 | 使用情况 |
|------|----------|
| CPU | 1 核用于调度 + N 核用于 Worker |
| GPU | N 张卡，每卡完整模型或分片 |
| 内存 | N × 模型大小 |
| NVLink | GPU 间高速互联（推荐） |

---

## 多机多卡模式

### 架构概述

多机多卡模式在**多个服务器**上部署，通过高速网络互联实现分布式推理。

### 进程结构

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Node 0                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  API Server + Engine Core 进程                              │    │
│  │  （可配置为仅 API Server，由 Node 0 统一调度）              │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                           │                                         │
│                           │ 高速网络 (InfiniBand/RoCE)              │
│                           ▼                                         │
├─────────────────────────────────────────────────────────────────────┤
│                        Node 0 Workers                              │
│  ┌───────────────┐    ┌───────────────┐    ┌───────────────┐       │
│  │   Worker 0   │    │   Worker 1   │    │  Worker N-1  │       │
│  │   (GPU 0)    │    │   (GPU 1)    │    │  (GPU N-1)   │       │
│  └───────────────┘    └───────────────┘    └───────────────┘       │
├─────────────────────────────────────────────────────────────────────┤
│                           │                                         │
│                           │ 高速网络                                 │
│                           ▼                                         │
├─────────────────────────────────────────────────────────────────────┤
│                        Node 1 Workers                              │
│  ┌───────────────┐    ┌───────────────┐    ┌───────────────┐       │
│  │   Worker M   │    │   Worker M+1 │    │  Worker 2N-1  │       │
│  │   (GPU 0)    │    │   (GPU 1)    │    │  (GPU N-1)   │       │
│  └───────────────┘    └───────────────┘    └───────────────┘       │
└─────────────────────────────────────────────────────────────────────┘
```

### 进程数量

```
总进程数 = 1 (API Server) + 1 (Engine Core) + M × N (Workers)
        = M × N + 2

其中：
  M = 节点数量
  N = 每节点 GPU 数量
```

**示例**：4 节点 × 4 卡 = 16 GPU
- 总进程数 = 1 + 1 + 16 = **18 个进程**

### 启动命令

```bash
# 在 Node 0 启动（作为主节点）
torchrun --nnodes=4 \
         --node_rank=0 \
         --nproc_per_node=4 \
         --master_addr=10.0.0.1 \
         --master_port=29500 \
         vllm(entrypoints.cli.main) serve distilgpt2 \
         --tensor-parallel-size 16 \
         --gpu-memory-utilization 0.9 \
         --max-num-seqs 64 \
         --port 8000
```

### 并行策略

#### 1. 数据并行（Data Parallelism）

- 跨节点的数据分片
- 通过 All-Reduce 汇总结果
- 需要高速网络支持

#### 2. 张量并行（Tensor Parallelism，可选）

- 模型张量分割到不同 GPU/节点
- 需要 NVLink + 高速网络
- 极低延迟通信

```python
# 张量并行示意
┌─────────────────────────────────────────────────┐
│           Attention 计算分片                    │
├─────────────────────────────────────────────────┤
│  Q = Q0 ╪ Q1 ╪ Q2 ╪ Q3  (按列分割)             │
│  K = K0 ╪ K1 ╪ K2 ╪ K3                        │
│  V = V0 ╪ V1 ╪ V2 ╪ V3                        │
│                                                 │
│  Attention = Softmax(Q × K^T / √d) × V        │
│  需要 All-Reduce 汇总                           │
└─────────────────────────────────────────────────┘
```

#### 3. 流水线并行（Pipeline Parallelism，可选）

- 模型按层分割到不同节点
- 减少单节点显存需求
- 支持超大模型

```python
# 流水线并行示意
Node 0: Layer 0~11   ─┐
Node 1: Layer 12~23  ─┼──► Micro-batch Pipeline
Node 2: Layer 24~35  ─┤
Node 3: Layer 36~47  ─┘
```

### 通信方式

| 通信方向 | 通信方式 | 带宽需求 | 用途 |
|----------|----------|----------|------|
| 跨节点通信 | NCCL + RDMA | 极高（100+ Gbps） | 数据并行同步 |
| 节点内通信 | NVLink + NCCL | 极高 | GPU 互联 |
| API Server ↔ Engine | RPC + TCP | 低 | 请求转发 |

### 适用场景

| 场景 | 推荐度 | 说明 |
|------|--------|------|
| 超大模型部署 | ⭐⭐⭐⭐⭐ | 单机无法容纳的模型 |
| 极高吞吐需求 | ⭐⭐⭐⭐⭐ | 分布式并行加速 |
| 集群环境 | ⭐⭐⭐⭐⭐ | 专用 GPU 集群 |
| 研究实验 | ⭐⭐⭐⭐ | 大规模推理实验 |

### 资源使用

| 资源 | 使用情况 |
|------|----------|
| CPU | M × (1 核调度 + N 核 Worker) |
| GPU | M × N 张卡 |
| 内存 | M × N × 模型大小 |
| 网络 | 高速网络（InfiniBand HDR/RoCE） |

---

## 进程间通信机制

### 1. RPC 通信

**用途**：API Server ↔ Engine Core 之间的请求转发

**实现**：基于 gRPC 或自定义 RPC 框架

```python
# RPC 客户端调用示例
class EngineCoreClient:
    def __init__(self, address):
        self.stub = EngineCoreStub(address)

    def add_request(self, request):
        # 通过 RPC 调用 Engine Core
        return self.stub.AddRequest(request)

    def get_output(self, request_id):
        return self.stub.GetOutput(request_id)
```

### 2. 共享内存通信

**用途**：Engine Core ↔ Workers 之间的数据传递

**实现**：Linux 共享内存 + 无锁队列

```python
# 共享内存通信示例
class SharedMemoryQueue:
    def __init__(self, size, shape, dtype):
        self.shm = multiprocessing.shared_memory.SharedMemory(
            create=True, size=np.prod(shape) * np.dtype(dtype).itemsize
        )
        self.queue = multiprocessing.Queue()  # 元数据队列

    def put(self, data):
        # 数据写入共享内存
        np.copyto(np.ndarray(self.shm.shape, self.shm.dtype, self.shm.buf), data)
        # 元数据放入队列
        self.queue.put({'shm_name': self.shm.name, 'shape': data.shape})
```

### 3. NCCL 通信

**用途**：Workers 之间的高性能集合通信

**支持的操作**：
- All-Reduce：多卡结果汇总
- All-Gather：收集所有 Worker 的数据
- Broadcast：广播数据到所有 Worker
- Reduce-Scatter：分散汇总

```python
# NCCL 通信示例
class NCCLCommunicator:
    def __init__(self, world_size, rank):
        import torch.distributed as dist
        dist.init_process_group(backend='nccl')
        self.group = dist.new_group(range(world_size))

    def all_reduce(self, tensor):
        dist.all_reduce(tensor, group=self.group)
        return tensor
```

### 4. CUDA IPC

**用途**：GPU 显存直接传输（单机多卡）

**实现**：CUDA 提供的进程间显存访问

```python
# CUDA IPC 示例
class CUDAIPCBridge:
    def send_tensor(self, tensor, target_rank):
        # 将 tensor 注册为可共享
        handle = torch.cuda.cuda.ipc.get_handle()
        # 发送句柄给目标进程
        self.send_handle(handle, target_rank)
```

### 通信性能对比

| 通信方式 | 延迟 | 带宽 | 开销 | 适用场景 |
|----------|------|------|------|----------|
| 直接函数调用 | ~0 | 内存带宽 | 极低 | 单进程 |
| 共享内存 | ~1μs | 内存带宽 | 低 | 节点内通信 |
| Unix Socket | ~10μs | 内存带宽 | 中 | 节点内 RPC |
| gRPC/TCP | ~100μs | 网络带宽 | 高 | 跨节点通信 |
| NCCL + RDMA | ~2μs | 100+ Gbps | 中 | 跨节点 GPU 通信 |

---

## 配置参数说明

### 关键启动参数

| 参数 | 说明 | CPU | 单机多卡 | 多机多卡 |
|------|------|-----|----------|----------|
| `--tensor-parallel-size` | 张量并行大小 | 1 | N | M×N |
| `--pipeline-parallel-size` | 流水线并行大小 | 1 | 1 | M |
| `--gpu-memory-utilization` | GPU 内存利用率 | N/A | 0.9 | 0.9 |
| `--max-num-seqs` | 最大并发序列数 | 1 | 16 | 64 |
| `--max-num-batched-tokens` | 最大批处理 token 数 | 256 | 8192 | 8192 |
| `--max-padding_length` | 最大填充长度 | 128 | 512 | 512 |

### 环境变量

| 环境变量 | 说明 | 推荐值 |
|----------|------|--------|
| `VLLM_TARGET_DEVICE` | 目标设备 | `cpu` / `cuda` |
| `VLLM_ENABLE_V1_MULTIPROCESSING` | 启用多进程模式 | `1` |
| `VLLM_WORKER_MULTIPROC_METHOD` | Worker 进程启动方式 | `forkserver` |
| `NCCL_DEBUG` | NCCL 调试日志 | `WARN` |
| `NCCL_IB_DISABLE` | 禁用 InfiniBand | `0` |

### 资源配置建议

#### CPU 单进程

```yaml
model: distilgpt2
max_num_seqs: 1
gpu_memory_utilization: 0.2
cpu_threads: 8
```

#### 单机多卡（4 卡 A100）

```yaml
model: llama-70b
tensor_parallel_size: 4
pipeline_parallel_size: 1
gpu_memory_utilization: 0.9
max_num_seqs: 16
max_num_batched_tokens: 8192
```

#### 多机多卡（4 节点 × 8 卡 A100）

```yaml
model: llama-280b
tensor_parallel_size: 32
pipeline_parallel_size: 4
gpu_memory_utilization: 0.9
max_num_seqs: 64
max_num_batched_tokens: 16384
```

---

## 设计优势与局限

### 优势

| 优势 | 说明 |
|------|------|
| **层次化设计** | API Server、Engine Core、Worker 职责分离，便于维护和扩展 |
| **透明性** | 多进程调用方式与单进程一致，降低使用成本 |
| **高性能** | 通过 PagedAttention、动态批处理最大化硬件利用率 |
| **灵活性** | 支持从单 CPU 到多机多卡的多种部署方式 |
| **容错性** | 多进程隔离，单组件故障不影响整体服务 |

### 局限

| 局限 | 说明 | 解决方案 |
|------|------|----------|
| **通信开销** | 多进程间通信有额外开销 | 使用高速网络、共享内存 |
| **资源需求** | 多进程需要更多资源 | 按需扩展、合理配置 |
| **调试复杂度** | 多进程调试困难 | 提供详细日志、调试工具 |
| **内存冗余** | 数据并行时模型副本占用内存 | 张量并行、流水线并行 |

### 性能优化建议

1. **CPU 模式**：增加 `cpu_threads`，启用 NUMA 亲和
2. **单机多卡**：启用 NVLink、使用张量并行
3. **多机多卡**：使用 RDMA 网络、优化批处理参数
4. **通用**：合理设置 `max_num_seqs` 和 `gpu_memory_utilization`

---

## 总结

vLLM 的多进程架构设计实现了：

1. **可扩展性**：从单进程到多机多卡，灵活适应不同规模
2. **高性能**：通过并行策略和优化技术最大化硬件利用率
3. **易用性**：统一的 API 接口，透明的进程调用
4. **可靠性**：进程隔离，良好的容错能力

### 场景选择指南

| 场景 | 推荐模式 | 进程数 |
|------|----------|--------|
| 开发调试 | CPU 单进程 | 1 |
| 小规模生产 | CPU 单进程 / 单机多卡 | 1 / N+2 |
| 中等规模生产 | 单机多卡 | N+2 |
| 大模型生产 | 单机多卡 / 多机多卡 | N+2 / M×N+2 |
| 超大模型 | 多机多卡 | M×N+2 |

---

**文档信息**

- 作者：vLLM Team
- 版本：v1.0
- 更新日期：2026-05-16
- 许可：Apache 2.0

---

# 补充章节：ZMQ 通信架构深度解析

> 本章节补充 vLLM v1 多进程架构的详细分析和 ZMQ 通信原理
> 
> 更新日期：2026-06-12

---

## 补充一、进程架构总览

### 1.1 进程层次结构图

#### 场景1: 单GPU推理 (TP=1, PP=1, DP=1) - InprocClient

```
┌──────────────────────────┐
│   API Server Process     │
│  ┌────────────────────┐  │
│  │   LLMEngine        │  │
│  │  ┌──────────────┐  │  │
│  │  │ EngineCore   │  │  │
│  │  │ (inline)     │  │  │
│  │  └──────────────┘  │  │
│  │  ┌──────────────┐  │  │
│  │  │ Worker       │  │  │
│  │  │ (inline)     │  │  │
│  │  └──────────────┘  │  │
│  └────────────────────┘  │
└──────────────────────────┘

特点：
- 所有组件运行在同一进程内
- 无进程间通信开销
- 适合简单推理场景
```

#### 场景2: 多GPU张量并行 (TP>1) - MultiprocExecutor

```
┌────────────────────────────────────────────────────────┐
│              API Server Process                        │
│  ┌──────────────────────────────────────────────────┐  │
│  │            LLMEngine                             │  │
│  │  ┌────────────────────────────────────────────┐  │  │
│  │  │      EngineCore (Scheduler)                │  │  │
│  │  └────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────┘
          │                    │                    │
          │ MessageQueue       │ MessageQueue       │ MessageQueue
          ▼                    ▼                    ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│  WorkerProc_0    │  │  WorkerProc_1    │  │  WorkerProc_n    │
│  ┌────────────┐  │  │  ┌────────────┐  │  │  ┌────────────┐  │
│  │ GPU_0      │  │  │  │ GPU_1      │  │  │  │ GPU_n      │  │
│  │ TP_rank_0  │  │  │  │ TP_rank_1  │  │  │  │ TP_rank_n  │  │
│  └────────────┘  │  │  └────────────┘  │  │  └────────────┘  │
└──────────────────┘  └──────────────────┘  └──────────────────┘
          │                    │                    │
          └────────────────────┼────────────────────┘
                               │ NCCL ProcessGroup
                               │ (All-Reduce, All-Gather)
```

#### 场景3: 多GPU数据并行 (DP>1) - Internal Load Balancer

```
┌────────────────────────────────────────────────────────────┐
│                 API Server Process (Rank 0)                │
│  ┌────────────────────────────────────────────────────────┐│
│  │                  LLMEngine                             ││
│  │  ┌──────────────────────────────────────────────────┐  ││
│  │  │        DPCoordinatorProc (协调器)                │  ││
│  │  │   - 收集队列统计                                  │  ││
│  │  │   - 负载均衡决策                                  │  ││
│  │  └──────────────────────────────────────────────────┘  ││
│  └────────────────────────────────────────────────────────┘│
└────────────────────────────────────────────────────────────┘
          │ ZMQ DEALER              │ ZMQ DEALER
          ▼                         ▼
┌──────────────────────┐    ┌──────────────────────┐
│ EngineCoreProc_DP0   │    │ EngineCoreProc_DP1   │
│ ┌────────────────┐   │    │ ┌────────────────┐   │
│ │   Scheduler    │   │    │ │   Scheduler    │   │
│ └────────────────┘   │    │ └────────────────┘   │
│ ┌─────┐ ┌─────┐      │    │ ┌─────┐ ┌─────┐      │
│ │W_0_0│ │W_0_1│      │    │ │W_1_0│ │W_1_1│      │
│ └─────┘ └─────┘      │    │ └─────┘ └─────┘      │
└──────────────────────┘    └──────────────────────┘
```

---

## 补充二、ZMQ 通信原理详解

### 2.1 ZMQ 简介

**ZeroMQ (ZMQ)** 是一个高性能的异步消息库，提供了多种消息模式。vLLM 选择 ZMQ 的原因：

1. **高性能**: 比 TCP socket 快 10-100 倍
2. **多种模式**: REQ-REP、PUB-SUB、PUSH-PULL 等
3. **零拷贝**: 支持 send/recv 的零拷贝传输
4. **异步**: 非阻塞 I/O，支持高并发
5. **跨语言**: C++、Python、Java 等多种语言支持

---

### 2.2 ZMQ Socket 类型

#### 1. ROUTER-DEALER

**模式**：异步请求-响应，支持多客户端

```
┌────────────┐                  ┌────────────┐
│  ROUTER    │◀────────────────▶│  DEALER    │
│ (Frontend) │  Identity + Msg  │ (Backend)  │
└────────────┘                  └────────────┘
     ▲   ▲                            │
     │   │                            │
     │   └────────────────────────────┘
     │
┌────┴────┐
│Client 1 │
└─────────┘

特点：
- ROUTER 自动添加/移除 Identity（客户端标识）
- DEALER 自动处理路由
- 支持多客户端
- 完全异步
```

**vLLM 中的应用**：

```python
# API Server (ROUTER)
frontend_socket = zmq.Context().socket(zmq.ROUTER)
frontend_socket.bind("tcp://*:5555")

# EngineCore (DEALER)
backend_socket = zmq.Context().socket(zmq.DEALER)
backend_socket.connect("tcp://localhost:5555")

# 发送消息（ROUTER 端）
# 消息格式：[identity][empty][data]
frontend_socket.send_multipart([identity, b"", data])

# 接收消息（DEALER 端）
# 自动剥离 identity
data = backend_socket.recv()
```

**消息路由流程**：

```
1. Client → ROUTER: [data]
2. ROUTER 接收，自动添加 identity: [client_id][data]
3. ROUTER → DEALER: [client_id][data]
4. DEALER 接收，自动剥离 identity: [data]
5. DEALER → ROUTER (响应): [response]
6. ROUTER 接收，自动添加 identity: [client_id][response]
7. ROUTER → Client: 自动剥离 identity: [response]
```

---

#### 2. PUSH-PULL

**模式**：单向数据流，支持负载均衡

```
┌────────────┐                  ┌────────────┐
│    PUSH    │──────────────────▶│    PULL    │
│ (Producer) │   One-way Stream  │ (Consumer) │
└────────────┘                  └────────────┘

多生产者-多消费者:
┌────────┐  ┌────────┐
│ PUSH_1 │  │ PUSH_2 │
└────┬───┘  └────┬───┘
     │           │
     └─────┬─────┘
           │
     ┌─────┴─────┐
     │           │
┌────▼───┐  ┌────▼───┐
│ PULL_1 │  │ PULL_2 │
└────────┘  └────────┘

特点：
- 单向通信
- 自动负载均衡（轮询分配）
- 消息不丢失（已发送的消息必达）
```

**vLLM 中的应用**：

```python
# EngineCore (PUSH) 发送输出
output_socket = zmq.Context().socket(zmq.PUSH)
output_socket.bind("tcp://*:5556")

# API Server (PULL) 接收输出
input_socket = zmq.Context().socket(zmq.PULL)
input_socket.connect("tcp://localhost:5556")

# EngineCore 发送
output_socket.send_multipart([output_data])

# API Server 接收
output_data = input_socket.recv_multipart()
```

---

#### 3. PUB-SUB (XPUB-XSUB)

**模式**：发布-订阅，支持主题过滤

```
标准 PUB-SUB:
┌────────────┐                  ┌────────────┐
│    PUB     │──────────────────▶│    SUB     │
│ (Publisher)│   Broadcast Msg   │(Subscriber)│
└────────────┘                  └────────────┘

XPUB-XSUB (可扩展):
┌────────────┐    XSUB    ┌──────────┐    XPUB    ┌────────────┐
│ Publisher  │◀───────────│  Proxy   │───────────▶│ Subscriber │
└────────────┘            └──────────┘            └────────────┘

特点：
- 一对多广播
- 支持主题过滤（订阅特定主题）
- XPUB/XSUB 支持代理转发
```

**vLLM 中的应用**（DP Coordinator）：

```python
# DPCoordinator (XPUB) 发布状态
publish_socket = zmq.Context().socket(zmq.XPUB)
publish_socket.bind("tcp://*:5557")

# EngineCore (XSUB) 订阅状态
subscribe_socket = zmq.Context().socket(zmq.XSUB)
subscribe_socket.connect("tcp://localhost:5557")

# 订阅特定主题
subscribe_socket.setsockopt(zmq.SUBSCRIBE, b"stats")
subscribe_socket.setsockopt(zmq.SUBSCRIBE, b"decisions")

# DPCoordinator 发布
publish_socket.send_multipart([b"stats", stats_data])

# EngineCore 接收
topic, data = subscribe_socket.recv_multipart()
```

---

### 2.3 ZMQ 消息模式对比

| 模式 | 通信方向 | 消息保证 | 适用场景 | vLLM 应用 |
|------|---------|---------|----------|-----------|
| REQ-REP | 双向同步 | 严格顺序 | 简单 RPC | - |
| ROUTER-DEALER | 双向异步 | 无序 | 多客户端 | 请求分发 |
| PUSH-PULL | 单向 | 不丢失 | 数据流 | 输出传输 |
| PUB-SUB | 单向广播 | 可能丢失 | 状态广播 | DP 协调 |

---

### 2.4 ZMQ 高级特性

#### 1. 零拷贝传输

```python
# 传统方式（数据拷贝）
data = bytearray(1000000)  # 1MB 数据
socket.send(data)  # 拷贝到 ZMQ buffer

# 零拷贝方式
data = bytearray(1000000)
tracker = socket.send(data, copy=False, track=True)

# 等待发送完成（可选）
tracker.wait()

# 优势：大消息传输时避免拷贝，提高性能
```

**vLLM 应用**：

```python
# vllm/v1/engine/core_client.py
def _send_input(self, request_type, request):
    msg = (self.core_engine, request_type.value, *self.encoder.encode(request))
    
    if len(msg) <= 3:
        # 小消息，直接发送
        self.input_socket.send_multipart(msg, copy=False)
    else:
        # 大消息，使用零拷贝 + 追踪
        tracker = self.input_socket.send_multipart(msg, copy=False, track=True)
        self.add_pending_message(tracker, request)
```

---

#### 2. 多部分消息

```python
# 发送多部分消息
socket.send_multipart([
    b"part1",  # 第一部分
    b"part2",  # 第二部分
    b"part3",  # 第三部分
])

# 接收多部分消息
parts = socket.recv_multipart()
# parts = [b"part1", b"part2", b"part3"]

# 优势：
# - 避免手动拼接消息
# - 零拷贝传递各部分
# - 原子性：要么全部发送，要么全部不发送
```

**vLLM 应用**：

```python
# 消息格式：[identity][type][data]
socket.send_multipart([
    engine_identity,              # 2 bytes
    EngineCoreRequestType.ADD.value,  # 1 byte
    *encoded_request,             # 多个 buffer
])
```

---

#### 3. 非阻塞 I/O

```python
# 同步阻塞
message = socket.recv()  # 阻塞直到收到消息

# 非阻塞（立即返回）
try:
    message = socket.recv(zmq.NOBLOCK)
except zmq.Again:
    # 没有消息
    pass

# 使用 Poller（推荐）
poller = zmq.Poller()
poller.register(socket, zmq.POLLIN)

# 等待消息（带超时）
if poller.poll(timeout=1000):  # 1秒超时
    message = socket.recv()
else:
    # 超时
    pass
```

---

### 2.5 ZMQ 性能优化技巧

#### 1. 连接管理

```python
# 设置高水位标记（High Water Mark）
# 防止消息堆积导致内存爆炸
socket.setsockopt(zmq.SNDHWM, 1000)  # 发送队列最多 1000 条
socket.setsockopt(zmq.RCVHWM, 1000)  # 接收队列最多 1000 条

# 设置缓冲区大小
socket.setsockopt(zmq.SNDBUF, 1024 * 1024)  # 1MB 发送缓冲
socket.setsockopt(zmq.RCVBUF, 1024 * 1024)  # 1MB 接收缓冲
```

#### 2. 连接方式对比

| 连接方式 | 延迟 | 吞吐量 | 适用场景 |
|---------|------|--------|----------|
| TCP | ~50μs | 中等 | 跨机器、跨容器 |
| IPC | ~10μs | 高 | 本地进程间 |
| INPROC | ~1μs | 极高 | 同一线程内 |

---

## 补充三、详细时序图

### 3.1 请求处理完整时序图

```
┌─────────┐     ┌──────────┐     ┌────────────┐     ┌─────────┐     ┌────────┐
│  User   │     │API Server│     │EngineCore  │     │Scheduler│     │Workers │
└────┬────┘     └────┬─────┘     └─────┬──────┘     └────┬────┘     └───┬────┘
     │               │                  │                 │              │
     │ generate()    │                  │                 │              │
     ├──────────────▶│                  │                 │              │
     │               │                  │                 │              │
     │               │ InputProcessor   │                 │              │
     │               ├─────────┐        │                 │              │
     │               │         │tokenize│                 │              │
     │               │<────────┘        │                 │              │
     │               │                  │                 │              │
     │               │ ZMQ SEND (ADD)   │                 │              │
     │               ├─────────────────▶│                 │              │
     │               │                  │                 │              │
     │               │                  │ add_request()   │              │
     │               │                  ├────────────────▶│              │
     │               │                  │                 │              │
     │               │                  │ SchedulerOutput │              │
     │               │                  │<────────────────│              │
     │               │                  │                 │              │
     │               │                  │ MessageQueue.broadcast         │
     │               │                  ├────────────────────────────────▶
     │               │                  │                 │              │
     │               │                  │                 │ execute_model()
     │               │                  │                 │              ├────┐
     │               │                  │                 │              │<───┘
     │               │                  │                 │              │
     │               │                  │ MessageQueue.response          │
     │               │                  │◀────────────────────────────────┤
     │               │                  │                 │              │
     │               │ ZMQ SEND (Output)│                 │              │
     │               │◀─────────────────┤                 │              │
     │               │                  │                 │              │
     │ Response      │                  │                 │              │
     │◀──────────────┤                  │                 │              │
     │               │                  │                 │              │
```

---

## 补充四、性能优化总结

### 4.1 优化策略对比

| 优化技术 | 适用场景 | 性能提升 |
|---------|----------|----------|
| 零拷贝序列化 | 大消息传输 | 2-5x |
| 共享内存 MQ | Worker 通信 | 5-10x |
| 异步 I/O 线程 | 高并发场景 | 2-3x |
| 批处理 | 高吞吐场景 | 3-5x |
| ZMQ IPC | 本地进程间 | 2-3x |

---

> 补充章节版本：v1.1  
> 更新日期：2026-06-12  
> 作者：Claude (glm-5.1)


---

# 补充章节：vLLM 多进程架构完整分析

> 本章节提供 vLLM v1 多进程架构的深度分析，包括核心组件、通信机制、时序图和性能优化
> 
> 更新日期：2026-06-12

---

## 补充五、核心进程详解

### 5.1 EngineCoreProc

**职责**：
- 调度管理
- 请求队列管理
- KV Cache 管理
- 输出处理

**进程结构**：

```python
# vllm/v1/engine/core.py

class EngineCoreProc:
    """EngineCore 进程实现"""
    
    def __init__(self, vllm_config, dp_rank, ...):
        # 主要组件
        self.scheduler = Scheduler(...)
        self.model_executor = executor_class(...)
        
        # ZMQ Sockets
        self.input_socket = zmq.DEALER    # 接收请求
        self.output_socket = zmq.PUSH     # 发送响应
        
        # 后台线程
        self.input_thread = threading.Thread(
            target=self.process_input_socket,
            daemon=True,
        )
        self.output_thread = threading.Thread(
            target=self.process_output_socket,
            daemon=True,
        )
    
    @staticmethod
    def run_engine_core(vllm_config, dp_rank, ...):
        """进程入口点"""
        engine_core = EngineCoreProc(vllm_config, dp_rank, ...)
        engine_core.start()
    
    def process_input_socket(self):
        """处理输入消息的后台线程"""
        while not self.stopped:
            frames = self.input_socket.recv_multipart()
            request_type = EngineCoreRequestType(frames[0])
            
            if request_type == EngineCoreRequestType.ADD:
                request = self.decoder.decode(frames[1:])
                self.add_request(request)
            
            elif request_type == EngineCoreRequestType.ABORT:
                request_ids = self.decoder.decode(frames[1:])
                self.abort_requests(request_ids)
            
            elif request_type == EngineCoreRequestType.UTILITY:
                # 处理工具调用
                call_id, method, args = self.decoder.decode(frames[1:])
                result = getattr(self, method)(*args)
                self.send_utility_response(call_id, result)
    
    def step(self):
        """执行一次调度和推理"""
        # 1. 调度
        scheduler_output = self.scheduler.schedule()
        
        # 2. 执行模型
        output = self.model_executor.execute_model(scheduler_output)
        
        # 3. 处理输出
        self.process_output(output)
        
        # 4. 发送响应
        self.output_socket.send_multipart(
            self.encoder.encode(output)
        )
```

---

### 5.2 WorkerProc

**职责**：
- 模型加载
- 前向传播执行
- Token 采样
- KV Cache 操作

**进程结构**：

```python
# vllm/v1/worker/worker_base.py

class WorkerProc:
    """Worker 进程实现"""
    
    @staticmethod
    def worker_main(vllm_config, local_rank, rank, ...):
        """Worker 进程入口点"""
        # 1. 初始化设备
        init_device(local_rank)
        
        # 2. 创建 Worker
        worker = Worker(vllm_config, local_rank, rank, ...)
        
        # 3. 加载模型
        worker.load_model()
        
        # 4. 主循环
        worker.worker_busy_loop()
    
    def worker_busy_loop(self):
        """Worker 主循环"""
        while not self.stopped:
            # 从 MessageQueue 接收 RPC 调用
            method, args, kwargs, output_rank = self.rpc_broadcast_mq.dequeue()
            
            # 执行方法
            if method == "execute_model":
                output = self.execute_model(*args, **kwargs)
            elif method == "load_model":
                output = self.load_model(*args, **kwargs)
            elif method == "profile":
                output = self.profile(*args, **kwargs)
            else:
                output = getattr(self, method)(*args, **kwargs)
            
            # 返回结果（如果需要）
            if output_rank == self.rank:
                self.worker_response_mq.enqueue(output)
    
    def execute_model(self, scheduler_output):
        """执行模型前向传播"""
        # 1. 准备输入
        input_batch = self.prepare_input(scheduler_output)
        
        # 2. 执行前向传播
        hidden_states = self.model_runner.execute_model(input_batch)
        
        # 3. 采样（如果需要）
        if self.is_sampler_rank:
            sampled_tokens = self.sampler.sample(hidden_states)
        
        # 4. 返回输出
        return ModelRunnerOutput(
            hidden_states=hidden_states,
            sampled_tokens=sampled_tokens,
        )
```

---

### 5.3 DPCoordinatorProc

**职责**：
- 数据并行协调
- 负载均衡决策
- 统计收集
- 波次同步

**进程结构**：

```python
# vllm/v1/engine/coordinator.py

class DPCoordinator:
    """数据并行协调器"""
    
    def __init__(self, vllm_config, ...):
        # ZMQ Sockets
        self.coord_input_socket = zmq.XSUB   # 订阅引擎统计
        self.coord_output_socket = zmq.PUSH  # 发送指令
        self.coord_publish_socket = zmq.XPUB # 发布状态
        
        # 状态管理
        self.engine_stats = {}
        self.wave_state = {}
    
    def run(self):
        """协调器主循环"""
        while not self.stopped:
            # 1. 收集统计
            self.collect_stats()
            
            # 2. 做负载均衡决策
            decisions = self.make_lb_decisions()
            
            # 3. 发布决策
            self.publish_decisions(decisions)
            
            # 4. 管理波次
            self.manage_waves()
    
    def collect_stats(self):
        """收集所有引擎的统计信息"""
        while True:
            try:
                messages = self.coord_input_socket.recv_multipart(zmq.NOBLOCK)
                for msg in messages:
                    stats = self.decoder.decode(msg)
                    self.engine_stats[stats.engine_id] = stats
            except zmq.Again:
                break
    
    def make_lb_decisions(self):
        """负载均衡决策"""
        if not self.engine_stats:
            return {}
        
        # 最小队列长度策略
        min_queue_engine = min(
            self.engine_stats.items(),
            key=lambda x: x[1].queue_length
        )
        
        return {
            "target_engine": min_queue_engine[0],
            "reason": "least_loaded",
            "stats": {
                "queue_length": min_queue_engine[1].queue_length,
            },
        }
```

---

## 补充六、进程间通信架构

### 6.1 通信架构总览

```
┌─────────────────────────────────────────────────────────────────┐
│                    ZMQ 通信架构                                  │
└─────────────────────────────────────────────────────────────────┘

1. Request-Response 模式 (前端 ↔ EngineCore)
┌──────────────────┐                  ┌──────────────────┐
│   API Server     │                  │   EngineCore     │
│  ┌────────────┐  │   ZMQ ROUTER     │  ┌────────────┐  │
│  │   Client   │◄─┼──────────────────┼──│ InputSock  │  │
│  │            │  │  (request_id)    │  │ (DEALER)   │  │
│  └────────────┘  │                  │  └────────────┘  │
│  ┌────────────┐  │   ZMQ PULL       │  ┌────────────┐  │
│  │ OutputSock │◄─┼──────────────────┼──│ OutputSock │  │
│  │ (PULL)     │  │  (responses)     │  │ (PUSH)     │  │
│  └────────────┘  │                  │  └────────────┘  │
└──────────────────┘                  └──────────────────┘

消息格式:
┌─────────────┬──────────────┬──────────────────────┐
│ Identity    │ RequestType  │ SerializedData       │
│ (2 bytes)   │ (1 byte)     │ (msgpack encoded)    │
└─────────────┴──────────────┴──────────────────────┘
```

---

### 6.2 消息类型定义

```python
# vllm/v1/engine/__init__.py

class EngineCoreRequestType(enum.Enum):
    """请求类型枚举"""
    ADD = b"\x00"              # 添加推理请求
    ABORT = b"\x01"            # 中止请求
    START_DP_WAVE = b"\x02"    # 数据并行波次启动
    UTILITY = b"\x03"          # 工具方法调用
    EXECUTOR_FAILED = b"\x04"  # 执行器失败通知
    WAKEUP = b"\x05"           # 唤醒信号


@msgspec.struct
class EngineCoreRequest:
    """引擎核心请求"""
    request_id: str
    prompt_token_ids: list[int]
    sampling_params: SamplingParams
    arrival_time: float
    lora_request: LoRARequest | None
    trace_headers: dict[str, str] | None
    priority: int = 0


@msgspec.struct
class EngineCoreOutput:
    """引擎核心输出"""
    request_id: str
    output_token_ids: list[int]
    finished: bool
    stop_reason: str | None
```

---

### 6.3 共享内存消息队列

```
┌──────────────────┐                  ┌──────────────────┐
│  EngineCore      │                  │    Worker_0      │
│  ┌────────────┐  │   RingBuffer     │  ┌────────────┐  │
│  │Producer MQ │══╪══════════════════╪══│Consumer MQ │  │
│  │(SchedulerOut)│  │  Shared Memory   │  │            │  │
│  └────────────┘  │                  │  └────────────┘  │
└──────────────────┘                  └──────────────────┘
                                             │
                                             │ RingBuffer
                                             ▼
                                      ┌──────────────────┐
                                      │    Worker_1      │
                                      │  ┌────────────┐  │
                                      │  │Consumer MQ │  │
                                      │  └────────────┘  │
                                      └──────────────────┘
```

**MessageQueue 特性**：
- 基于共享内存，避免序列化开销
- RingBuffer 设计，支持高吞吐量
- SpinCondition 同步，平衡性能和 CPU 使用
- 支持本地和跨节点通信

---

## 补充七、详细时序图

### 7.1 多GPU张量并行时序图

```
┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│API Server│  │EngineCore│  │Scheduler │  │Worker_0  │  │Worker_1  │  │Worker_n  │
└────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘
     │             │             │             │             │             │
     │ add_request │             │             │             │             │
     ├────────────▶│             │             │             │             │
     │             │             │             │             │             │
     │             │ schedule()  │             │             │             │
     │             ├────────────▶│             │             │             │
     │             │             │             │             │             │
     │             │ SchedulerOut│             │             │             │
     │             │◀────────────┤             │             │             │
     │             │             │             │             │             │
     │             │ MessageQueue.broadcast (所有Workers同时接收)           │
     │             ├────────────────────────────┼─────────────┼─────────────▶
     │             │             │             │             │             │
     │             │             │   ┌─────────┼─────────────┼─────────────┤
     │             │             │   │         │             │             │
     │             │             │   │  Each worker executes forward       │
     │             │             │   │         │             │             │
     │             │             │   │  Worker_0: Layer_0 (shard_0)        │
     │             │             │   │  Worker_1: Layer_0 (shard_1)        │
     │             │             │   │  Worker_n: Layer_0 (shard_n)        │
     │             │             │   │         │             │             │
     │             │             │   │         │◄────────────┼─────────────┤
     │             │             │   │         │  NCCL       │             │
     │             │             │   │         │  All-Reduce │             │
     │             │             │   │         │             │             │
     │             │             │   └─────────┼─────────────┼─────────────┤
     │             │             │             │             │             │
     │             │ MessageQueue.response (Worker_0返回)    │             │
     │             │◀────────────────────────────┤             │             │
     │             │             │             │             │             │
     │ Output      │             │             │             │             │
     │◀────────────┤             │             │             │             │
     │             │             │             │             │             │
```

**张量并行通信模式**：
- **All-Reduce**: 每层计算后同步梯度或激活值
- **All-Gather**: 收集所有分片的结果
- **Reduce-Scatter**: 分发聚合结果到各 rank

---

### 7.2 数据并行负载均衡时序图

```
┌──────────┐  ┌───────────┐  ┌──────────┐  ┌──────────┐
│API Server│  │DPCoordinat│  │EngineCore│  │EngineCore│
│          │  │    or     │  │   DP0    │  │   DP1    │
└────┬─────┘  └─────┬─────┘  └────┬─────┘  └────┬─────┘
     │              │             │             │
     │              │◀────────────┼─────────────┤
     │              │  Periodic Stats Collection
     │              │  - queue_length              │
     │              │  - num_running_requests      │
     │              │  - gpu_memory_used           │
     │              │             │             │
     │  get_stats() │             │             │
     │◀─────────────┤             │             │
     │              │             │             │
     │  ┌───────────┼─────────────┼─────────────┤
     │  │           │             │             │
     │  │  Load Balancing Decision              │
     │  │  - EngineCore_DP0: queue_len=5        │
     │  │  - EngineCore_DP1: queue_len=3        │
     │  │  → Route to DP1 (less loaded)         │
     │  │           │             │             │
     │  └───────────┼─────────────┼─────────────┤
     │              │             │             │
     │ add_request  │             │             │
     ├─────────────────────────────────────────▶│
     │ (to DP1)     │             │             │
     │              │             │             │
     │ output       │             │             │
     │◀─────────────────────────────────────────┤
     │              │             │             │
```

**负载均衡策略**：
1. **最小队列长度**: 选择队列最短的引擎
2. **轮询 (Round-Robin)**: 依次分配请求
3. **资源感知**: 考虑 GPU 内存和计算资源

---

### 7.3 流水线并行时序图

```
Time: ─────────────────────────────────────────────────────▶

PP0: [Batch0]──────[Batch1]──────[Batch2]──────
     embed  send    embed  send   embed  send

PP1:       [Batch0]──────[Batch1]──────[Batch2]
           recv  layer   send  recv   layer  send

PP2:             [Batch0]──────[Batch1]──────[Batch2]
                 recv  layer   lm    recv   layer  lm

Output:                 ^Batch0       ^Batch1       ^Batch2
```

**流水线并行优势**：
- **隐藏延迟**: 不同阶段并行执行
- **提高吞吐**: 多批次同时处理
- **内存优化**: 每个阶段只保存部分模型

---

## 补充八、性能优化设计

### 8.1 零拷贝序列化

```python
# vllm/v1/engine/serial_utils.py

class MsgpackEncoder:
    """高效的 msgpack 编码器"""
    
    def encode_into(self, obj, buffer):
        """直接编码到 buffer，避免拷贝"""
        # 使用 buffer protocol
        ...
    
    def encode(self, obj) -> list[bytes]:
        """返回零拷贝的 buffer 列表"""
        buffers = []
        self.encode_into(obj, buffers)
        return buffers


# ZMQ zero-copy 发送
buffers = encoder.encode(request)
tracker = socket.send_multipart(buffers, copy=False, track=True)
```

**性能对比**：

| 方法 | 序列化时间 | 内存拷贝 | 适用场景 |
|------|-----------|----------|----------|
| JSON | 慢 | 多次拷贝 | 调试、小数据 |
| Pickle | 中等 | 多次拷贝 | Python 对象 |
| msgpack | 快 | 零拷贝 | 高性能场景 |

---

### 8.2 共享内存消息队列

```python
# vllm/distributed/device_communicators/shm_broadcast.py

class MessageQueue:
    """基于共享内存的高性能消息队列"""
    
    def __init__(self, size, max_bytes):
        # 创建共享内存 buffer
        self.buffer = shared_memory.SharedMemory(size=size * max_bytes)
        
        # Ring buffer 结构
        self.header = HeaderStruct(
            write_ptr=0,
            read_ptr=0,
            count=0,
        )
        
        # 同步原语
        self.cond = SpinCondition()
    
    def enqueue(self, item):
        """入队（零拷贝）"""
        # 1. 序列化到共享内存
        buffer = self.encoder.encode_into(item, self.get_write_buffer())
        
        # 2. 更新 write_ptr
        self.header.write_ptr = (self.header.write_ptr + 1) % self.size
        self.header.count += 1
        
        # 3. 通知消费者
        self.cond.notify()
    
    def dequeue(self, timeout=None):
        """出队（零拷贝）"""
        # 1. 等待数据
        self.cond.wait(timeout)
        
        # 2. 从共享内存读取
        buffer = self.get_read_buffer()
        item = self.decoder.decode(buffer)
        
        # 3. 更新 read_ptr
        self.header.read_ptr = (self.header.read_ptr + 1) % self.size
        self.header.count -= 1
        
        return item
```

**关键特性**：
- **零拷贝**: 直接在共享内存中序列化/反序列化
- **Ring Buffer**: 循环使用，避免频繁分配
- **SpinCondition**: 自旋等待，平衡延迟和 CPU 使用
- **多生产者多消费者**: 支持并发访问

---

### 8.3 异步 I/O 重叠

```python
# 多线程处理 ZMQ I/O

class EngineCoreProc:
    def __init__(self, ...):
        # 输入线程：处理请求
        input_thread = threading.Thread(
            target=self.process_input_socket,
            daemon=True,
        )
        input_thread.start()
        
        # 输出线程：发送响应
        output_thread = threading.Thread(
            target=self.process_output_socket,
            daemon=True,
        )
        output_thread.start()
        
        # 主线程：执行调度和推理
        # 这样可以实现 I/O 和计算的并行
```

**性能优势**：
- I/O 和计算并行执行
- 减少 GIL 影响（ZMQ 在 C++ 层释放 GIL）
- 提高吞吐量

---

## 补充九、总结

### 9.1 进程模型对比

| 场景 | 进程模型 | 通信方式 | 适用场景 |
|------|----------|----------|----------|
| 单GPU | 单进程 | 无 | 简单推理、调试 |
| 多GPU TP | 多进程 | ZMQ + MessageQueue + NCCL | 单机多卡 |
| 多GPU DP | 多进程 | ZMQ + XSUB/XPUB | 数据并行、负载均衡 |
| 流水线 PP | 多进程 | ZMQ + MessageQueue | 大模型分层 |
| 分布式 | Ray Actors | Ray + NCCL/TCP | 多节点部署 |

### 9.2 关键设计特点

1. **分层解耦**：API Server → EngineCore → Workers 清晰分层
2. **灵活通信**：ZMQ 支持多种模式，MessageQueue 提供高性能共享内存
3. **容错机制**：进程监控、自动重启、优雅关闭
4. **性能优化**：零拷贝、异步 I/O、流水线并行

### 9.3 性能优化策略

| 优化技术 | 适用场景 | 性能提升 |
|---------|----------|----------|
| 零拷贝序列化 | 大消息传输 | 2-5x |
| 共享内存 MQ | Worker 通信 | 5-10x |
| 异步 I/O 线程 | 高并发场景 | 2-3x |
| 批处理 | 高吞吐场景 | 3-5x |
| ZMQ IPC | 本地进程间 | 2-3x |

---

## 附录

### A. 参考文档

- [ZeroMQ 官方文档](https://zeromq.org/documentation/)
- [msgspec 文档](https://jcristharif.com/msgspec/)
- [vLLM 源码](https://github.com/vllm-project/vllm)

### B. 相关源码文件

- `vllm/v1/engine/core_client.py` - EngineCore 客户端
- `vllm/v1/engine/core.py` - EngineCore 进程
- `vllm/v1/engine/utils.py` - 进程管理工具
- `vllm/v1/executor/multiproc_executor.py` - 多进程执行器
- `vllm/v1/worker/worker_base.py` - Worker 基类
- `vllm/distributed/device_communicators/shm_broadcast.py` - 共享内存消息队列

---

> 补充章节版本：v1.2  
> 更新日期：2026-06-12  
> 作者：Claude (glm-5.1)

---

# 补充章节：KVCache 功能设计详细介绍

> 本章节深入分析 vLLM 的 KVCache 功能设计，包括核心实现、数据结构、管理机制和通信优化
> 
> 更新日期：2026-06-12

---

## 补充十、KVCache 核心实现

### 10.1 核心接口定义

**KVCacheSpec 体系**：vLLM 定义了完整的 KVCache 规格体系，用于描述不同类型的 KV 缓存：

```python
# vllm/v1/kv_cache_interface.py

class KVCacheSpec:
    """基础规格类"""
    block_size: int  # 每个 block 的 token 数量
    
    def page_size_bytes(self) -> int:
        """计算页面大小（字节）"""
        pass


class AttentionSpec(KVCacheSpec):
    """注意力层规格"""
    num_kv_heads: int      # KV heads 数量
    head_size: int         # 每个 head 的维度
    dtype: torch.dtype     # 数据类型
    kv_quant_mode: KVQuantMode  # 量化模式


class FullAttentionSpec(AttentionSpec):
    """全注意力规格"""
    sliding_window: int | None  # 滑动窗口大小
    attention_chunk_size: int | None  # attention chunk 大小


class MLAAttentionSpec(AttentionSpec):
    """MLA（Multi-Head Latent Attention）规格"""
    # 支持压缩和特殊内存布局
    

class SlidingWindowSpec(AttentionSpec):
    """滑动窗口注意力规格"""
    sliding_window: int  # 固定的滑动窗口大小


class MambaSpec(KVCacheSpec):
    """Mamba 状态缓存规格"""
    cache_mode: MambaCacheMode  # "all" / "align" / "none"
```

**KVQuantMode 枚举**：

```python
class KVQuantMode(enum.Enum):
    NONE = 0                # 无量化
    FP8_PER_TENSOR = 1      # FP8 per-tensor 量化
    INT8_PER_TOKEN_HEAD = 2 # INT8 per-token-head 动态量化
    FP8_PER_TOKEN_HEAD = 3  # FP8 per-token-head 动态量化
    NVFP4 = 4               # NVFP4 打包量化
```

---

### 10.2 KVCacheConfig 配置

```python
# vllm/config/cache.py

@config
class CacheConfig:
    """KV Cache 配置"""
    
    # 基础配置
    block_size: int = 16  # 每个 block 的 token 数量
    gpu_memory_utilization: float = 0.92  # GPU 内存利用率
    cache_dtype: CacheDType = "auto"  # KV cache 数据类型
    
    # 前缀缓存
    enable_prefix_caching: bool = True  # 启用前缀缓存
    prefix_caching_hash_algo: Literal[
        "sha256", "sha256_cbor", "xxhash", "xxhash_cbor"
    ] = "sha256"
    
    # KV Offloading
    kv_offloading_size: float | None = None  # KV offloading 缓冲区大小（GiB）
    kv_offloading_backend: Literal["native", "lmcache"] = "native"
    
    # 量化配置
    calculate_kv_scales: bool = False  # 动态计算 KV scales
    kv_cache_dtype_skip_layers: list[str] = []  # 跳过量化的层
    
    # Mamba 配置
    mamba_cache_dtype: MambaDType = "auto"
    mamba_cache_mode: MambaCacheMode = "none"
    
    # 运行时状态（由引擎设置）
    num_gpu_blocks: int | None = None  # GPU 块数量
    num_cpu_blocks: int | None = None  # CPU 块数量
```

---

### 10.3 KVCache 管理器

**KVCacheManager** 是核心管理类，提供完整的 KV cache 生命周期管理：

```python
# vllm/v1/core/kv_cache_manager.py

class KVCacheManager:
    """KV Cache 统一管理器"""
    
    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        kv_cache_specs: list[KVCacheSpec],
        ...
    ):
        # 创建块池
        self.block_pool = BlockPool(kv_cache_config.num_blocks)
        
        # 创建协调器（支持混合模型）
        self.coordinator = KVCacheCoordinator(kv_cache_specs)
        
        # 前缀缓存哈希映射
        self.cached_block_hash_to_block: dict[
            BlockHashWithGroupId, KVCacheBlock
        ] = {}
    
    # ========== 核心方法 ==========
    
    def get_computed_blocks(
        self,
        block_hashes: list[BlockHash],
        ...
    ) -> tuple[list[KVCacheBlock], ...]:
        """获取前缀缓存命中的块"""
        # 1. 查找最长前缀匹配
        # 2. 返回已计算的块序列
        pass
    
    def allocate_slots(
        self,
        request_id: str,
        num_blocks: int,
        ...
    ) -> KVCacheBlocks:
        """为请求分配新的 KV cache 槽位"""
        # 1. 从块池获取空闲块
        # 2. 设置引用计数
        # 3. 返回分配的块序列
        pass
    
    def free(self, request_id: str):
        """释放请求占用的块"""
        # 1. 减少引用计数
        # 2. 将空闲块加入队列
        # 3. 清理缓存映射（如果需要）
        pass
    
    def cache_blocks(
        self,
        request_id: str,
        block_hashes: list[BlockHash],
        ...
    ):
        """缓存已计算的块"""
        # 1. 计算块哈希
        # 2. 添加到缓存映射
        # 3. 更新 LRU 状态
        pass
```

**内存分配布局**：

```
----------------------------------------------------------------------
| < comp > | < new_comp > | < ext_comp >  | < new >  | < lookahead > |
----------------------------------------------------------------------
   已计算      新缓存命中    外部计算块     新分配     推测解码预分配

说明：
- comp: 已计算并缓存的块（前缀缓存命中）
- new_comp: 本次新发现的前缀缓存命中
- ext_comp: 通过 KV connector 外部计算的块
- new: 本次需要新计算的块
- lookahead: 推测解码的预分配块（EAGLE/MTP）
```

---

## 补充十一、KVCache 数据结构

### 11.1 Block 表设计

**KVCacheBlock 数据结构**：

```python
# vllm/v1/core/kv_cache_utils.py

@dataclass(slots=True)
class KVCacheBlock:
    """KV Cache 块"""
    
    block_id: int  # 块 ID（0 到 num_gpu_blocks-1）
    ref_cnt: int = 0  # 引用计数（支持多请求共享）
    _block_hash: BlockHashWithGroupId | None = None  # 块哈希（用于前缀缓存）
    
    # 双向链表指针（用于空闲队列）
    prev_free_block: KVCacheBlock | None = None
    next_free_block: KVCacheBlock | None = None
    
    is_null: bool = False  # 是否为空块（永不缓存）
```

**关键特性**：
- **引用计数**：支持多个请求共享相同的块
- **块哈希**：用于前缀缓存查找
- **双向链表**：O(1) 时间复杂度的块分配和释放
- **空块标记**：某些特殊场景（如 sliding window）需要空块

---

### 11.2 BlockPool 内存管理

```python
# vllm/v1/core/block_pool.py

class BlockPool:
    """块池管理器"""
    
    def __init__(self, num_blocks: int):
        # 创建所有块
        self.blocks: list[KVCacheBlock] = [
            KVCacheBlock(block_id=i) for i in range(num_blocks)
        ]
        
        # 空闲块队列（双向链表）
        self.free_block_queue = FreeKVCacheBlockQueue(self.blocks)
        
        # 缓存块哈希映射
        self.cached_block_hash_to_block: dict[
            BlockHashWithGroupId, KVCacheBlock
        ] = {}
    
    def get_new_blocks(self, num_blocks: int) -> list[KVCacheBlock]:
        """分配新的空闲块"""
        # 1. 检查是否有足够的空闲块
        if num_blocks > self.free_block_queue.num_free_blocks:
            raise ValueError("Cannot get free blocks")
        
        # 2. 从空闲队列弹出块
        blocks = self.free_block_queue.popleft_n(num_blocks)
        
        # 3. 设置引用计数
        for block in blocks:
            block.ref_cnt += 1
        
        return blocks
    
    def free_blocks(self, ordered_blocks: Iterable[KVCacheBlock]):
        """释放块"""
        blocks_list = list(ordered_blocks)
        
        # 1. 减少引用计数
        for block in blocks_list:
            block.ref_cnt -= 1
        
        # 2. 将引用计数为 0 的块加入空闲队列
        free_blocks = [
            block for block in blocks_list 
            if block.ref_cnt == 0 and not block.is_null
        ]
        
        # 反转顺序：确保尾部块优先淘汰（LRU）
        free_blocks.reverse()
        self.free_block_queue.append_n(free_blocks)
```

**FreeKVCacheBlockQueue**：

```python
class FreeKVCacheBlockQueue:
    """空闲块队列（双向链表）"""
    
    def __init__(self, blocks: list[KVCacheBlock]):
        # 所有块初始化为空闲
        self.num_free_blocks = len(blocks)
        
        # 构建双向链表
        for i in range(len(blocks)):
            if i > 0:
                blocks[i].prev_free_block = blocks[i - 1]
            if i < len(blocks) - 1:
                blocks[i].next_free_block = blocks[i + 1]
        
        self.head = blocks[0] if blocks else None
        self.tail = blocks[-1] if blocks else None
    
    def popleft_n(self, n: int) -> list[KVCacheBlock]:
        """从队列前端弹出 n 个块"""
        # O(n) 时间复杂度
        # 弹出的块是最少使用的（LRU）
        pass
    
    def append_n(self, blocks: list[KVCacheBlock]):
        """将块加入队列尾部"""
        # 加入尾部的块是最近使用的
        pass
```

---

### 11.3 PagedAttention 实现

**核心思想**：
- 将 KV cache 分割成固定大小的块（block_size = 16）
- 每个块可以独立分配、释放和共享
- 支持非连续内存访问

**内存布局示例**：

```
Request 1 (prompt length = 35):
Logical Blocks: [Block_0, Block_1, Block_2]
              ↓      ↓      ↓
Physical Blocks: [P5, P10, P23]  # 非连续分配

Request 2 (prompt length = 20, 共享前缀):
Logical Blocks: [Block_0, Block_1]  # 前两个块与 Request 1 共享
              ↓      ↓
Physical Blocks: [P5, P10]  # 引用计数 +1

Block Table Mapping:
Request 1: [5, 10, 23]
Request 2: [5, 10]  (ref_cnt: P5=2, P10=2)
```

**FlashInfer Backend 实现**：

```python
# vllm/v1/attention/backends/flashinfer.py

class FlashInferAttentionBackend:
    """FlashInfer 注意力后端"""
    
    def begin_forward(
        self,
        block_tables: torch.Tensor,  # 块表映射
        query_lens: torch.Tensor,     # 查询长度
        ...
    ):
        # 构建分页索引
        paged_kv_indptr = torch.tensor([
            0, num_blocks_req1, num_blocks_req1+num_blocks_req2, ...
        ])
        
        paged_kv_indices = torch.tensor([
            block_id_0, block_id_1, ...  # 所有物理块 ID
        ])
        
        paged_kv_last_page_len = torch.tensor([
            last_page_len_req1, last_page_len_req2, ...
        ])
        
        # 调用 FlashInfer 分页注意力
        wrapper.begin_forward(
            paged_kv_indptr,
            paged_kv_indices,
            paged_kv_last_page_len,
        )
```

**关键参数说明**：

| 参数 | 含义 | 示例 |
|------|------|------|
| `paged_kv_indptr` | 请求的块索引指针 | `[0, 3, 5]` 表示 Request 0 有 3 块，Request 1 有 2 块 |
| `paged_kv_indices` | 物理块 ID 序列 | `[5, 10, 23, 5, 10]` |
| `paged_kv_last_page_len` | 最后一块的有效 token 数 | `[16, 4]` 表示最后一块有 4 个有效 token |

---

### 11.4 Prefix Caching 实现

**BlockHash 机制**：

```python
# vllm/v1/core/kv_cache_utils.py

BlockHash = NewType("BlockHash", bytes)  # 32 bytes (SHA256)
BlockHashWithGroupId = NewType("BlockHashWithGroupId", bytes)

def hash_block_tokens(
    hash_function: Callable,
    parent_block_hash: BlockHash | None,
    curr_block_token_ids: Sequence[int],
    extra_keys: tuple[Any, ...] | None = None,
) -> BlockHash:
    """计算块的哈希值
    
    Args:
        hash_function: SHA256 或 xxHash
        parent_block_hash: 父块哈希（确保前缀依赖）
        curr_block_token_ids: 当前块的 token IDs（最多 block_size 个）
        extra_keys: 额外键（MM features, LoRA ID 等）
    
    Returns:
        BlockHash: 32 bytes 的哈希值
    """
    # 哈希包含：
    # 1. 父块哈希（保证前缀唯一性）
    # 2. 当前块 token IDs
    # 3. 额外键（多模态、LoRA 等）
```

**前缀缓存查找流程**：

```
1. 请求处理：
   Input: "Hello, how are you today?" (token_ids: [1, 2, 3, 4, 5, 6, 7, 8])
   block_size = 4
   
   计算块哈希：
   Block_0: hash(None, [1, 2, 3, 4]) → Hash_A
   Block_1: hash(Hash_A, [5, 6, 7, 8]) → Hash_B

2. 缓存查找：
   查找 Hash_A: 找到 → Physical Block P5
   查找 Hash_B: 未找到 → 需要新计算
   
   返回: [P5] (已计算), 需要 1 个新块

3. 块复用：
   - P5.ref_cnt += 1 (防止被淘汰)
   - 分配新块 P10 用于 Block_1
   - 计算 Block_1 的 KV cache
   - 缓存 Hash_B → P10

4. 后续请求：
   Input: "Hello, how are you?" (token_ids: [1, 2, 3, 4, 5, 6, 7])
   
   查找 Hash_A: 找到 → P5 (ref_cnt: 2)
   完全复用，无需计算！
```

**代码实现**：

```python
def find_longest_cache_hit(
    block_hashes: list[BlockHash],
    max_length: int,
    kv_cache_group_ids: list[int],
) -> tuple[list[KVCacheBlock], ...]:
    """查找最长前缀缓存命中
    
    Returns:
        tuple: 每个 KV cache 组的命中块序列
    """
    # 从左到右扫描块哈希
    for i, block_hash in enumerate(block_hashes[:max_length]):
        # 构建带组 ID 的哈希
        hash_with_group_id = BlockHashWithGroupId(
            block_hash + kv_cache_group_id.to_bytes(2, "little")
        )
        
        # 查找缓存映射
        if hash_with_group_id not in cached_block_hash_to_block:
            # 未找到，返回已找到的块
            return blocks[:i]
        
        # 找到缓存块
        cached_block = cached_block_hash_to_block[hash_with_group_id]
        
        # 检查引用计数（避免淘汰）
        if cached_block.ref_cnt == 0:
            # 块即将被淘汰，不能使用
            return blocks[:i]
        
        # 增加引用计数
        cached_block.ref_cnt += 1
        blocks.append(cached_block)
    
    return tuple(blocks)
```

---

### 11.5 Sliding Window 支持

**SlidingWindowManager** 实现：

```python
# vllm/v1/core/kv_cache_utils.py

class SlidingWindowManager:
    """滑动窗口管理器"""
    
    def __init__(self, sliding_window: int):
        self.sliding_window = sliding_window
    
    def get_num_skipped_tokens(
        self, 
        num_computed_tokens: int
    ) -> int:
        """计算需要跳过的 token 数
        
        示例：
        sliding_window = 4
        num_computed_tokens = 7
        
        Tokens:   [ 0  1  2  3  4  5  6  7 ]
                  | ---- computed -----|
                                        ^ next token
                    |--skipped---|
                    |--- window -|
        
        skipped_tokens = max(0, 7 - 4 + 1) = 4
        """
        return max(0, num_computed_tokens - self.sliding_window + 1)
    
    def get_num_skipped_blocks(
        self,
        num_computed_tokens: int,
        block_size: int,
    ) -> int:
        """计算需要跳过的块数"""
        skipped_tokens = self.get_num_skipped_tokens(num_computed_tokens)
        return skipped_tokens // block_size
```

**块管理示例**：

```
Request with sliding_window=4, block_size=2:

Token positions:  [0 1] [2 3] [4 5] [6 7] [8 9]
                   B0    B1    B2    B3    B4

num_computed_tokens = 8, next token = 9

skipped_blocks = (8 - 4 + 1) // 2 = 2

Effective blocks:
[B0(skip), B1(skip), B2(valid), B3(valid), B4(new)]

实际内存：
- B0, B1: 使用 null_block（不占用内存）
- B2, B3: 使用物理块（复用已计算的 KV cache）
- B4: 新分配的块

内存节省：
- 只保留窗口内的 4 个 token 的 KV cache
- 窗口外的块被替换为 null_block
```

**ChunkedLocalAttention**：

```python
class ChunkedLocalAttentionSpec(AttentionSpec):
    """分块局部注意力规格
    
    用于 LLaMA4 等模型的局部注意力
    - 窗口边界对齐到 chunk_size
    - 支持 chunk 内的全注意力
    """
    
    chunk_size: int  # 分块大小（如 1024）
```

---

## 补充十二、KVCache 管理机制

### 12.1 Block 分配和回收

**完整流程图**：

```
请求处理流程：

┌─────────────┐
│ Request Arr │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│ 1. 计算块哈希   │
│ (token_ids →   │
│  BlockHash)    │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│ 2. 前缀缓存查找 │
│ find_longest_   │
│ cache_hit()     │
└──────┬──────────┘
       │
       ├─ Found ───▶ 增加引用计数
       │              返回缓存块
       │
       ▼ Not Found
┌─────────────────┐
│ 3. 分配新块     │
│ get_new_blocks()│
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│ 4. 计算 KV Cache│
│ (GPU forward)   │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│ 5. 缓存块哈希   │
│ cache_blocks()  │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│ 6. 请求完成     │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│ 7. 释放块       │
│ free()          │
│ ref_cnt -= 1    │
└─────────────────┘
```

**引用计数管理**：

```python
# 分配时
block.ref_cnt += 1  # 新请求使用

# 前缀缓存命中时
cached_block.ref_cnt += 1  # 多请求共享

# 释放时
block.ref_cnt -= 1
if block.ref_cnt == 0:
    # 加入空闲队列
    free_block_queue.append(block)
```

**淘汰策略**：

```
LRU（Least Recently Used）：

空闲队列顺序：
[Head] ← Block_A (oldest) ← Block_B ← Block_C ← Block_D (newest) [Tail]

分配顺序：从 Head 弹出（淘汰最少使用的）
新空闲块加入：从 Tail 加入（保留最近使用的）

示例：
- 分配：弹出 Block_A（最少使用）
- 释放：Block_X 加入 Tail（最近释放）
- 结果：Block_X 不容易被淘汰
```

---

### 12.2 内存容量规划

**自动适配流程**：

```python
# vllm/v1/core/kv_cache_interface.py

def _auto_fit_max_model_len(
    vllm_config: VllmConfig,
    available_gpu_memory: int,
) -> int:
    """二分搜索找到适合内存的最大模型长度
    
    流程：
    1. 从模型配置的最大长度开始
    2. 计算需要的 KV cache 块数
    3. 如果超出可用内存，降低长度
    4. 重复直到找到合适的长度
    """
    max_len = vllm_config.model_config.max_model_len
    
    while max_len > 0:
        # 计算 KV cache 需要的内存
        num_blocks = calculate_num_blocks(max_len)
        memory_needed = num_blocks * block_size_bytes
        
        if memory_needed <= available_gpu_memory:
            return max_len
        
        # 降低长度
        max_len = max_len // 2
    
    return 0  # 无法适配
```

**内存计算公式**：

```python
def calculate_kv_cache_memory(
    num_blocks: int,
    kv_cache_spec: KVCacheSpec,
) -> int:
    """计算 KV cache 需要的内存
    
    公式：
    memory = num_blocks × block_size × num_layers × num_kv_heads × head_size × dtype_size
    
    示例（LLaMA-7B）：
    - num_blocks = 10000
    - block_size = 16
    - num_layers = 32
    - num_kv_heads = 32
    - head_size = 128
    - dtype = fp16 (2 bytes)
    
    memory = 10000 × 16 × 32 × 32 × 128 × 2
           = 10000 × 16 × 262144
           = 4.19 GB
    """
    page_size = kv_cache_spec.page_size_bytes()
    return num_blocks * page_size * num_layers
```

---

### 12.3 Hybrid 模型内存布局

**混合 KV cache 组**：

```python
# 示例：模型包含多种 attention 类型

Layers:
- Layer 0-10: Full Attention
- Layer 11-15: Sliding Window (window=1024)
- Layer 16-20: Mamba

KV Cache Groups:
- Group 0: Full Attention (Layer 0-10)
- Group 1: Sliding Window (Layer 11-15)
- Group 2: Mamba (Layer 16-20)

内存池分配：
- Tensor 0: Group 0 (Full Attention)
- Tensor 1: Group 1 (Sliding Window)
- Tensor 2: Group 2 (Mamba)

共享策略：
- 如果多个组有相同的 page_size，可以共享张量
- 通过 coordinator 协调块的分配和释放
```

**Coordinator 实现**：

```python
class KVCacheCoordinator:
    """KV Cache 组协调器"""
    
    def __init__(self, kv_cache_specs: list[KVCacheSpec]):
        # 创建每个组的管理器
        self.managers = [
            SingleTypeKVCacheManager(spec) for spec in kv_cache_specs
        ]
        
        # 统一页面大小
        self.unified_page_size = self._compute_unified_page_size()
    
    def find_longest_prefix_hit(
        self,
        block_hashes: list[BlockHash],
    ) -> tuple[list[KVCacheBlock], ...]:
        """找到所有组的公共最长前缀
        
        使用迭代定点算法：
        1. 初始估计前缀长度
        2. 检查每个组是否支持该长度
        3. 更新估计，重复直到收敛
        """
        pass
```

---

## 补充十三、KVCache 通信与传输

### 13.1 KV Transfer 机制

**KVCacheEvent 事件系统**：

```python
# vllm/distributed/kv_events.py

@dataclass
class BlockStored(KVCacheEvent):
    """块存储事件"""
    
    block_hashes: list[ExternalBlockHash]  # 块哈希
    parent_block_hash: ExternalBlockHash | None  # 父块哈希
    token_ids: list[int]  # Token IDs
    block_size: int  # 块大小
    lora_id: int | None  # LoRA ID
    medium: str | None  # 存储介质
    extra_keys: list[tuple[Any, ...] | None] | None  # 额外键
    group_idx: int | None  # KV cache 组索引
    kv_cache_spec_kind: str | None  # Spec 类型
    kv_cache_spec_sliding_window: int | None  # 滑动窗口大小
```

**事件发布架构**：

```
┌─────────────────┐
│  Scheduler      │
│                 │
│  KVTransferState│
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│ KVEventPublisher│
│  (ZMQ PUB)      │
└──────┬──────────┘
       │
       │ ZMQ PUB-SUB
       ▼
┌─────────────────┐
│ KVEventAggregator│
│  (聚合事件)      │
└──────┬──────────┘
       │
       │ 聚合后的事件
       ▼
┌─────────────────┐
│  KV Connector   │
│  (传输实现)      │
└─────────────────┘
```

---

### 13.2 KV Connector 架构

**支持的 Connector**：

| Connector | 说明 | 适用场景 |
|-----------|------|----------|
| LMCacheConnector | LMCache 集成 | 分布式缓存 |
| NixlConnector | NIXL 高性能传输 | RDMA 传输 |
| MooncakeConnector | Mooncake 存储 | 分布式存储 |
| FlexKVConnector | FlexKV 系统 | 企业级 KV cache |
| SimpleCPUOffloadConnector | 简单 CPU offload | 本地 CPU 内存 |

**Connector 接口**：

```python
# vllm/distributed/kv_transfer/kv_connector/v1/base.py

class KVConnectorBase(ABC):
    """KV Connector 基类"""
    
    @abstractmethod
    def lookup(
        self,
        block_hashes: list[BlockHash],
        req_context: dict,
    ) -> bool | None:
        """检查块是否已在 connector 中"""
        pass
    
    @abstractmethod
    def load(
        self,
        block_hashes: list[BlockHash],
        req_context: dict,
    ) -> LoadSpec:
        """从 connector 加载块"""
        pass
    
    @abstractmethod
    def store(
        self,
        block_hashes: list[BlockHash],
        kv_cache_data: torch.Tensor,
        req_context: dict,
    ) -> StoreSpec:
        """将块存储到 connector"""
        pass
```

---

### 13.3 KV Offloading 实现

**OffloadingManager 接口**：

```python
# vllm/v1/kv_offload/offloading_manager.py

class OffloadingManager(ABC):
    """Offloading 管理器基类"""
    
    @abstractmethod
    def lookup(
        self,
        key: OffloadKey,
        req_context: dict,
    ) -> bool | None:
        """检查块是否已被 offload"""
        pass
    
    @abstractmethod
    def prepare_load(
        self,
        keys: list[OffloadKey],
        req_context: dict,
    ) -> LoadStoreSpec:
        """准备加载 offloaded 块"""
        pass
    
    @abstractmethod
    def prepare_store(
        self,
        keys: list[OffloadKey],
        req_context: dict,
    ) -> PrepareStoreOutput:
        """准备 offload 块"""
        pass
```

**Offloading 流程**：

```
1. Scheduler 端：
   ┌─────────────────┐
   │ lookup()        │ ← 检查块是否已 offload
   │ prepare_load()  │ ← 准备加载操作
   │ prepare_store() │ ← 准备存储操作
   └─────────────────┘

2. Worker 端：
   ┌─────────────────┐
   │ OffloadHandler  │
   │  - load()       │ ← 从 CPU 加载到 GPU
   │  - store()      │ ← 从 GPU 存储到 CPU
   └─────────────────┘

3. 传输：
   ┌─────────────────┐
   │ Transfer Engine │
   │  - cudaMemcpy   │ ← GPU-CPU 传输
   │  - RDMA         │ ← 高性能传输
   │  - Async        │ ← 异步传输
   └─────────────────┘
```

**CPU Offloading 配置**：

```python
# CacheConfig
kv_offloading_size: float = 4.0  # 4 GiB CPU 内存用于 KV offloading
kv_offloading_backend: str = "native"  # 使用 vLLM 原生 backend

# 工作原理：
# 1. GPU KV cache 块满时，将部分块 offload 到 CPU
# 2. 需要使用时，从 CPU 加载回 GPU
# 3. 支持异步传输，避免阻塞推理
```

---

### 13.4 跨节点 KV 共享

**P/D（Prefill/Decode）分离场景**：

```
┌──────────────────────────────────────────────┐
│              Prefill Node                     │
│  ┌────────────────────────────────────────┐  │
│  │  Request: "Hello, how are you?"        │  │
│  │  - Compute full KV cache               │  │
│  │  - Publish BlockStored events          │  │
│  └────────────────────────────────────────┘  │
└────────────────────┬─────────────────────────┘
                     │
                     │ KV Transfer (RDMA/TCP)
                     │ - Block hashes
                     │ - Token IDs
                     │ - KV cache tensors
                     ▼
┌──────────────────────────────────────────────┐
│              Decode Node                      │
│  ┌────────────────────────────────────────┐  │
│  │  Request: "Hello, how are you?"        │  │
│  │  - Receive KV cache                    │  │
│  │  - Continue decoding                   │  │
│  │  - No prefill computation needed!      │  │
│  └────────────────────────────────────────┘  │
└──────────────────────────────────────────────┘

优势：
- Prefill 专用节点：大批量 prompt 处理
- Decode 专用节点：专注于生成 token
- KV cache 共享：避免重复计算
- 吞吐量提升：2-3x
```

---

## 补充十四、设计原理总结

### 14.1 核心设计原则

| 设计原则 | 说明 | 实现方式 |
|---------|------|----------|
| **分页管理** | 类似操作系统的虚拟内存 | Block table mapping |
| **引用计数** | 多请求共享前缀块 | KVCacheBlock.ref_cnt |
| **哈希索引** | 快速前缀缓存查找 | SHA256/xxHash |
| **分层抽象** | 灵活支持多种模型 | KVCacheSpec体系 |

---

### 14.2 优化技术总结

| 优化技术 | 适用场景 | 性能提升 |
|---------|----------|----------|
| **Prefix Caching** | 重复前缀（文档、对话） | 90%+ 时间节省 |
| **Sliding Window** | 长序列推理 | 内存节省 50-70% |
| **KV Quantization** | 内存受限场景 | 内存节省 50% |
| **KV Offloading** | GPU 内存不足 | 支持更大 batch |
| **P/D Separation** | 高吞吐场景 | 吞吐量提升 2-3x |

---

### 14.3 适用场景

1. **长文档查询**：前缀缓存大幅减少重复计算
2. **多轮对话**：历史对话复用，提升响应速度
3. **多模态模型**：高效管理编码器状态
4. **分布式推理**：P/D 分离、跨节点 KV 共享
5. **内存受限场景**：KV offloading、量化、滑动窗口

---

## 附录：关键源码文件

**核心实现**：
- `vllm/v1/kv_cache_interface.py` - KV Cache 接口定义
- `vllm/v1/core/kv_cache_manager.py` - KV Cache 管理器
- `vllm/config/cache.py` - CacheConfig 配置

**数据结构**：
- `vllm/v1/core/kv_cache_utils.py` - Block 数据结构
- `vllm/v1/core/block_pool.py` - BlockPool 内存管理
- `vllm/v1/core/kv_cache_coordinator.py` - Coordinator 协调器
- `vllm/v1/core/single_type_kv_cache_manager.py` - 单类型管理器

**通信和传输**：
- `vllm/distributed/kv_events.py` - KV 事件系统
- `vllm/distributed/kv_transfer/` - KV Connector 实现
- `vllm/v1/kv_offload/` - KV Offloading 实现

**Attention Backend**：
- `vllm/v1/attention/backends/flashinfer.py` - FlashInfer 实现
- `vllm/v1/attention/backends/flash_attn.py` - FlashAttention 实现

---

> 补充章节版本：v1.3  
> 更新日期：2026-06-12  
> 作者：Claude (glm-5.1)

---

# 补充章节：模型架构解析机制详解

> 本章节详细分析 vLLM 的模型架构解析机制，解释如何从 HF 配置映射到 vLLM 实现类
> 
> 更新日期：2026-06-12

---

## 补充十五、get_model_architecture 处理流程

### 15.1 函数概览

`get_model_architecture` 是 vLLM 模型加载的核心入口，负责将 HuggingFace 模型的 `architectures` 配置映射到 vLLM 的模型实现类。

**位置**：`vllm/model_executor/model_loader/utils.py`

```python
def get_model_architecture(model_config: ModelConfig) -> tuple[type[nn.Module], str]:
    """获取模型架构类
    
    Args:
        model_config: 模型配置对象
        
    Returns:
        tuple: (模型类, 架构名称)
    """
    # 1. 计算缓存 key
    key = hash(
        (
            model_config.model,
            model_config.convert_type,
            model_config.runner_type,
            model_config.trust_remote_code,
            model_config.model_impl,
            tuple(getattr(model_config.hf_config, "architectures", None) or []),
        )
    )
    
    # 2. 检查缓存
    if key in _MODEL_ARCH_BY_HASH:
        return _MODEL_ARCH_BY_HASH[key]
    
    # 3. 解析模型架构
    model_arch = _get_model_architecture(model_config)
    
    # 4. 缓存结果
    _MODEL_ARCH_BY_HASH[key] = model_arch
    return model_arch
```

---

### 15.2 完整处理流程图

```
┌─────────────────────────────────────────────────────────────────────┐
│                    get_model_architecture 流程                       │
└─────────────────────────────────────────────────────────────────────┘

输入: ModelConfig
    │
    │ 包含:
    │ - model: "glm-5.1/Qwen2-7B"
    │ - hf_config.architectures: ["Qwen2ForCausalLM"]
    │ - model_impl: "auto"
    │ - convert_type: "none"
    │ - runner_type: "generate"
    │
    ▼
┌──────────────────────────────────────────────────────────────────┐
│ Step 1: 计算缓存 Key                                              │
│                                                                   │
│ key = hash(                                                       │
│     model,                                                        │
│     convert_type,                                                 │
│     runner_type,                                                  │
│     trust_remote_code,                                            │
│     model_impl,                                                   │
│     tuple(architectures),                                         │
│ )                                                                 │
└──────────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────────┐
│ Step 2: 检查缓存                                                  │
│                                                                   │
│ if key in _MODEL_ARCH_BY_HASH:                                    │
│     return cached result  ← 快速返回                              │
└──────────────────────────────────────────────────────────────────┘
    │ (缓存未命中)
    ▼
┌──────────────────────────────────────────────────────────────────┐
│ Step 3: 调用 _get_model_architecture                              │
│                                                                   │
│ 内部流程:                                                         │
│ ┌────────────────────────────────────────────────────────────┐   │
│ │ 3.1 获取 architectures                                    │   │
│ │     architectures = hf_config.architectures                │   │
│ │     例如: ["Qwen2ForCausalLM"]                              │   │
│ └────────────────────────────────────────────────────────────┘   │
│     │                                                              │
│     ▼                                                              │
│ ┌────────────────────────────────────────────────────────────┐   │
│ │ 3.2 调用 registry.resolve_model_cls()                     │   │
│ │                                                            │   │
│ │ 核心解析逻辑:                                              │   │
│ │ - 尝试 vLLM 注册表                                         │   │
│ │ - 尝试 Transformers backend                                │   │
│ │ - 处理 convert_type                                        │   │
│ └────────────────────────────────────────────────────────────┘   │
│     │                                                              │
│     ▼                                                              │
│ ┌────────────────────────────────────────────────────────────┐   │
│ │ 3.3 处理 convert_type                                     │   │
│ │                                                            │   │
│ │ if convert_type == "embed":                                │   │
│ │     model_cls = as_embedding_model(model_cls)              │   │
│ │ elif convert_type == "classify":                           │   │
│ │     model_cls = as_seq_cls_model(model_cls)                │   │
│ └────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────────────┐
│ Step 4: 缓存结果                                                  │
│                                                                   │
│ _MODEL_ARCH_BY_HASH[key] = (model_cls, arch)                     │
└──────────────────────────────────────────────────────────────────┘
    │
    ▼
返回: (model_cls, arch)
    例如: (Qwen2ForCausalLM, "Qwen2ForCausalLM")
```

---

### 15.3 registry.resolve_model_cls() 核心解析流程

**位置**：`vllm/model_executor/models/registry.py`

```
┌─────────────────────────────────────────────────────────────────────┐
│              resolve_model_cls() 解析流程                            │
└─────────────────────────────────────────────────────────────────────┘

输入: architectures=["Qwen2ForCausalLM"], model_config
    │
    ▼
┌──────────────────────────────────────────────────────────────────┐
│ 判断 1: model_impl == "transformers"?                             │
│                                                                   │
│ Yes → 强制使用 Transformers backend                              │
│     │                                                             │
│     ▼                                                             │
│     _try_resolve_transformers(architectures[0], model_config)     │
│     返回: Transformers 实现类                                     │
└──────────────────────────────────────────────────────────────────┘
    │ (model_impl == "auto" 或 "vllm")
    ▼
┌──────────────────────────────────────────────────────────────────┐
│ 判断 2: model_impl == "terratorch"?                               │
│                                                                   │
│ Yes → 使用 Terratorch 实现                                        │
│     返回: (Terratorch, "Terratorch")                              │
└──────────────────────────────────────────────────────────────────┘
    │ (model_impl == "auto" 或 "vllm")
    ▼
┌──────────────────────────────────────────────────────────────────┐
│ 判断 3: 所有架构都不在注册表 + convert_type == "none"?             │
│                                                                   │
│ Yes → 回退到 Transformers backend (第一次尝试)                   │
│     │                                                             │
│     ▼                                                             │
│     _try_resolve_transformers(architectures[0], model_config)     │
│     如果成功，返回 Transformers 实现                              │
└──────────────────────────────────────────────────────────────────┘
    │ (仍未找到)
    ▼
┌──────────────────────────────────────────────────────────────────┐
│ 判断 4: 遍历 architectures                                        │
│                                                                   │
│ for arch in architectures:                                        │
│     # 4.1 规范化架构名                                            │
│     normalized_arch = _normalize_arch(arch, model_config)         │
│                                                                   │
│     # 4.2 尝试加载 vLLM 注册表                                    │
│     model_cls = _try_load_model_cls(normalized_arch)              │
│                                                                   │
│     if model_cls is not None:                                     │
│         return (model_cls, arch)  ← 成功返回                      │
└──────────────────────────────────────────────────────────────────┘
    │ (仍未找到)
    ▼
┌──────────────────────────────────────────────────────────────────┐
│ 判断 5: 所有架构都不在注册表 + model_impl == "auto"?               │
│                                                                   │
│ Yes → 回退到 Transformers backend (第二次尝试)                   │
│     │                                                             │
│     ▼                                                             │
│     _try_resolve_transformers(architectures[0], model_config)     │
│     如果成功，返回 Transformers 实现                              │
└──────────────────────────────────────────────────────────────────┘
    │ (仍未找到)
    ▼
┌──────────────────────────────────────────────────────────────────┐
│ 最终: 抛出异常                                                    │
│                                                                   │
│ _raise_for_unsupported(architectures)                             │
│                                                                   │
│ 错误信息:                                                         │
│ - 如果曾经支持但已弃用: 提示使用旧版本                            │
│ - 如果需要插件: 提示安装插件                                      │
│ - 如果完全不支持: 列出支持的架构                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

### 15.4 模型注册表结构

**注册表字典**：`vllm/model_executor/models/registry.py`

```python
# 文本生成模型注册表
_TEXT_GENERATION_MODELS = {
    # 格式: "HF架构名": ("vllm模块名", "vllm类名")
    
    "LlamaForCausalLM": ("llama", "LlamaForCausalLM"),
    "Qwen2ForCausalLM": ("qwen2", "Qwen2ForCausalLM"),
    "MistralForCausalLM": ("mistral", "MistralForCausalLM"),
    "MixtralForCausalLM": ("mixtral", "MixtralForCausalLM"),
    # ... 更多模型
}

# 嵌入模型注册表
_EMBEDDING_MODELS = {
    "BertModel": ("bert", "BertEmbeddingModel"),
    "Qwen2Model": ("qwen2", "Qwen2ForCausalLM"),
    # ... 更多模型
}

# 多模态模型注册表
_MULTIMODAL_MODELS = {
    "Qwen2VLForConditionalGeneration": (
        "qwen2_vl", "Qwen2VLForConditionalGeneration"
    ),
    "LlavaNextForConditionalGeneration": (
        "llava_next", "LlavaNextForConditionalGeneration"
    ),
    # ... 更多模型
}

# 全局注册表实例
ModelRegistry = _ModelRegistry(
    models={
        **_TEXT_GENERATION_MODELS,
        **_EMBEDDING_MODELS,
        **_MULTIMODAL_MODELS,
        # ... 其他类型
    }
)
```

**映射示例**：

```
HF Config architectures: ["Qwen2ForCausalLM"]
                    ↓
注册表查找: _TEXT_GENERATION_MODELS["Qwen2ForCausalLM"]
                    ↓
找到: ("qwen2", "Qwen2ForCausalLM")
                    ↓
动态导入: from vllm.model_executor.models.qwen2 import Qwen2ForCausalLM
                    ↓
返回: (Qwen2ForCausalLM 类, "Qwen2ForCausalLM")
```

---

### 15.5 _try_load_model_cls 加载机制

```python
@lru_cache(maxsize=128)
def _try_load_model_cls(
    model_arch: str,
    model: _BaseRegisteredModel,
) -> type[nn.Module] | None:
    """尝试加载模型类
    
    支持两种注册方式:
    1. 直接注册类: _RegisteredModel
    2. 懒加载注册: _LazyRegisteredModel
    """
    try:
        return model.load_model_cls()
    except Exception:
        logger.exception("Error in loading model architecture '%s'", model_arch)
        return None


# 懒加载模型注册
class _LazyRegisteredModel(_BaseRegisteredModel):
    """懒加载注册模型
    
    格式: "module:class"
    例如: "qwen2:Qwen2ForCausalLM"
    """
    
    def __init__(self, module_name: str, class_name: str):
        self.module_name = module_name
        self.class_name = class_name
    
    def load_model_cls(self) -> type[nn.Module]:
        # 动态导入模块
        module = importlib.import_module(
            f"vllm.model_executor.models.{self.module_name}"
        )
        
        # 获取类
        return getattr(module, self.class_name)


# 直接注册模型
class _RegisteredModel(_BaseRegisteredModel):
    """直接注册的模型类"""
    
    def __init__(self, model_cls: type[nn.Module]):
        self.model_cls = model_cls
    
    def load_model_cls(self) -> type[nn.Module]:
        return self.model_cls
```

---

### 15.6 Transformers Backend 回退机制

```python
def _try_resolve_transformers(
    self,
    architecture: str,
    model_config: ModelConfig,
) -> str | None:
    """尝试使用 Transformers backend
    
    适用场景:
    1. vLLM 没有实现该模型
    2. model_impl == "transformers"
    3. 自定义模型 (trust_remote_code=True)
    """
    
    # 1. 检查是否在允许列表
    if architecture in _TRANSFORMERS_BACKEND_MODELS:
        return architecture
    
    # 2. 处理 auto_map (自定义模型)
    auto_map = getattr(model_config.hf_config, "auto_map", None) or dict()
    
    # 3. 加载 auto_map 中的类
    for prefix in ("AutoConfig", "AutoModel"):
        for name, module in auto_map.items():
            if name.startswith(prefix):
                try_get_class_from_dynamic_module(
                    module,
                    model_config.model,
                    revision=model_config.revision,
                    trust_remote_code=model_config.trust_remote_code,
                )
    
    # 4. 尝试从 transformers 导入
    model_module = getattr(transformers, architecture, None)
    
    if model_module is None:
        # 尝试从 auto_map 加载
        for name, module in auto_map.items():
            if name.startswith("AutoModel"):
                model_module = try_get_class_from_dynamic_module(...)
                if model_module is not None:
                    break
    
    # 5. 检查兼容性
    if not model_module.is_backend_compatible():
        return None
    
    # 6. 返回 backend 类名
    return model_config._get_transformers_backend_cls()
```

**支持的 Transformers Backend 模型**：

```python
_TRANSFORMERS_BACKEND_MODELS = {
    # 允许使用 Transformers 实现的模型
    "Qwen2ForCausalLM",
    "MistralForCausalLM",
    # ... 更多
}
```

---

### 15.7 convert_type 处理机制

```python
def _get_model_architecture(model_config: ModelConfig) -> tuple[type[nn.Module], str]:
    """处理模型转换"""
    
    # 1. 解析基础模型类
    model_cls, arch = model_config.registry.resolve_model_cls(
        architectures,
        model_config=model_config,
    )
    
    # 2. 处理 convert_type
    convert_type = model_config.convert_type
    
    if convert_type == "none":
        # 不转换，直接返回
        pass
    
    elif convert_type == "embed":
        # 转换为嵌入模型
        logger.debug_once("Converting to embedding model.")
        model_cls = as_embedding_model(model_cls)
    
    elif convert_type == "classify":
        # 转换为序列分类模型
        logger.debug_once("Converting to sequence classification model.")
        model_cls = as_seq_cls_model(model_cls)
    
    else:
        assert_never(convert_type)
    
    return model_cls, arch
```

**转换适配器**：

```python
# vllm/model_executor/models/adapters.py

def as_embedding_model(model_cls: type[nn.Module]) -> type[nn.Module]:
    """将生成模型转换为嵌入模型
    
    添加嵌入相关的方法和属性
    """
    # 包装模型类
    # 添加 pooling 方法
    # 添加 embed 方法
    return EmbeddingModelWrapper(model_cls)


def as_seq_cls_model(model_cls: type[nn.Module]) -> type[nn.Module]:
    """将生成模型转换为序列分类模型
    
    添加分类相关的方法
    """
    return SequenceClassificationModelWrapper(model_cls)
```

---

### 15.8 实际案例分析

#### 案例 1: Qwen2-7B 模型

```
输入:
- model: "glm-5.1/Qwen2-7B"
- hf_config.architectures: ["Qwen2ForCausalLM"]
- model_impl: "auto"
- convert_type: "none"

处理流程:
1. 计算 key: hash(...)
2. 缓存未命中
3. 获取 architectures: ["Qwen2ForCausalLM"]
4. 调用 resolve_model_cls():
   - model_impl == "auto" → 继续
   - "Qwen2ForCausalLM" 在 _TEXT_GENERATION_MODELS 中
   - 找到: ("qwen2", "Qwen2ForCausalLM")
   - 动态导入: from vllm.model_executor.models.qwen2 import Qwen2ForCausalLM
   - 返回: (Qwen2ForCausalLM, "Qwen2ForCausalLM")
5. convert_type == "none" → 不转换
6. 缓存结果
7. 返回: (Qwen2ForCausalLM 类, "Qwen2ForCausalLM")
```

#### 案例 2: 自定义模型（trust_remote_code=True）

```
输入:
- model: "custom/my-model"
- hf_config.architectures: ["MyCustomModel"]
- hf_config.auto_map: {"AutoModel": "modeling_my_model--MyCustomModel"}
- model_impl: "auto"
- trust_remote_code: True

处理流程:
1. 计算 key
2. 缓存未命中
3. 获取 architectures: ["MyCustomModel"]
4. 调用 resolve_model_cls():
   - "MyCustomModel" 不在注册表中
   - 尝试 _try_resolve_transformers():
     - auto_map 存在
     - 加载 AutoConfig 和 AutoModel
     - model_module = try_get_class_from_dynamic_module(...)
     - 检查兼容性: is_backend_compatible()
     - 成功，返回 backend 类名
5. 返回 Transformers 实现类
```

#### 案例 3: 嵌入模型转换

```
输入:
- model: "glm-5.1/Qwen2-7B"
- architectures: ["Qwen2ForCausalLM"]
- convert_type: "embed"

处理流程:
1-4. 同案例 1，获得 Qwen2ForCausalLM 类
5. convert_type == "embed":
   - 调用 as_embedding_model(Qwen2ForCausalLM)
   - 包装为嵌入模型
6. 返回: (嵌入模型类, "Qwen2ForCausalLM")
```

---

### 15.9 缓存机制详解

```python
# 全局缓存字典
_MODEL_ARCH_BY_HASH = dict[int, tuple[type[nn.Module], str]]()
"""缓存 _get_model_architecture 的输出"""

# 缓存 key 计算因素
key_factors = (
    model_config.model,              # 模型路径
    model_config.convert_type,       # 转换类型
    model_config.runner_type,        # 运行器类型
    model_config.trust_remote_code,  # 是否信任远程代码
    model_config.model_impl,         # 实现选择
    tuple(architectures),            # HF 架构列表
)

# 缓存效果:
# - 避免重复解析相同模型
# - 减少 importlib.import_module 调用
# - 加速多请求场景的初始化
```

---

### 15.10 错误处理机制

```python
def _raise_for_unsupported(self, architectures: list[str]):
    """抛出不支持错误
    
    分三种情况:
    """
    
    # 1. 检查是否曾经支持但已弃用
    for arch in architectures:
        if arch in _PREVIOUSLY_SUPPORTED_MODELS:
            previous_version = _PREVIOUSLY_SUPPORTED_MODELS[arch]
            raise ValueError(
                f"Model architecture {arch} was supported in vLLM "
                f"until v{previous_version}, and is not supported anymore."
            )
    
    # 2. 检查是否需要插件
    for arch in architectures:
        if arch in _OOT_SUPPORTED_MODELS:
            plugin_url = _OOT_SUPPORTED_MODELS[arch]
            raise ValueError(
                f"Model architecture {arch} is not supported in-tree."
                f"Please install the plugin at {plugin_url}."
            )
    
    # 3. 完全不支持
    all_supported_archs = self.get_supported_archs()
    raise ValueError(
        f"Model architectures {architectures} are not supported."
        f"Supported architectures: {all_supported_archs}"
    )
```

**弃用模型示例**：

```python
_PREVIOUSLY_SUPPORTED_MODELS = {
    # 曾经支持但已移除的模型
    "MossForCausalLM": "0.4.0",
    "BaichuanForCausalLM": "0.5.0",
}
```

---

## 补充十六、initialize_model 模型初始化

### 16.1 初始化流程

```python
@instrument(span_name="Initialize model")
def initialize_model(
    vllm_config: VllmConfig,
    *,
    prefix: str = "",
    model_class: type[nn.Module] | None = None,
    model_config: ModelConfig | None = None,
) -> nn.Module:
    """初始化模型实例
    
    Args:
        vllm_config: vLLM 配置
        prefix: 模型前缀（用于 PP）
        model_class: 模型类（可选）
        model_config: 模型配置（可选）
    
    Returns:
        nn.Module: 初始化的模型实例
    """
    # 1. 获取配置
    if model_config is None:
        model_config = vllm_config.model_config
    
    # 2. 获取模型类
    if model_class is None:
        model_class, _ = get_model_architecture(model_config)
    
    # 3. 配置量化
    if vllm_config.quant_config is not None:
        configure_quant_config(vllm_config.quant_config, model_class)
    
    # 4. 检查签名（新式 vs 老式）
    signatures = inspect.signature(model_class.__init__)
    all_params = [param.name for param in signatures.parameters.values()]
    
    if "vllm_config" in all_params and "prefix" in all_params:
        # 新式模型类
        with set_current_vllm_config(vllm_config, check_compile=True, prefix=prefix):
            model = model_class(vllm_config=vllm_config, prefix=prefix)
            record_metadata_for_reloading(model)
            return model
    
    # 5. 老式模型类（兼容处理）
    warnings.warn("Old-style model class detected", DeprecationWarning)
    
    kwargs = {}
    if "prefix" in all_params:
        kwargs["prefix"] = prefix
    if "config" in all_params:
        kwargs["config"] = model_config.hf_config
    # ... 更多参数
    
    with set_current_vllm_config(vllm_config, check_compile=True, prefix=prefix):
        model = model_class(**kwargs)
    
    return model
```

---

### 16.2 新式模型类接口

```python
# 新式模型类必须接受的参数

class MyModel(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,  # 必需
        prefix: str = "",          # 必需
    ):
        super().__init__()
        
        # 从 vllm_config 中获取配置
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        # ...
```

**优势**：
- 统一的配置接口
- 支持 PP（通过 prefix）
- 自动配置管理
- 便于扩展

---

## 补充十七、设计原理总结

### 17.1 核心设计原则

| 设计原则 | 说明 | 实现方式 |
|---------|------|----------|
| **分层解析** | 多层次查找，确保兼容性 | registry → transformers → fallback |
| **缓存优化** | 避免重复解析 | `_MODEL_ARCH_BY_HASH` 全局缓存 |
| **灵活扩展** | 支持自定义模型 | `trust_remote_code` + `auto_map` |
| **兼容转换** | 模型类型转换 | `as_embedding_model` 等适配器 |

---

### 17.2 模型查找优先级

```
优先级顺序（model_impl == "auto"）:

1. vLLM 注册表（性能最优）
   ↓
2. Transformers Backend (convert_type == "none" 时)
   ↓
3. vLLM 注册表 + normalize
   ↓
4. Transformers Backend (最终 fallback)
   ↓
5. 抛出异常
```

---

### 17.3 关键配置参数影响

| 参数 | 影响 | 示例 |
|------|------|------|
| `model_impl` | 强制选择实现 | `"vllm"` / `"transformers"` |
| `convert_type` | 模型类型转换 | `"embed"` / `"classify"` |
| `trust_remote_code` | 是否加载自定义代码 | `True` / `False` |
| `architectures` | HF 模型架构 | `["Qwen2ForCausalLM"]` |

---

## 附录：关键源码文件

- `vllm/model_executor/model_loader/utils.py` - 模型加载工具
- `vllm/model_executor/models/registry.py` - 模型注册表
- `vllm/model_executor/models/__init__.py` - 模型导出
- `vllm/model_executor/models/adapters.py` - 模型适配器
- `vllm/transformers_utils/dynamic_module.py` - 动态模块加载

---

> 补充章节版本：v1.4  
> 更新日期：2026-06-12  
> 作者：Claude (glm-5.1)
