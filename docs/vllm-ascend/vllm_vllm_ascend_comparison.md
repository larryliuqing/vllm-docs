# vLLM 详细功能列表与 Ascend vs CUDA 实现对比分析

**文档版本**: v2.0  
**更新日期**: 2026-06-19  
**基于源码版本**: vLLM (latest), vLLM-Ascend (latest)

## 目录
1. [vLLM 完整功能列表](#1-vllm-完整功能列表)
2. [vLLM-Ascend 架构概览](#2-vllm-ascend-架构概览)
3. [功能重用与实现详细清单](#3-功能重用与实现详细清单)
4. [Patch 机制详细清单](#4-patch-机制详细清单)
5. [Ascend vs CUDA 详细对比](#5-ascend-vs-cuda-详细对比)
6. [总结与建议](#6-总结与建议)

---

## 1. vLLM 完整功能列表

### 1.1 核心架构模块

```
vLLM 项目结构 (1621 Python 文件):
├── vllm/
│   ├── v1/                          # v1 架构 (253 Python 文件)
│   │   ├── attention/               # Attention Backend 系统 (37 文件)
│   │   │   ├── backends/            # 后端实现 (20 文件)
│   │   │   │   ├── mla/             # MLA 后端 (15 文件)
│   │   │   │   ├── flash_attn.py    # FlashAttention
│   │   │   │   ├── flashinfer.py    # FlashInfer
│   │   │   │   ├── triton_attn.py   # Triton Attention
│   │   │   │   ├── flex_attention.py # FlexAttention
│   │   │   │   ├── cpu_attn.py      # CPU Attention
│   │   │   │   ├── rocm_attn.py     # ROCm Attention
│   │   │   │   ├── gdn_attn.py      # GDN Attention
│   │   │   │   └── ...              # 其他后端
│   │   │   ├── ops/                 # Attention Ops (18 文件)
│   │   │   └── selector.py          # Backend 选择器
│   │   │
│   │   ├── core/                    # 核心调度逻辑 (13 文件)
│   │   │   ├── sched/               # 调度器 (6 文件)
│   │   │   └── ...
│   │   │
│   │   ├── engine/                  # 引擎实现 (13 文件)
│   │   │   ├── core.py              # 核心引擎
│   │   │   ├── async_llm.py         # 异步引擎
│   │   │   ├── llm_engine.py        # LLM 引擎
│   │   │   └── ...
│   │   │
│   │   ├── executor/                # 执行器 (7 文件)
│   │   │   ├── multiproc_executor.py # 多进程执行器
│   │   │   ├── ray_executor.py      # Ray 执行器
│   │   │   ├── ray_executor_v2.py   # Ray v2 执行器
│   │   │   └── uniproc_executor.py  # 单进程执行器
│   │   │
│   │   ├── worker/                  # Worker 实现 (23 文件)
│   │   │   ├── gpu_model_runner.py  # GPU 模型运行器
│   │   │   ├── gpu_worker.py        # GPU Worker
│   │   │   ├── cpu_model_runner.py  # CPU 模型运行器
│   │   │   ├── cpu_worker.py        # CPU Worker
│   │   │   ├── gpu/                 # GPU 子模块 (16 文件)
│   │   │   │   ├── sample/          # 采样 (10 文件)
│   │   │   │   ├── spec_decode/     # 推测解码 (4 文件)
│   │   │   │   └── ...
│   │   │   └── ...
│   │   │
│   │   ├── sample/                  # 采样逻辑 (16 文件)
│   │   │   ├── ops/                 # 采样操作 (5 文件)
│   │   │   │   ├── topk_topp_sampler.py  # Top-k/Top-p 采样
│   │   │   │   ├── topk_topp_triton.py   # Triton 实现
│   │   │   │   ├── penalties.py          # 惩罚项
│   │   │   │   ├── logprobs.py            # Logprobs
│   │   │   │   └── bad_words.py           # Bad words 过滤
│   │   │   ├── logits_processor/    # Logits 处理器 (3 文件)
│   │   │   ├── sampler.py           # 采样器
│   │   │   ├── rejection_sampler.py # Rejection 采样器
│   │   │   └── metadata.py          # 采样元数据
│   │   │
│   │   ├── spec_decode/             # 推测解码 (12 文件)
│   │   │   ├── eagle/               # Eagle 方法
│   │   │   ├── medusa/              # Medusa 方法
│   │   │   └── ...
│   │   │
│   │   ├── kv_offload/              # KV Offload (13 文件)
│   │   │   ├── cpu/                 # CPU Offload (6 文件)
│   │   │   ├── worker/              # Worker Offload (2 文件)
│   │   │   └── ...
│   │   │
│   │   ├── structured_output/       # 结构化输出 (7 文件)
│   │   ├── metrics/                 # 指标 (7 文件)
│   │   └── pool/                    # 内存池 (2 文件)
│   │
│   ├── config/                      # 配置系统
│   │   ├── model.py                 # 模型配置
│   │   ├── cache.py                 # Cache 配置
│   │   ├── load.py                  # 加载配置
│   │   ├── compilation.py           # 编译配置
│   │   └── ...
│   │
│   ├── distributed/                 # 分布式通信
│   │   ├── parallel_state.py        # 并行状态
│   │   ├── kv_transfer/             # KV 传输
│   │   ├── ec_transfer/             # EC 传输
│   │   ├── eplb/                    # EPLB 负载均衡
│   │   └── device_communicators/    # 设备通信器
│   │
│   ├── model_executor/              # 模型执行器
│   │   ├── models/                  # 模型实现 (100+ 模型)
│   │   │   ├── llama.py             # LLaMA 系列
│   │   │   ├── qwen.py              # Qwen 系列
│   │   │   ├── deepseek.py          # DeepSeek 系列
│   │   │   ├── glm.py               # GLM 系列
│   │   │   └── ...
│   │   │
│   │   ├── layers/                  # 层实现
│   │   │   ├── attention.py         # Attention 层
│   │   │   ├── linear.py            # Linear 层
│   │   │   ├── mlp.py               # MLP 层
│   │   │   ├── quantization/        # 量化层 (101 文件)
│   │   │   │   ├── awq.py           # AWQ 量化
│   │   │   │   ├── gptq.py          # GPTQ 量化
│   │   │   │   ├── fp8.py           # FP8 量化
│   │   │   │   ├── bitsandbytes.py  # BitsAndBytes
│   │   │   │   ├── gguf.py          # GGUF 格式
│   │   │   │   ├── mxfp4.py         # MXFP4 量化
│   │   │   │   ├── turboquant/      # TurboQuant
│   │   │   │   └── ...
│   │   │   └── ...
│   │   │
│   │   ├── model_loader/            # 模型加载器
│   │   ├── kernels/                 # 自定义 kernel
│   │   └── offloader/               # 模型卸载器
│   │
│   ├── engine/                      # 引擎核心
│   │   ├── async_llm_engine.py      # 异步引擎
│   │   ├── llm_engine.py            # 同步引擎
│   │   └── ...
│   │
│   ├── entrypoints/                 # API 入口
│   │   ├── openai/                  # OpenAI API
│   │   │   ├── chat_completion/     # Chat API
│   │   │   ├── completions/         # Completions API
│   │   │   ├── embeddings/          # Embeddings API
│   │   │   └── ...
│   │   └── ...
│   │
│   ├── platforms/                   # 平台抽象层 (7 文件)
│   │   ├── interface.py             # Platform 接口
│   │   ├── cuda.py                  # CUDA 平台
│   │   ├── rocm.py                  # ROCm 平台
│   │   ├── cpu.py                   # CPU 平台
│   │   ├── tpu.py                   # TPU 平台
│   │   ├── xpu.py                   # XPU 平台
│   │   └── zen_cpu.py               # Zen CPU 平台
│   │
│   ├── lora/                        # LoRA 支持
│   │   ├── worker_manager.py        # LoRA Worker 管理器
│   │   ├── request.py               # LoRA 请求
│   │   └── punica_wrapper/          # Punica 包装器
│   │
│   ├── multimodal/                  # 多模态支持
│   │   ├── inputs.py                # 多模态输入
│   │   ├── encoder.py               # 编码器
│   │   └── ...
│   │
│   └── transformers_utils/          # Transformers 工具
```

### 1.2 功能分类详细列表

#### **1.2.1 推理引擎核心**

| 功能模块 | 子功能 | 实现文件 | 说明 |
|---------|-------|---------|------|
| **引擎架构** | v1 Engine | `v1/engine/` | 新一代引擎架构 (推荐) |
| | v0 Engine | `engine/` | 传统引擎 (兼容性) |
| | Async Engine | `engine/async_llm_engine.py` | 异步引擎支持 |
| | LLM Engine | `engine/llm_engine.py` | 同步引擎 |
| **调度系统** | Scheduler | `v1/core/sched/` | 请求调度 (6 文件) |
| | Chunked Prefill | `v1/core/sched/` | 分块 prefill |
| | Balance Scheduling | `v1/core/sched/` | 均衡调度 |
| | Priority Scheduling | `v1/core/sched/` | 优先级调度 |
| | Dynamic Batch | `v1/core/sched/` | 动态批处理 |
| **执行器** | Multiproc Executor | `v1/executor/multiproc_executor.py` | 多进程执行器 (WorkerProc) |
| | Ray Executor | `v1/executor/ray_executor.py` | Ray 分布式 |
| | Ray Executor v2 | `v1/executor/ray_executor_v2.py` | Ray v2 分布式 |
| | Uniproc Executor | `v1/executor/uniproc_executor.py` | 单进程执行器 |

#### **1.2.2 模型支持**

| 功能模块 | 子功能 | 说明 |
|---------|-------|------|
| **模型架构** | LLaMA 系列 | LLaMA, LLaMA-2, LLaMA-3, LLaMA-4 |
| | Qwen 系列 | Qwen, Qwen2, Qwen2.5, Qwen3, Qwen3-VL, Qwen3.5 |
| | DeepSeek 系列 | DeepSeek-V2, V3, V4, Coder, R1 |
| | GLM 系列 | glm-5.1, glm-5.1 V3, glm-5.1 V4 |
| | MiniMax 系列 | MiniMax-M1, MiniMax-M2 |
| | Kimi 系列 | Kimi-K2.5 |
| | Mixtral/Mistral | MoE 模型支持 |
| | Multi-Modal | LLaVA, Qwen-VL, InternVL, DeepSeek-VL 等 |
| | Mamba | Mamba-1, Mamba-2 |
| **模型特性** | MoE 支持 | Mixture of Experts |
| | MLA 支持 | Multi-Head Latent Attention (DeepSeek, glm-5.1 V3) |
| | DSA 支持 | Dynamic Sparse Attention (glm-5.1 V4) |
| | SFA 支持 | Sparse Flash Attention |
| | GQA/MQA | Grouped/Multi-Query Attention |
| | Sliding Window | 滑动窗口注意力 |
| | Prefix Caching | 前缀缓存 |
| | GDN | Gated Delta Network |

#### **1.2.3 Attention 系统**

| 功能模块 | 子功能 | 实现文件 | 说明 |
|---------|-------|---------|------|
| **Backend 系统** | Backend 注册机制 | `v1/attention/backends/registry.py` | AttentionBackendEnum |
| | Backend 选择器 | `v1/attention/selector.py` | get_attn_backend() |
| **CUDA Backend** | FlashAttention | `v1/attention/backends/flash_attn.py` | FlashAttention v2/v3 |
| | FlashAttention DiffKV | `v1/attention/backends/flash_attn_diffkv.py` | DiffKV 支持 |
| | FlashInfer | `v1/attention/backends/flashinfer.py` | FlashInfer 后端 |
| | Triton Attention | `v1/attention/backends/triton_attn.py` | Triton 实现 |
| | FlexAttention | `v1/attention/backends/flex_attention.py` | FlexAttention |
| | TurboQuant | `v1/attention/backends/turboquant_attn.py` | TurboQuant |
| **MLA Backend** | FlashMLA | `v1/attention/backends/mla/flashmla.py` | FlashMLA |
| | FlashMLA Sparse | `v1/attention/backends/mla/flashmla_sparse.py` | Sparse FlashMLA |
| | FlashInferMLA | `v1/attention/backends/mla/flashinfer_mla.py` | FlashInfer MLA |
| | FlashInferMLA Sparse | `v1/attention/backends/mla/flashinfer_mla_sparse.py` | Sparse FlashInfer MLA |
| | TritonMLA | `v1/attention/backends/mla/triton_mla.py` | Triton MLA |
| | AiterMLA | `v1/attention/backends/mla/rocm_aiter_mla.py` | ROCm Aiter MLA |
| | CutlassMLA | `v1/attention/backends/mla/cutlass_mla.py` | Cutlass MLA |
| **其他 Backend** | ROCm Backend | `v1/attention/backends/rocm_attn.py` | ROCm Attention |
| | ROCm Aiter | `v1/attention/backends/rocm_aiter_fa.py` | ROCm Aiter |
| | CPU Backend | `v1/attention/backends/cpu_attn.py` | CPU Attention |
| | Mamba Backend | `v1/attention/backends/mamba_attn.py` | Mamba Attention |
| | Mamba1 Backend | `v1/attention/backends/mamba1_attn.py` | Mamba-1 |
| | Mamba2 Backend | `v1/attention/backends/mamba2_attn.py` | Mamba-2 |
| | Linear Backend | `v1/attention/backends/linear_attn.py` | Linear Attention |
| | Short Conv | `v1/attention/backends/short_conv_attn.py` | Short Convolution |
| | GDN Backend | `v1/attention/backends/gdn_attn.py` | Gated Delta Network |
| | Tree Backend | `v1/attention/backends/tree_attn.py` | Tree Attention |

#### **1.2.4 KV Cache 管理**

| 功能模块 | 子功能 | 说明 |
|---------|-------|------|
| **KV Cache 结构** | PagedAttention | 分页 KV cache (默认 block_size=16) |
| | Block Manager | Block 分配管理 |
| | KV Cache Interface | KV cache 接口抽象 (v1) |
| | KV Cache Group | KV cache 分组 |
| **KV Cache 优化** | Prefix Caching | 前缀缓存复用 |
| | KV Offloading | KV cache 卸载到 CPU |
| | KV Transfer | KV cache 传输 (分布式) |
| | KV Compression | KV cache 压缩 |
| | KV Reuse | KV cache 复用管理 |
| **量化支持** | FP8 KV Cache | FP8 格式 KV cache |
| | INT8 KV Cache | INT8 格式 KV cache |
| | C8 KV Cache | Compressed KV cache |

#### **1.2.5 采样与解码**

| 功能模块 | 子功能 | 实现文件 | 说明 |
|---------|-------|---------|------|
| **采样策略** | Top-k Sampling | `v1/sample/ops/topk_topp_sampler.py` | Top-k 过滤 |
| | Top-p Sampling | `v1/sample/ops/topk_topp_sampler.py` | Top-p (Nucleus) 过滤 |
| | Min-p Sampling | `v1/sample/` | Min-p 过滤 |
| | Temperature | `v1/sample/` | 温度缩放 |
| | Beam Search | `v1/sample/` | 束搜索 |
| | Bad Words | `v1/sample/ops/bad_words.py` | 坏词过滤 |
| | Penalties | `v1/sample/ops/penalties.py` | 惩罚项 |
| | Logprobs | `v1/sample/ops/logprobs.py` | Logprobs 计算 |
| **采样优化** | FlashInfer Sampler | `v1/sample/ops/topk_topp_sampler.py` | FlashInfer 采样 |
| | Triton Sampler | `v1/sample/ops/topk_topp_triton.py` | Triton 采样 |
| | Aiter Sampler | `v1/sample/ops/topk_topp_sampler.py` | ROCm Aiter 采样 |
| **推测解码** | Eagle/Eagle3 | `v1/spec_decode/` | Eagle 方法 |
| | Medusa | `v1/spec_decode/` | Medusa 方法 |
| | MTP | `v1/spec_decode/` | Multi-Token Prediction |
| | Ngram | `v1/spec_decode/` | Ngram 方法 |
| **Rejection Sampling** | Speculative Decoding | `v1/sample/rejection_sampler.py` | Rejection 采样 |

#### **1.2.6 分布式支持**

| 功能模块 | 子功能 | 说明 |
|---------|-------|------|
| **并行策略** | Tensor Parallel (TP) | 张量并行 |
| | Pipeline Parallel (PP) | 流水线并行 |
| | Data Parallel (DP) | 数据并行 |
| | Expert Parallel (EP) | 专家并行 (MoE) |
| | Context Parallel (CP) | 上下文并行 |
| | Sequence Parallel (SP) | 序列并行 |
| **通信库** | NCCL | NVIDIA GPU 通信 |
| | HCCL | Ascend NPU 通信 |
| | Gloo | CPU 通信 |
| | Ray | Ray 分布式框架 |
| **分布式特性** | All-Reduce | 全局归约 |
| | All-Gather | 全局收集 |
| | All-to-All | 全局交换 (MoE) |
| | Reduce-Scatter | 归约散射 |
| | KV Transfer | KV cache 传输 |
| | EC Transfer | Expert Coordination 传输 |
| **负载均衡** | EPLB | Expert Parallel Load Balancing |
| | Balance Scheduling | 均衡调度 |

#### **1.2.7 量化支持**

| 量化方法 | CUDA 支持 | ROCm 支持 | 说明 |
|---------|----------|----------|------|
| **FP8** | ✅ | ✅ | FP8 E4M3/E5M2 |
| **INT8** | ✅ | ✅ | INT8 per-token/per-channel |
| **GPTQ** | ✅ | ✅ | GPTQ 量化 |
| **AWQ** | ✅ | ✅ | AWQ 量化 |
| **AWQ-Marlin** | ✅ | ⚠️ | AWQ Marlin kernel |
| **GPTQ-Marlin** | ✅ | ⚠️ | GPTQ Marlin kernel |
| **BitsAndBytes** | ✅ | ⚠️ | 4bit/8bit 量化 |
| **FP4** | ✅ | ⚠️ | FP4 格式 |
| **MXFP4** | ✅ | ⚠️ | MXFP4 格式 |
| **GGUF** | ✅ | ✅ | GGUF 格式支持 |
| **FP6** | ✅ | ⚠️ | FP6 格式 |
| **ModelOpt** | ✅ | ⚠️ | NVIDIA ModelOpt |
| **FBGEMM FP8** | ✅ | ⚠️ | FBGEMM FP8 |
| **CompressedTensors** | ✅ | ✅ | 压缩张量格式 |
| **TurboQuant** | ✅ | ⚠️ | TurboQuant |
| **Quark** | ✅ | ⚠️ | Quark 量化 |
| **Humming** | ✅ | ⚠️ | Humming 量化 |
| **INC** | ✅ | ⚠️ | Intel Neural Compressor |
| **ModelSlim** | ⚠️ | ⚠️ | Ascend 专用 |

#### **1.2.8 LoRA 支持**

| 功能模块 | 子功能 | 实现文件 | 说明 |
|---------|-------|---------|------|
| **LoRA 核心** | LoRA Manager | `lora/worker_manager.py` | LoRA Worker 管理器 |
| | LoRA Request | `lora/request.py` | LoRA 请求 |
| | Punica Wrapper | `lora/punica_wrapper/` | Punica 内核包装器 |
| **LoRA 特性** | Multi-LoRA | ✅ | 多 LoRA 支持 |
| | LoRA Adapter | ✅ | LoRA 适配器管理 |
| | LoRA Offloading | ✅ | LoRA 卸载 |
| | LoRA Budget | ✅ | LoRA 预算管理 |

#### **1.2.9 多模态支持**

| 功能模块 | 子功能 | 说明 |
|---------|-------|------|
| **输入处理** | Image Input | 图像输入处理 |
| | Audio Input | 音频输入处理 |
| | Video Input | 视频输入处理 |
| | Multi-Modal Budget | 多模态预算管理 |
| **模型支持** | LLaVA | 图像理解模型 |
| | Qwen-VL | Qwen 视觉语言模型 |
| | InternVL | InternVL 视觉语言模型 |
| | DeepSeek-VL | DeepSeek 视觉语言模型 |
| | MiniMax-VL | MiniMax 视觉语言模型 |
| **Encoder** | Vision Encoder | 视觉编码器 |
| | Audio Encoder | 音频编码器 |
| | Encoder Budget | 编码器预算 |

#### **1.2.10 API 与服务**

| 功能模块 | 子功能 | 实现文件 | 说明 |
|---------|-------|---------|------|
| **OpenAI API** | Chat Completions | `entrypoints/openai/chat_completion/` | Chat API |
| | Completions | `entrypoints/openai/completions/` | Completions API |
| | Embeddings | `entrypoints/openai/embeddings/` | Embeddings API |
| | Models | `entrypoints/openai/models/` | Models API |
| | Batch | `entrypoints/openai/batch/` | Batch API |
| **API 特性** | Streaming | ✅ | 流式响应 |
| | Batch API | ✅ | 批处理 API |
| | Tool Calling | ✅ | 工具调用 |
| | Structured Output | ✅ | 结构化输出 (JSON, Regex) |
| | Reasoning | ✅ | 推理模式 (DeepSeek R1) |
| | Response Format | ✅ | 响应格式控制 |

#### **1.2.11 配置系统**

| 配置模块 | 配置项 | 说明 |
|---------|-------|------|
| **ModelConfig** | model, tokenizer, dtype | 模型配置 |
| | max_model_len, trust_remote_code | 模型参数 |
| **CacheConfig** | block_size, gpu_memory_utilization | Cache 配置 |
| | cache_dtype, cpu_kvcache_space_bytes | Cache 参数 |
| **LoadConfig** | load_format, download_dir | 加载配置 |
| | safetensors_load_strategy | 加载策略 |
| **CompilationConfig** | cudagraph_mode, use_inductor | 编译配置 |
| | cudagraph_capture_sizes | Graph 捕获大小 |
| **SchedulerConfig** | max_num_seqs, max_num_batched_tokens | 调度配置 |
| | enable_chunked_prefill | Chunked Prefill |
| **ParallelConfig** | tensor_parallel_size, pipeline_parallel_size | 并行配置 |
| | data_parallel_size, expert_parallel_size | 并行参数 |
| **AttentionConfig** | backend, use_mla | Attention 配置 |
| | use_sparse, use_cascade_attn | Attention 参数 |

#### **1.2.12 平台支持**

| 平台 | 实现文件 | 说明 |
|-----|---------|------|
| **CUDA** | `platforms/cuda.py` | NVIDIA GPU (默认) |
| **ROCm** | `platforms/rocm.py` | AMD GPU |
| **CPU** | `platforms/cpu.py` | CPU 后端 |
| **TPU** | `platforms/tpu.py` | Google TPU |
| **XPU** | `platforms/xpu.py` | Intel XPU |
| **Zen CPU** | `platforms/zen_cpu.py` | AMD Zen CPU |
| **OOT** | Platform Plugin | Out-of-Tree 平台插件 |

---

## 2. vLLM-Ascend 架构概览

### 2.1 项目结构

```
vLLM-Ascend 项目结构 (378 Python 文件):
├── vllm_ascend/
│   ├── platform.py                 # NPUPlatform 平台实现
│   ├── ascend_config.py            # Ascend 配置
│   ├── ascend_forward_context.py   # Forward Context
│   │
│   ├── attention/                  # Attention 实现 (17 文件)
│   │   ├── attention_v1.py         # 标准 Attention Backend
│   │   ├── mla_v1.py               # MLA Backend (glm-5.1 V3)
│   │   ├── dsa_v1.py               # DSA Backend (glm-5.1 V4)
│   │   ├── sfa_v1.py               # SFA Backend
│   │   ├── fa3_v1.py               # FlashAttention v3 Backend
│   │   ├── context_parallel/       # Context Parallel (5 文件)
│   │   │   ├── attention_cp.py     # CP Attention
│   │   │   ├── mla_cp.py           # MLA CP
│   │   │   ├── dsa_cp.py           # DSA CP
│   │   │   └── sfa_cp.py           # SFA CP
│   │   └── kvcomp_attn/            # KV Compressed Attention
│   │
│   ├── ops/                        # NPU 算子 (21 文件)
│   │   ├── activation.py           # 激活函数
│   │   ├── linear.py               # 线性层
│   │   ├── layernorm.py            # LayerNorm
│   │   ├── rotary_embedding.py     # RoPE
│   │   ├── mla.py                  # MLA 算子
│   │   ├── dsa.py                  # DSA 算子
│   │   ├── gdn.py                  # GDN 算子
│   │   ├── fused_moe/              # Fused MoE (11 文件)
│   │   │   ├── fused_moe.py        # Fused MoE 实现
│   │   │   ├── moe_mlp.py          # MoE MLP
│   │   │   ├── token_dispatcher.py # Token 调度器
│   │   │   └── ...
│   │   ├── triton/                 # Triton 算子 (40 文件)
│   │   │   ├── activation/         # 激活函数
│   │   │   ├── batch_invariant/    # Batch Invariant 算子
│   │   │   ├── fla/                # Flash Linear Attention (14 文件)
│   │   │   ├── mamba/              # Mamba 算子
│   │   │   ├── linearnorm/         # LinearNorm 算子
│   │   │   ├── spec_decode/        # 推测解码算子
│   │   │   └── ...
│   │   └── ...
│   │
│   ├── worker/                     # Worker 实现 (6 文件)
│   │   ├── worker.py               # NPUWorker
│   │   ├── model_runner_v1.py      # NPU Model Runner (221KB)
│   │   ├── block_table.py          # Block Table
│   │   ├── npu_input_batch.py      # NPU Input Batch
│   │   ├── kvcomp_utils.py         # KV Compressed Utils
│   │   ├── pcp_utils.py            # Prefill Context Parallel Utils
│   │   ├── v2/                     # v2 Model Runner (15 文件)
│   │   │   ├── model_runner_v2.py  # v2 Model Runner
│   │   │   ├── sample/             # v2 采样 (5 文件)
│   │   │   ├── spec_decode/        # v2 推测解码
│   │   │   └── ...
│   │   └── ...
│   │
│   ├── distributed/                # 分布式通信 (32 文件)
│   │   ├── device_communicators/   # 设备通信器 (3 文件)
│   │   │   ├── npu_communicator.py # NPU Communicator (HCCL)
│   │   │   ├── pyhccl.py           # PyHCCL 封装
│   │   │   └── pyhccl_wrapper.py   # PyHCCL Wrapper
│   │   ├── kv_transfer/            # KV Transfer (26 文件)
│   │   │   ├── kv_p2p/             # P2P 传输 (3 文件)
│   │   │   │   ├── mooncake_connector.py         # Mooncake 连接器
│   │   │   │   ├── mooncake_hybrid_connector.py  # Mooncake Hybrid
│   │   │   │   └── mooncake_layerwise_connector.py
│   │   │   ├── kv_pool/            # KV Pool (17 文件)
│   │   │   │   ├── ascend_store/   # Ascend Store (9 文件)
│   │   │   │   │   ├── backend/    # Backend 实现 (4 文件)
│   │   │   │   │   │   ├── memcache_backend.py
│   │   │   │   │   │   ├── mooncake_backend.py
│   │   │   │   │   │   └── yuanrong_backend.py
│   │   │   │   │   └── ...
│   │   │   │   ├── cpu_offload/    # CPU Offload (3 文件)
│   │   │   │   ├── lmcache_ascend_connector.py
│   │   │   │   └── ucm_connector.py
│   │   │   └── ...
│   │   ├── parallel_state.py       # 并行状态
│   │   └── utils.py                # 分布式工具
│   │
│   ├── quantization/               # 量化支持 (20 文件)
│   │   ├── methods/                # 量化方法 (14 文件)
│   │   │   ├── w4a16.py            # W4A16 量化
│   │   │   ├── w4a8.py             # W4A8 量化
│   │   │   ├── w8a16.py            # W8A16 量化
│   │   │   ├── w8a8_static.py      # W8A8 Static 量化
│   │   │   ├── w8a8_dynamic.py     # W8A8 Dynamic 量化
│   │   │   ├── w8a8_mxfp8.py       # W8A8 MXFP8 量化
│   │   │   ├── w4a4_mxfp4.py       # W4A4 MXFP4 量化
│   │   │   ├── kv_c8.py            # KV C8 量化
│   │   │   └── ...
│   │   ├── compressed_tensors_config.py  # CompressedTensors
│   │   ├── modelslim_config.py     # ModelSlim 配置
│   │   └── utils.py                # 量化工具
│   │
│   ├── compilation/                # 编译优化 (13 文件)
│   │   ├── acl_graph.py            # ACL Graph (NPU Graph)
│   │   ├── compiler_interface.py   # 编译器接口
│   │   ├── graph_fusion_pass_manager.py  # Graph Fusion
│   │   └── passes/                 # Pass 实现 (9 文件)
│   │       ├── sequence_parallelism.py   # 序列并行
│   │       └── ...
│   │
│   ├── sample/                     # 采样实现 (3 文件)
│   │   ├── sampler.py              # NPU Sampler
│   │   ├── rejection_sampler.py    # Rejection Sampler
│   │   └── penalties.py            # 惩罚项
│   │
│   ├── spec_decode/                # 推测解码 (10 文件)
│   │   └── ...
│   │
│   ├── lora/                       # LoRA 支持 (3 文件)
│   │   ├── punica_npu.py           # NPU Punica Wrapper
│   │   └── ...
│   │
│   ├── models/                     # 模型适配 (3 文件)
│   │   ├── layer/                  # 层适配
│   │   └── ...
│   │
│   ├── model_loader/               # 模型加载器 (11 文件)
│   │   ├── netloader/              # 网络加载器 (6 文件)
│   │   ├── rfork/                  # RFork 加载器 (5 文件)
│   │   └── ...
│   │
│   ├── core/                       # 核心扩展 (5 文件)
│   │   ├── scheduler_dynamic_batch.py    # 动态批处理调度器
│   │   ├── scheduler_profiling_chunk.py  # Profiling Chunk 调度器
│   │   ├── recompute_scheduler.py        # Recompute 调度器
│   │   └── ...
│   │
│   ├── eplb/                       # EPLB 负载均衡 (12 文件)
│   │   ├── core/                   # EPLB 核心 (3 文件)
│   │   │   ├── policy/             # 策略 (6 文件)
│   │   │   └── ...
│   │   ├── adaptor/                # 适配器
│   │   └── ...
│   │
│   ├── profiler/                   # 性能分析 (1 文件)
│   │   └── ...
│   │
│   ├── patch/                      # Patch 机制 (820 行文档)
│   │   ├── platform/               # Platform Patch (21 文件)
│   │   │   ├── patch_distributed.py
│   │   │   ├── patch_mamba_config.py
│   │   │   ├── patch_multiproc_executor.py
│   │   │   ├── patch_balance_schedule.py
│   │   │   ├── patch_kv_cache_interface.py
│   │   │   ├── patch_minimax_m2_config.py
│   │   │   ├── patch_speculative_config.py
│   │   │   └── ...
│   │   └── worker/                 # Worker Patch (27 文件)
│   │       ├── patch_cudagraph.py
│   │       ├── patch_distributed.py
│   │       ├── patch_deepseek_compressor.py
│   │       ├── patch_minimax_m2.py
│   │       ├── patch_qwen3_5.py
│   │       ├── patch_v2/           # v2 Patch (6 文件)
│   │       │   ├── patch_block_table.py
│   │       │   ├── patch_input_batch.py
│   │       │   ├── patch_model_state.py
│   │       │   └── ...
│   │       └── ...
│   │
│   ├── _310p/                      # 310P 芯片专用 (7 文件)
│   │   ├── quantization/           # 310P 量化
│   │   ├── attention/              # 310P Attention
│   │   └── worker_310p.py          # 310P Worker
│   │
│   ├── xlite/                      # Xlite 支持 (4 文件)
│   │   ├── xlite_worker.py         # Xlite Worker
│   │   └── ...
│   │
│   └── utils.py                    # 工具函数
```

### 2.2 代码规模统计

| 模块 | vLLM 文件数 | vLLM-Ascend 文件数 | 重用比例 |
|-----|-----------|------------------|---------|
| **总计** | 1621 | 378 | 23.3% |
| **v1 核心** | 253 | - | 完全重用 |
| **Attention** | 37 | 17 | 部分重写 |
| **Worker** | 23 | 6 | 大部分重写 |
| **Distributed** | - | 32 | 完全重写 |
| **Quantization** | 101 | 20 | 部分重写 |
| **Ops/Kernels** | - | 61 | 完全重写 |
| **Patch** | - | 48 | 补丁机制 |

### 2.3 核心设计原则

#### **2.3.1 平台抽象层 (Platform Abstraction)**

```python
# vLLM Platform 接口 (vllm/platforms/interface.py)
class Platform:
    device_name: str
    device_type: str
    
    @classmethod
    def get_attn_backend_cls(cls, ...) -> str:
        """返回 Attention Backend 类路径"""
        
    @classmethod
    def get_device_communicator_cls(cls) -> str:
        """返回设备通信器类路径"""
        
    @classmethod
    def check_and_update_config(cls, vllm_config) -> None:
        """检查和更新配置"""
```

```python
# vLLM-Ascend NPUPlatform (vllm_ascend/platform.py)
class NPUPlatform(Platform):
    _enum = PlatformEnum.OOT  # Out-of-Tree 插件
    device_name: str = "npu"
    device_type: str = "npu"
    
    @classmethod
    def get_attn_backend_cls(cls, selected_backend, attn_selector_config, ...):
        """Ascend Attention Backend 映射"""
        backend_map = {
            (True, False, False): "vllm_ascend.attention.mla_v1.AscendMLABackend",
            (False, False, False): "vllm_ascend.attention.attention_v1.AscendAttentionBackend",
            (True, True, False): "vllm_ascend.attention.sfa_v1.AscendSFABackend",
            (True, False, True): "vllm_ascend.attention.dsa_v1.AscendDSABackend",
        }
        
    @classmethod
    def get_device_communicator_cls(cls) -> str:
        return "vllm_ascend.distributed.device_communicators.npu_communicator.NPUCommunicator"
```

#### **2.3.2 Patch 机制**

vLLM-Ascend 使用 Monkey Patch 机制来适配 vLLM 代码，分为两类：

1. **Platform Patch** (全局 Patch，在 worker 启动前应用)
   - 位置: `vllm_ascend/patch/platform/`
   - 调用点: `NPUPlatform.pre_register_and_update()`
   - 文件数: 21 个

2. **Worker Patch** (Worker Patch，在 worker 初始化时应用)
   - 位置: `vllm_ascend/patch/worker/`
   - 调用点: `NPUWorker.__init__()`
   - 文件数: 27 个

---

## 3. 功能重用与实现详细清单

### 3.1 完全重用 vLLM 功能 (60%)

这些功能直接使用 vLLM 的实现，无需修改：

| 功能模块 | vLLM 实现 | 说明 |
|---------|----------|------|
| **引擎架构** | `v1/engine/` | 完全重用 v1 引擎 |
| **调度系统** | `v1/core/sched/` | 完全重用调度逻辑 |
| **执行器** | `v1/executor/multiproc_executor.py` | 完全重用多进程执行器 |
| **配置系统** | `config/` | 完全重用配置系统 |
| **模型定义** | `model_executor/models/` | 完全重用模型定义 |
| **API 服务** | `entrypoints/` | 完全重用 API 层 |
| **LoRA 管理** | `lora/` | 完全重用 LoRA 管理 |
| **多模态处理** | `multimodal/` | 完全重用多模态处理 |
| **采样逻辑** | `v1/sample/` | 大部分重用 (除 FlashInfer) |
| **推测解码** | `v1/spec_decode/` | 大部分重用 |
| **KV Cache Interface** | `v1/kv_cache_interface.py` | 完全重用接口定义 |
| **Metrics** | `v1/metrics/` | 完全重用指标系统 |

### 3.2 部分重用/适配功能 (30%)

这些功能基于 vLLM 的接口，但需要 Ascend 特定实现：

| 功能模块 | vLLM 接口 | Ascend 实现 | 说明 |
|---------|----------|-----------|------|
| **Platform** | `Platform` 基类 | `NPUPlatform` | 平台抽象层 |
| **Attention Backend** | `AttentionBackend` 接口 | `AscendAttentionBackend` | Attention 后端 |
| **Worker** | `GPUWorker` | `NPUWorker` | Worker 实现 |
| **Model Runner** | `GPUModelRunner` | `NPUModelRunner` | 模型运行器 |
| **Sampler** | `Sampler` 基类 | `NPUSampler` | 采样器 |
| **LoRA Punica** | `PunicaWrapper` | `PunicaWrapperNPU` | LoRA Punica 内核 |
| **Quantization** | 量化接口 | Ascend 量化方法 | 量化实现 |
| **KV Transfer** | `KVTransfer` 接口 | Ascend KV Transfer | KV 传输 |

### 3.3 完全重写功能 (10%)

这些功能需要针对 Ascend NPU 完全重写：

| 功能模块 | 说明 | Ascend 实现 |
|---------|------|-----------|
| **Distributed Communication** | HCCL 替代 NCCL | `distributed/device_communicators/` |
| **Custom Ops** | NPU 特定算子 | `ops/` (61 文件) |
| **Triton Kernels** | Triton 算子适配 | `ops/triton/` (40 文件) |
| **ACL Graph** | NPU Graph 优化 | `compilation/acl_graph.py` |
| **FlashAttention** | NPU FlashAttention | `ops/` + `attention/` |
| **Fused MoE** | Fused MoE 算子 | `ops/fused_moe/` (11 文件) |
| **KV Transfer Backend** | KV 传输后端 | `distributed/kv_transfer/` (26 文件) |

---

## 4. Patch 机制详细清单

### 4.1 Platform Patch (21 个文件)

Platform Patch 在 worker 启动前应用，影响全局配置和行为：

| Patch 文件 | Patch 目标 | 目的 | 状态 |
|-----------|----------|------|------|
| `patch_distributed.py` | `torch.distributed.all_reduce` | 310P 张量对齐 | 临时 |
| `patch_mamba_config.py` | `HybridAttentionMambaModelConfig` | Block size 适配 | 待上游合并 |
| `patch_mamba_config_310.py` | Mamba config (310P) | 310P 特定配置 | 临时 |
| `patch_multiproc_executor.py` | `MultiprocExecutor` | EPLB daemon=False | 待上游修复 |
| `patch_balance_schedule.py` | `EngineCoreProc`, `Scheduler` | 均衡调度 | 待上游合并 |
| `patch_kv_cache_interface.py` | `KVCacheSpec` | KV cache 接口扩展 | 长期 |
| `patch_kv_cache_utils.py` | KV cache 工具 | KV cache 工具扩展 | 长期 |
| `patch_kv_cache_coordinator.py` | KV cache 协调器 | KV cache 协调 | 长期 |
| `patch_minimax_m2_config.py` | `ModelConfig`, `SpeculativeConfig` | MiniMax-M2 配置 | 待上游支持 |
| `patch_minimax_m2_tool_call_parser.py` | Tool call parser | MiniMax-M2 工具调用 | 待上游支持 |
| `patch_minimax_usage_accounting.py` | OpenAI serving | MiniMax 使用统计 | 待上游支持 |
| `patch_speculative_config.py` | `SpeculativeConfig` | 推测解码配置 | 待上游合并 |
| `patch_mla_prefill_backend.py` | MLA prefill backend | MLA prefill 优化 | 长期 |
| `patch_profiling_chunk.py` | Profiling chunk scheduler | Profiling chunk 调度 | 长期 |
| `patch_deepseek_v4_thinking.py` | DeepSeek-V4 thinking | DeepSeek-V4 思考模式 | 待上游支持 |
| `patch_deepseek_v4_tool_call_parser.py` | Tool call parser | DeepSeek-V4 工具调用 | 待上游支持 |
| `patch_glm47_tool_call_parser.py` | Tool call parser | GLM-47 工具调用 | 待上游支持 |
| `patch_glm_tool_call_streaming.py` | Tool call streaming | GLM 工具调用流式 | 待上游支持 |
| `patch_tool_choice_none_content.py` | Tool choice | Tool choice none | 待上游修复 |
| `patch_torch_accelerator.py` | `torch.Accelerator` | Torch accelerator 适配 | 长期 |
| `patch_camem_allocator.py` | CAMemAllocator | 内存分配器适配 | 长期 |

### 4.2 Worker Patch (27 个文件)

Worker Patch 在 worker 初始化时应用，影响 worker 内部行为：

| Patch 文件 | Patch 目标 | 目的 | 状态 |
|-----------|----------|------|------|
| `patch_cudagraph.py` | CUDA Graph | ACL Graph 适配 | 长期 |
| `patch_distributed.py` | 分布式通信 | HCCL 通信适配 | 长期 |
| `patch_deepseek_compressor.py` | DeepSeek compressor | DeepSeek 压缩器适配 | 长期 |
| `patch_deepseek_mtp.py` | DeepSeek MTP | DeepSeek MTP 适配 | 长期 |
| `patch_minimax_m2.py` | MiniMax-M2 model | MiniMax-M2 模型适配 | 长期 |
| `patch_minimax_m2_linear_attn.py` | MiniMax-M2 linear attn | MiniMax-M2 线性注意力 | 长期 |
| `patch_qwen3_5.py` | Qwen3.5 model | Qwen3.5 模型适配 | 长期 |
| `patch_qwen3vl.py` | Qwen3-VL model | Qwen3-VL 模型适配 | 长期 |
| `patch_qwen3_next_mtp.py` | Qwen3-Next MTP | Qwen3-Next MTP 适配 | 长期 |
| `patch_qwen3_dflash.py` | Qwen3 DFlash | Qwen3 DFlash 适配 | 长期 |
| `patch_kimi_k25.py` | Kimi-K2.5 model | Kimi-K2.5 模型适配 | 长期 |
| `patch_gdn_attn.py` | GDN attention | GDN 注意力适配 | 长期 |
| `patch_gqa_c8.py` | GQA C8 | GQA C8 适配 | 长期 |
| `patch_mamba_utils.py` | Mamba utils | Mamba 工具适配 | 长期 |
| `patch_rejection_sampler.py` | Rejection sampler | Rejection 采样器适配 | 长期 |
| `patch_triton.py` | Triton kernels | Triton 内核适配 | 长期 |
| `patch_npugraph_ex_triton.py` | NPU Graph Triton | NPU Graph Triton 适配 | 长期 |
| `patch_weight_utils.py` | Weight utils | 权重工具适配 | 长期 |
| `patch_draft_quarot.py` | Draft QuaRot | Draft QuaRot 适配 | 长期 |
| `patch_idex_310.py` | IDEX 310P | 310P IDEX 适配 | 长期 |
| `_hccl_pg_registry.py` | HCCL process group | HCCL 进程组注册 | 长期 |
| **v2 Patch** | | | |
| `patch_v2/patch_block_table.py` | Block table (v2) | v2 Block table 适配 | 长期 |
| `patch_v2/patch_input_batch.py` | Input batch (v2) | v2 Input batch 适配 | 长期 |
| `patch_v2/patch_model_state.py` | Model state (v2) | v2 Model state 适配 | 长期 |
| `patch_v2/patch_attn_utils.py` | Attention utils (v2) | v2 Attention 工具适配 | 长期 |
| `patch_v2/patch_triton.py` | Triton (v2) | v2 Triton 适配 | 长期 |
| `patch_v2/patch_uva.py` | UVA (v2) | v2 UVA 适配 | 长期 |

### 4.3 Patch 应用流程

```python
# 1. Platform Patch (全局，在 worker 启动前)
# vllm_ascend/platform.py
class NPUPlatform(Platform):
    @classmethod
    def pre_register_and_update(cls, parser=None):
        from vllm_ascend.utils import adapt_patch
        adapt_patch(is_global_patch=True)  # 应用 platform patch

# 2. Worker Patch (worker 初始化时)
# vllm_ascend/worker/worker.py
class NPUWorker:
    def __init__(self, ...):
        from vllm_ascend.utils import adapt_patch
        adapt_patch(is_global_patch=False)  # 应用 worker patch
```

---

## 5. Ascend vs CUDA 详细对比

### 5.1 Attention Backend 对比

| 特性 | CUDA (vLLM) | Ascend (vLLM-Ascend) |
|-----|------------|---------------------|
| **Backend 注册** | `AttentionBackendEnum` | 继承 + Platform 映射 |
| **Backend 选择** | `get_attn_backend()` | `NPUPlatform.get_attn_backend_cls()` |
| **标准 Attention** | FlashAttention, FlashInfer, Triton | `AscendAttentionBackend` (torch_npu.npu_flash_attention) |
| **MLA** | FlashMLA, FlashInferMLA, TritonMLA | `AscendMLABackend` (MLAPO 算子) |
| **DSA** | - | `AscendDSABackend` (KV Scatter/Gather) |
| **SFA** | - | `AscendSFABackend` (Sparse FlashAttention) |
| **FA3** | FlashAttention v3 | `AscendFABackend` (flash_attn_npu_v3) |
| **Block Size** | 16 (默认) | 128 (默认) |
| **KV Cache Shape** | (2, num_blocks, block_size, num_heads, head_size) | MLA: (num_blocks, block_size, kv_lora_rank) |
| **Context Parallel** | - | ✅ (PCP, DCP, MLA-CP, DSA-CP, SFA-CP) |

### 5.2 Distributed Communication 对比

| 特性 | CUDA (vLLM) | Ascend (vLLM-Ascend) |
|-----|------------|---------------------|
| **通信库** | NCCL | HCCL (Huawei Collective Communication Library) |
| **Communicator** | `NcclCommunicator` | `NPUCommunicator` |
| **All-Reduce** | `nccl.allReduce()` | `hccl.all_reduce()` |
| **All-Gather** | `nccl.allGather()` | `hccl.all_gather()` |
| **All-to-All** | `nccl.allToAll()` | `hccl.all_to_all()` |
| **Reduce-Scatter** | `nccl.reduceScatter()` | `hccl.reduce_scatter()` |
| **Broadcast** | `nccl.broadcast()` | `hccl.broadcast()` |
| **Process Group** | `torch.distributed` | PyHCCL Wrapper |
| **KV Transfer** | NIXL, LMCache | Mooncake, Ascend Store, CPU Offload |
| **Expert Parallel** | ✅ | ✅ |
| **Sequence Parallel** | ✅ | ✅ (FlashComm v1/v2) |

### 5.3 量化支持对比

| 量化方法 | CUDA | Ascend | 说明 |
|---------|------|--------|------|
| **FP8** | ✅ | ✅ | E4M3/E5M2 |
| **INT8** | ✅ | ✅ | Per-token/per-channel |
| **W4A16** | ✅ | ✅ | Weight-only INT4 |
| **W8A16** | ✅ | ✅ | Weight-only INT8 |
| **W4A8** | ✅ | ✅ | Weight INT4, Activation INT8 |
| **W8A8 Static** | ✅ | ✅ | Static quantization |
| **W8A8 Dynamic** | ✅ | ✅ | Dynamic quantization |
| **W8A8 MXFP8** | ✅ | ✅ | MXFP8 format |
| **W4A4 MXFP4** | ✅ | ✅ | MXFP4 format |
| **KV C8** | ✅ | ✅ | KV cache INT8 |
| **GPTQ** | ✅ | ⚠️ | 部分支持 |
| **AWQ** | ✅ | ⚠️ | 部分支持 |
| **BitsAndBytes** | ✅ | ❌ | 不支持 |
| **GGUF** | ✅ | ⚠️ | 部分支持 |
| **ModelSlim** | ❌ | ✅ | Ascend 专用 |

### 5.4 Graph Optimization 对比

| 特性 | CUDA (vLLM) | Ascend (vLLM-Ascend) |
|-----|------------|---------------------|
| **Graph 类型** | CUDA Graph | ACL Graph (NPU Graph) |
| **Graph Mode** | NONE, PIECEWISE, FULL, FULL_DECODE_ONLY | 继承 CUDA Graph 模式 |
| **Graph Wrapper** | `CUDAGraphWrapper` | `ACLGraphWrapper` |
| **Graph Capture** | `torch.cuda.make_graphed_callables()` | `torch_npu.npu.make_graphed_callables()` |
| **Graph Sizes** | cudagraph_capture_sizes | 继承 + Ascend 特定调整 |
| **Graph Warmup** | cudagraph_num_of_warmups | 1 (固定) |
| **Splitting Ops** | vLLM splitting ops | 继承 + MLA ops |
| **Graph Fusion** | Inductor fusion | Graph Fusion Pass Manager |
| **Full Graph** | ✅ | ✅ (实验性) |
| **Piecewise Graph** | ✅ | ✅ |

### 5.5 Worker & Model Runner 对比

| 特性 | CUDA (vLLM) | Ascend (vLLM-Ascend) |
|-----|------------|---------------------|
| **Worker 类** | `GPUWorker` | `NPUWorker` |
| **Model Runner** | `GPUModelRunner` | `NPUModelRunner` |
| **Input Batch** | `GPUInputBatch` | `NPUInputBatch` |
| **Block Table** | `BlockTable` | 继承 + 扩展 |
| **设备初始化** | `torch.cuda.set_device()` | `torch.npu.set_device()` |
| **内存分析** | `torch.cuda.memory_stats()` | `torch.npu.memory_stats()` |
| **设备属性** | `torch.cuda.get_device_properties()` | `torch.npu.get_device_properties()` |
| **设备数量** | `torch.cuda.device_count()` | `torch.npu.device_count()` |
| **当前设备** | `torch.cuda.current_device()` | `torch.npu.current_device()` |
| **设备名称** | `torch.cuda.get_device_name()` | `torch.npu.get_device_name()` |
| **内存分配配置** | `PYTORCH_CUDA_ALLOC_CONF` | `PYTORCH_NPU_ALLOC_CONF` |

### 5.6 Sampling 对比

| 特性 | CUDA (vLLM) | Ascend (vLLM-Ascend) |
|-----|------------|---------------------|
| **Top-k/Top-p** | FlashInfer, Triton, Native | Native + Triton |
| **FlashInfer Sampler** | ✅ | ❌ |
| **Triton Sampler** | ✅ | ✅ (适配) |
| **Aiter Sampler** | ✅ (ROCm) | ❌ |
| **Rejection Sampler** | ✅ | ✅ (适配) |
| **Penalties** | ✅ | ✅ (适配) |
| **Bad Words** | ✅ | ✅ |
| **Logprobs** | ✅ | ✅ |

### 5.7 Model Support 对比

| 模型系列 | CUDA | Ascend | 说明 |
|---------|------|--------|------|
| **LLaMA 系列** | ✅ | ✅ | 完全支持 |
| **Qwen 系列** | ✅ | ✅ | 完全支持 (含 Qwen3, Qwen3-VL, Qwen3.5) |
| **DeepSeek 系列** | ✅ | ✅ | 完全支持 (含 V2, V3, V4, R1) |
| **GLM 系列** | ✅ | ✅ | 完全支持 (含 glm-5.1 V3/V4) |
| **MiniMax 系列** | ✅ | ✅ | 完全支持 (含 M1, M2) |
| **Kimi 系列** | ✅ | ✅ | 完全支持 (Kimi-K2.5) |
| **Mixtral/Mistral** | ✅ | ✅ | 完全支持 |
| **Mamba** | ✅ | ✅ | 完全支持 (Mamba-1, Mamba-2) |
| **Multi-Modal** | ✅ | ✅ | 完全支持 |

### 5.8 Special Features 对比

| 特性 | CUDA (vLLM) | Ascend (vLLM-Ascend) |
|-----|------------|---------------------|
| **MLA (Multi-Head Latent Attention)** | ✅ | ✅ (MLAPO 算子优化) |
| **DSA (Dynamic Sparse Attention)** | ❌ | ✅ (glm-5.1 V4) |
| **SFA (Sparse Flash Attention)** | ❌ | ✅ |
| **GDN (Gated Delta Network)** | ✅ | ✅ |
| **Context Parallel** | ⚠️ | ✅ (PCP, DCP, MLA-CP, DSA-CP, SFA-CP) |
| **Sequence Parallel** | ✅ | ✅ (FlashComm v1/v2) |
| **Expert Parallel** | ✅ | ✅ |
| **EPLB (Expert Parallel Load Balancing)** | ✅ | ✅ |
| **KV Transfer** | ✅ | ✅ (Mooncake, Ascend Store, CPU Offload) |
| **KV Offload** | ✅ | ✅ |
| **Prefix Caching** | ✅ | ✅ |
| **LoRA** | ✅ | ✅ |
| **Speculative Decoding** | ✅ | ✅ (Eagle, Eagle3, Medusa, MTP, Ngram) |
| **Structured Output** | ✅ | ✅ |
| **Tool Calling** | ✅ | ✅ |
| **Reasoning** | ✅ | ✅ (DeepSeek R1) |
| **Xlite** | ❌ | ✅ (openEuler Xlite) |

---

## 6. 总结与建议

### 6.1 代码重用总结

| 重用类型 | 比例 | 说明 |
|---------|------|------|
| **完全重用** | ~60% | 引擎、调度、配置、API、模型定义等 |
| **部分重用/适配** | ~30% | Platform、Attention Backend、Worker、Sampler、Quantization |
| **完全重写** | ~10% | Distributed、Custom Ops、Triton Kernels、ACL Graph |

### 6.2 架构优势

#### **vLLM 架构优势**
1. **平台抽象层**: 通过 `Platform` 基类实现硬件无关性
2. **Backend 注册机制**: 灵活的 Attention Backend 注册和选择
3. **v1 架构**: 高性能的 v1 引擎架构
4. **配置系统**: 强大的配置系统和验证机制
5. **多进程执行器**: 高效的多进程 Worker 管理

#### **vLLM-Ascend 架构优势**
1. **插件化设计**: 通过 OOT (Out-of-Tree) 平台插件实现
2. **Patch 机制**: 灵活的 Monkey Patch 适配机制
3. **Ascend 特定优化**: MLAPO、DSA、SFA 等 Ascend 特定优化
4. **Context Parallel**: 完整的 Context Parallel 支持 (PCP, DCP, MLA-CP, DSA-CP, SFA-CP)
5. **KV Transfer**: 多种 KV Transfer 后端 (Mooncake, Ascend Store, CPU Offload)
6. **Xlite 支持**: openEuler Xlite 支持

### 6.3 未来改进建议

#### **短期改进**
1. **减少 Patch 数量**: 推动更多 Patch 上游合并到 vLLM
2. **统一量化支持**: 统一 CUDA 和 Ascend 的量化接口
3. **改进文档**: 完善 Patch 文档和迁移指南

#### **长期改进**
1. **Platform 接口增强**: 增强 Platform 接口，减少 Patch 需求
2. **Backend 注册标准化**: 标准化 Attention Backend 注册机制
3. **分布式通信抽象**: 抽象分布式通信层，支持多种通信库
4. **Graph 优化统一**: 统一 CUDA Graph 和 ACL Graph 接口

### 6.4 关键技术对比总结

| 技术领域 | vLLM (CUDA) | vLLM-Ascend (Ascend) | 差异原因 |
|---------|------------|---------------------|---------|
| **Attention** | FlashAttention, FlashInfer | NPU FlashAttention, MLAPO | 硬件 API 不同 |
| **Distributed** | NCCL | HCCL | 通信库不同 |
| **Graph** | CUDA Graph | ACL Graph | Graph 机制不同 |
| **Block Size** | 16 | 128 | 硬件优化参数不同 |
| **KV Transfer** | NIXL, LMCache | Mooncake, Ascend Store | 传输机制不同 |
| **特殊 Attention** | - | DSA, SFA | Ascend 特定优化 |

### 6.5 贡献者指南

#### **如何添加新的 Attention Backend**
1. 继承 `AttentionBackend` 基类
2. 实现 `forward()`、`get_kv_cache_shape()` 等方法
3. 在 `NPUPlatform.get_attn_backend_cls()` 中添加映射
4. 如需 Patch，在 `patch/worker/` 中添加

#### **如何添加新的量化方法**
1. 在 `quantization/methods/` 中实现量化方法
2. 继承 vLLM 的量化基类
3. 在 `quantization/__init__.py` 中注册
4. 实现 NPU 特定的量化算子

#### **如何添加新的 Patch**
1. 确定 Patch 类型 (Platform 或 Worker)
2. 在对应目录创建 Patch 文件
3. 在 `patch/__init__.py` 中添加文档
4. 在 `utils.adapt_patch()` 中调用

---

## 7. Ascend 特定芯片支持

### 7.1 Ascend 950 (Atlas A5) 支持情况

#### **7.1.1 支持状态**

| 维度 | 状态 | 说明 |
|------|------|------|
| **芯片支持** | ✅ 已支持 | Atlas A5 (Ascend950) |
| **发布版本** | ✅ | [PR #7151](https://github.com/vllm-project/vllm-ascend/pull/7151) |
| **自定义算子** | ⚠️ 部分禁用 | 见下方详细列表 |

#### **7.1.2 A5DeviceAdaptor 专用实现**

```python
class A5DeviceAdaptor(BaseDeviceAdaptor):
    @classmethod
    def reshape_and_cache(cls, key, value, key_cache, value_cache, slot_mapping):
        # 使用 torch_npu 原生 API
        torch_npu.npu_scatter_pa_kv_cache(...)

    @staticmethod
    def npu_moe_init_routing(...):
        # 使用 V2 版本路由
        return torch_npu.npu_moe_init_routing_v2(...)
```

#### **7.1.3 MXFP 量化支持**

Ascend 950 是唯一支持 MXFP 量化格式的平台：

```python
# MXFP MoE 量化仅在 A5 上支持
if ascend_device_type != AscendDeviceType.A5:
    raise RuntimeError("MXFP MoE quantization is only supported on Ascend A5.")
```

**MXFP 量化方法**：
- **MXFP4**: 4-bit Microscaling Format
- **MXFP8**: 8-bit Microscaling Format
- **MXFP MoE**: MoE 专用 MXFP 量化

#### **7.1.4 专门的 C++ Kernel**

| 文件 | 内容 |
|------|------|
| `csrc/moe/moe_gating_top_k/op_host/moe_gating_top_k_tiling_arch35.cpp` | A5 专用 tiling |
| `csrc/moe/chunk_gated_delta_rule_fwd_h/op_kernel/gemm/kernel/gdn_fwd_h_kernel.hpp` | `Arch::Ascend950` |
| `csrc/moe/chunk_fwd_o/op_kernel/gemm/kernel/gdn_fwd_o_kernel.hpp` | `Arch::Ascend950` |

#### **7.1.5 当前限制**

**禁用的自定义算子**：

| 算子文件 | 功能描述 | 状态 |
|----------|----------|------|
| `mla_preprocess_kernel.cpp` | MLA 预处理算子 | ❌ 禁用 |
| `batch_matmul_transpose_kernel.cpp` | 批量矩阵转置 | ❌ 禁用 |

**限制原因**：

```python
# FIXME(linfeng): Currently custom op compilation and execution are partially available
# in ASCEND950 chip, we temporarily disable all custom ops.
# Please refer to https://github.com/vllm-project/vllm-ascend/issues/7157 for latest update.
if get_ascend_device_type() == AscendDeviceType.A5:
    _CUSTOM_OP_ENABLED = False
```

**限制原因详解**：
- 部分自定义算子未实现批处理不变性 (batch invariant)
- 使用 CANN 原生算子作为替代方案

#### **7.1.6 配置方式**

**环境变量设置**：
```bash
# 设置 SOC 版本（针对 Atlas A5）
export SOC_VERSION="<value starting with ascend950>"
```

**CMake 支持**：
```cmake
# CMakeLists.txt 中已包含 ascend950
set(ASCEND_ALL_COMPUTE_UNIT "ascend310p;ascend910b;ascend910_93;ascend950;kirinx90")
```

#### **7.1.7 进展跟踪**

| 资源 | 链接 |
|------|------|
| **官方 Issue** | [vllm-ascend#7157](https://github.com/vllm-project/vllm-ascend/issues/7157) |
| **PR 链接** | [vllm-ascend#7151](https://github.com/vllm-project/vllm-ascend/pull/7151) |

---

### 7.2 Ascend 310P 支持

vLLM-Ascend 还提供专门的 310P 实现（见 `_310p/` 目录）：

| 模块 | 文件数 | 说明 |
|------|--------|------|
| `model_runner_310p.py` | 1 | 310P 专用 ModelRunner |
| `worker_310p.py` | 1 | 310P 专用 Worker |
| `attention/` | 2 | 310P 专用 Attention Backend |
| `quantization/` | 2 | 310P 专用量化方法 |
| `ops/` | 3 | 310P 专用算子 |

**310P 特点**：
- 专用优化推理场景
- 特定的算子实现
- 独立的 Block Table 实现

---

## 8. 文件路径映射

### 8.1 关键模块映射

| NVIDIA vLLM | vLLM-Ascend | 功能描述 |
|-------------|-------------|----------|
| `vllm/platforms/cuda.py` | `vllm_ascend/platform.py` | 平台抽象 |
| `vllm/v1/worker/gpu_worker.py` | `vllm_ascend/worker/worker.py` | Worker 实现 |
| `vllm/v1/worker/gpu_model_runner.py` | `vllm_ascend/worker/model_runner_v1.py` | Model Runner |
| `vllm/v1/attention/backends/flashinfer.py` | `vllm_ascend/attention/attention_v1.py` | Attention Backend |
| `vllm/v1/worker/gpu/cudagraph_utils.py` | `vllm_ascend/compilation/acl_graph.py` | Graph 捕获 |
| `vllm/model_executor/layers/fused_moe/` | `vllm_ascend/ops/fused_moe/` | MoE 实现 |
| `vllm/distributed/parallel_state.py` | `vllm_ascend/distributed/parallel_state.py` | 并行状态 |
| `vllm/distributed/device_communicators/` | `vllm_ascend/distributed/device_communicators/` | 设备通信器 |

### 8.2 API 替换映射

| CUDA API | Ascend API | 说明 |
|----------|-----------|------|
| `torch.cuda` | `torch_npu` | 设备操作 |
| `dist.init_process_group(backend='nccl')` | `PyHcclCommunicator` | 集合通信 |
| `torch.cuda.memory_allocated()` | `torch_npu.npu.memory_allocated()` | 内存查询 |
| `torch.cuda.synchronize()` | `torch_npu.synchronize()` | 设备同步 |
| `CUDA Graph` | `ACL Graph` | Graph 捕获 |

---

**文档更新历史**:
- v3.0 (2026-06-20): 合并 vllm_ascend_nvidia_comparison.md 内容，增加 Ascend 950 支持、310P 支持、文件路径映射
- v2.0 (2026-06-19): 基于最新源码全面更新，增加详细统计和对比
- v1.0 (2026-06-19): 初始版本

**维护者**: vLLM-Ascend 项目团队
