# vLLM-Ascend 集成架构与 Patch 机制详解

> 本文档深入分析 vLLM-Ascend 项目如何通过插件化架构和 Monkey Patch 机制对接上游 vLLM 项目，实现对 Ascend NPU 的全面支持。

---

## 1. 概述

vLLM-Ascend 是 vLLM 在华为昇腾 NPU 上的硬件适配层。其核心设计原则是：

- **最小侵入**：尽量不改动上游 vLLM 源码
- **插件化**：通过 vLLM 的插件机制注册为 NPU 后端
- **Monkey Patch**：在运行时替换上游模块的函数/类
- **分层适配**：平台层 + Worker 层 + 算子层 + 编译层

```
┌─────────────────────────────────────────────────────────────┐
│                     vLLM Plugin System                      │
├─────────────────────────────────────────────────────────────┤
│  setup.py entry_points                                      │
│  ├─ vllm.platform_plugins → NPUPlatform                     │
│  └─ vllm.general_plugins  → __init__.py 注册函数            │
├─────────────────────────────────────────────────────────────┤
│                     NPUPlatform (核心入口)                    │
│  ├─ pre_register_and_update() → 平台 Patch                   │
│  ├─ check_and_update_config() → Worker/编译配置             │
│  ├─ get_attn_backend_cls()   → Attention 后端路由           │
│  └─ get_compile_backend()    → AscendCompiler 图融合        │
├─────────────────────────────────────────────────────────────┤
│                     Worker (运行时)                          │
│  └─ __init__() → Worker Patch                               │
├─────────────────────────────────────────────────────────────┤
│              自定义组件（Attention / Ops / 量化等）           │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. 插件化注册机制

### 2.1 Entry Points

在 `setup.py` 中注册的插件入口：

```python
# setup.py
entry_points={
    "vllm.platform_plugins": [
        "ascend = vllm_ascend:register",        # → NPUPlatform
    ],
    "vllm.general_plugins": [
        "ascend_kv_connector = vllm_ascend:register_connector",
        "ascend_model_loader = vllm_ascend:register_model_loader",
        "ascend_service_profiling = vllm_ascend:register_service_profiling",
        "ascend_model = vllm_ascend:register_model",
    ],
}
```

### 2.2 注册入口

```python
# vllm_ascend/__init__.py

def register():
    """注册NPUPlatform为vLLM的硬件后端"""
    return "vllm_ascend.platform.NPUPlatform"

def register_connector():
    """注册KV传输连接器"""
    _ensure_global_patch()  # 先应用全局Patch
    ...

def register_model_loader():
    """注册自定义模型加载器（netloader, rfork）"""
    _ensure_global_patch()
    ...

def register_model():
    """注册Ascend专用模型架构"""
    from vllm_ascend.models import register_model
    register_model()

def _ensure_global_patch():
    """确保平台级Patch已应用"""
    from vllm_ascend.utils import adapt_patch
    adapt_patch(is_global_patch=True)
```

---

## 3. Patch 应用机制

### 3.1 核心原理

Monkey Patch 的原理是 Python 的**运行时属性替换**：将上游 vLLM 模块中的函数/类/方法替换为 vLLM-Ascend 的 NPU 兼容实现。替换发生在**模块导入时**（顶层代码），而非运行时显式调用。

```
Python 模块导入流程
    │
    ├─ import vllm_ascend.patch.platform
    │   └─ __init__.py 导入各 patch_*.py
    │       └─ patch_*.py 执行顶层代码：
    │           vllm.module.function = ascend_function
    │
    └─ 从此，所有对 vllm.module.function 的调用 → ascend_function
```

### 3.2 Patch 示例详解

#### 示例1：最简单的 Patch — 重定向 torch.accelerator → torch.npu

**上游 vLLM 的行为**：使用 `torch.accelerator.memory_stats()` 获取 GPU 内存统计。

**NPU 的问题**：`torch.accelerator` 只适配了 CUDA，对 NPU 返回空值。

**Patch 实现**：

```python
# vllm_ascend/patch/platform/patch_torch_accelerator.py

import torch

def patch_empty_cache() -> None:
    torch.npu.empty_cache()

# 逐个替换 torch.accelerator 的 API 为 torch.npu 等价函数
torch.accelerator.empty_cache = patch_empty_cache
torch.accelerator.memory_stats = torch.npu.memory_stats
torch.accelerator.memory_reserved = torch.npu.memory_reserved
torch.accelerator.reset_peak_memory_stats = torch.npu.reset_peak_memory_stats
```

**Patch 结果**：上游 vLLM 调用 `torch.accelerator.memory_stats()` 时，实际执行的是 `torch.npu.memory_stats()`，对 NPU 完全透明。

---

#### 示例2：替换方法 — Qwen3-VL 模型前向优化

**上游 vLLM 的行为**：Qwen3-VL 的 Attention 前向在 GPU 上分步执行（QKV 投影 → split → QK-Norm → RoPE → Attention）。

**NPU 的问题**：分步执行导致多次 kernel launch，在 NPU 上性能差。

**Patch 实现**：

```python
# vllm_ascend/patch/worker/patch_qwen3vl.py

def forward_with_split_qkv_rmsnorm_mrope(
    self, positions: torch.Tensor, hidden_states: torch.Tensor
):
    # 1. QKV 投影（与原版一致）
    qkv, _ = self.qkv_proj(hidden_states)

    if isinstance(self.rotary_emb, AscendMRotaryEmbedding):
        # 2. NPU 融合算子：QKV split + QK-Norm + RoPE 一步完成
        q, k, v, _ = torch.ops.vllm.triton_split_qkv_rmsnorm_mrope(
            qkv=qkv,
            q_weight=self.q_norm.weight,
            k_weight=self.k_norm.weight,
            cos_sin=cos_sin,
            num_q_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_dim,
            eps=self.q_norm.variance_epsilon,
            mrope_section=self.rotary_emb.mrope_section,
            is_interleaved=self.rotary_emb.mrope_interleaved,
            rope_dim=self.rotary_emb.rotary_dim,
        )
    else:
        # 回退到分步执行
        q, k, v = qkv.split(...)
        q_by_head = self.q_norm(q_by_head)
        k_by_head = self.k_norm(k_by_head)
        q, k = self.rotary_emb(positions, q, k)

    # 3. Attention 计算 + 输出投影（与原版一致）
    attn_output = self.attn(q, k, v)
    output, _ = self.o_proj(attn_output)
    return output

# 用 NPU 优化版本替换上游的 forward 方法
Qwen3Attention.forward = forward_with_split_qkv_rmsnorm_mrope
Qwen3MoeAttention.forward = forward_with_split_qkv_rmsnorm_mrope
```

**Patch 结果**：Qwen3-VL 的 Attention 前向被替换为融合版本，在 NPU 上减少 3 次 kernel launch，QK-Norm 和 RoPE 合并到 Triton 算子中一次完成。

---

#### 示例3：覆盖函数 — KV Cache Block Size 适配

**上游 vLLM 的行为**：`resolve_kv_cache_block_sizes()` 在多 KV cache group + CP（context parallelism）时返回错误。

**NPU 的问题**：上游代码假设多个 block size 不能与 CP 共存（这是 CUDA 的限制），但 Ascend 支持这种组合。

**Patch 实现**：

```python
# vllm_ascend/patch/platform/patch_kv_cache_utils.py

import vllm.v1.core.kv_cache_utils

# 保存原始函数引用
_orig_resolve = vllm.v1.core.kv_cache_utils.resolve_kv_cache_block_sizes

def _ascend_resolve(kv_cache_config, vllm_config):
    cache_config = vllm_config.cache_config
    dcp = vllm_config.parallel_config.decode_context_parallel_size
    pcp = vllm_config.parallel_config.prefill_context_parallel_size
    groups = kv_cache_config.kv_cache_groups

    if len(groups) <= 1:
        # 单 group → 走原始逻辑
        bs = cache_config.block_size * dcp * pcp
        return bs, bs

    if dcp != 1 or pcp != 1:
        # Ascend 支持多 group + CP：
        # 计算所有 group block size 的 LCM × CP 因子
        group_block_sizes = [g.kv_cache_spec.block_size for g in groups]
        scheduler_block_size = math.lcm(*group_block_sizes) * dcp * pcp
        return scheduler_block_size, scheduler_block_size

    return _orig_resolve(kv_cache_config, vllm_config)

# 替换上游函数
vllm.v1.core.kv_cache_utils.resolve_kv_cache_block_sizes = _ascend_resolve

# 同时替换 engine/core.py 中直接 import 的引用
import vllm.v1.engine.core
vllm.v1.engine.core.resolve_kv_cache_block_sizes = _ascend_resolve
```

**Patch 结果**：DS V4 在 8 卡 CP 场景下可以正确计算 KV Cache block size，不会触发上游的 CP 限制断言。

---

#### 示例4：覆盖类方法 — Mamba 配置适配

**上游 vLLM 的行为**：`HybridAttentionMambaModelConfig.verify_and_update_config()` 设置默认 block size。

**NPU 的问题**：Ascend 硬件要求 cache tensor 连续，需要 attention block size 对齐到 kernel block size（128）的倍数。

**Patch 实现**：

```python
# vllm_ascend/patch/platform/patch_mamba_config.py

import vllm.model_executor.models.config

@classmethod
def verify_and_update_config(cls, vllm_config):
    # 1. 先执行原始逻辑
    MambaModelConfig.verify_and_update_config(vllm_config)

    # 2. 计算 SSM block 和 attention block 的对齐
    kernel_block_size = 128
    ssm_block_page_size = max(mamba_sizes)

    # 计算 attention block size 使其 page size ≥ ssm page size
    attn_block_size = kernel_block_size * cdiv(
        ssm_block_page_size,
        kernel_block_size * attn_single_token_k_page_size,
    )

    # 3. 覆盖 block size 为对齐后的值
    if cache_config.block_size is None or cache_config.block_size < attn_block_size:
        cache_config.block_size = attn_block_size

    # 4. padding mamba page 使其与 attention page 相等
    cache_config.mamba_page_size_padded = attn_page_size + conv_block_page_size

# 用 Ascend 版本替换类方法
vllm.model_executor.models.config.HybridAttentionMambaModelConfig.verify_and_update_config = \
    verify_and_update_config
```

**Patch 结果**：Mamba+Attention 混合模型在 NPU 上运行时，block size 自动对齐到 128 的倍数，确保 cache tensor 连续。

---

### 3.3 分阶段 Patch

Patch 分为两个阶段应用：

```
启动流程
    │
    ├─ 阶段1：平台Patch（is_global_patch=True）
    │   │
    │   ├─ 触发时机：
    │   │   ├─ NPUPlatform.pre_register_and_update()  ← 在线服务
    │   │   └─ _ensure_global_patch()                ← EngineCore子进程
    │   │
    │   └─ 覆盖范围：全局配置、KV Cache、分布式通信、Tool Call等
    │
    └─ 阶段2：Worker Patch（is_global_patch=False）
        │
        ├─ 触发时机：NPUWorker.__init__()
        │
        └─ 覆盖范围：算子替换、权重加载、模型前向、Triton替换
```

### 3.2 Patch 应用代码

```python
# vllm_ascend/utils.py
def adapt_patch(is_global_patch: bool = False):
    if is_global_patch:
        from vllm_ascend.patch import platform  # noqa: F401
    else:
        from vllm_ascend.patch import worker  # noqa: F401
```

每个 patch 模块在**导入时**执行 monkey patch（顶层代码）：

```python
# vllm_ascend/patch/worker/patch_qwen3vl.py  (示例)

import vllm.model_executor.models.qwen3_vl as qwen3vl_module
from vllm_ascend.ops.mm_encoder_attention import npu_mm_encoder_attention

def apply_patch():
    """导入时执行"""
    # 替换多模态编码器Attention
    qwen3vl_module.Qwen3VLForConditionalGeneration._get_deepstack_input_embeds = \
        npu_get_deepstack_input_embeds

    # 支持Flash Comm v1
    qwen3vl_module.Qwen3VLVisionEncoder.attention = npu_mm_encoder_attention
```

---

## 4. 平台级 Patch 详解

平台级 Patch 覆盖 vLLM 的**全局行为**，在引擎初始化之前应用。

### 4.1 完整 Patch 清单

| Patch 文件 | 作用 | 替换的 vLLM 模块 |
|-----------|------|-----------------|
| `patch_camem_allocator.py` | 使休眠分配器检查通过（CaMem → CuMem） | `vllm.config.model.is_cumem_allocator_available` |
| `patch_distributed.py` | 310P 张量对齐 | `torch.distributed.all_reduce`, `broadcast` |
| `patch_kv_cache_interface.py` | MLA Spec 扩展 DSA 属性 | `vllm.v1.kv_cache_interface.MLAAttentionSpec` |
| `patch_kv_cache_utils.py` | 混合 KV Cache + 上下文并行 | `resolve_kv_cache_block_sizes` |
| `patch_mla_prefill_backend.py` | 注册空 MLAPrefillBackend 防崩溃 | `get_mla_prefill_backend()` |
| `patch_mamba_config.py` | Block Size 128（Ascend 不支持 16） | `HybridAttentionMambaModelConfig.verify_and_update_config` |
| `patch_minimax_m2_config.py` | 禁用 fp8 量化、设置 HCCL 模式 | `ModelConfig`, `SpeculativeConfig` |
| `patch_balance_schedule.py` | 平衡调度（分离 prefill/decode） | `EngineCoreProc.run_engine_core`, `Scheduler` |
| `patch_profiling_chunk.py` | 动态 chunk size 分析 | `EngineCore.__init__` |
| `patch_torch_accelerator.py` | NPU 内存统计重定向 | `torch.accelerator.*` → `torch.npu.*` |
| `patch_tool_choice_none_content.py` | Tool Call content=None 处理 | `OpenAIServing`, `DelegatingParser` |

**Tool Call 系列 Patch：**

| Patch 文件 | 作用 |
|-----------|------|
| `patch_deepseek_v4_tool_call_parser.py` | DeepSeek V4 增量参数流式输出 |
| `patch_deepseek_v4_thinking.py` | 推理 effort 处理 |
| `patch_minimax_m2_tool_call_parser.py` | MiniMax M2 增量参数流式输出 |
| `patch_glm_tool_call_streaming.py` | GLM tool call 流式 delta 修复 |
| `patch_glm47_tool_call_parser.py` | GLM47 零参数工具调用解析 |

### 4.2 条件 Patch

| 环境变量 | Patch 文件 | 作用 |
|---------|-----------|------|
| `DYNAMIC_EPLB` / `EXPERT_MAP_RECORD` | `patch_multiproc_executor.py` | 多进程执行器 daemon=False |
| `VLLM_ASCEND_APPLY_DSV4_PATCH=1` | `patch_kv_cache_coordinator.py` | KV Cache 协调器替换 |
| `VLLM_ASCEND_APPLY_DSV4_PATCH=1` | `patch_speculative_config.py` | 投机解码配置 |

---

## 5. Worker 级 Patch 详解

Worker 级 Patch 在**每个 NPU Worker 初始化时**应用，替换计算密集型操作。

### 5.1 算子替换

| Patch 文件 | 替换内容 | 替换原因 |
|-----------|---------|---------|
| `patch_triton.py` | Mamba、FLA、Gumbel Sample 等 Triton 算子 | Ascend-optimized Triton |
| `patch_npugraph_ex_triton.py` | npugraph_ex ValuePack 处理 | NPU 图捕获兼容 |
| `patch_v2/patch_triton.py` | v2 Logprob、Penalties、Gumbel | v2 版本 NPU 优化 |

### 5.2 模型前向替换

| Patch 文件 | 替换的模型 | 替换内容 |
|-----------|-----------|---------|
| `patch_minimax_m2.py` | MiniMax M2 | MoE all-reduce、fp8 反量化、Eagle3 |
| `patch_minimax_m2_linear_attn.py` | MiniMax M2 | 线性 Attention RMSNorm、kv-head 复制 |
| `patch_qwen3vl.py` | Qwen3-VL | 多模态编码器 Attention、Flash Comm v1 |
| `patch_qwen3_5.py` | Qwen3.5 | Gated Delta Net float32 状态 |
| `patch_qwen3_dflash.py` | Qwen3 DFlash | 不支持算子的替换 |
| `patch_deepseek_compressor.py` | DeepSeek V4 | Compressor/Indexer Cache |
| `patch_deepseek_mtp.py` | DeepSeek V4 MTP | MTP 层权重加载 |
| `patch_kimi_k25.py` | Kimi K2.5 | CPU interpolate（NPU 不支持） |
| `patch_gdn_attn.py` | GDN Attention | 预构建 varlen chunk metadata |
| `patch_gqa_c8.py` | Qwen3 C8 量化 | KV Cache scale/offset 拦截 |
| `patch_qwen3_next_mtp.py` | Qwen3 Next MTP | 跳过 NPU 异常 |

### 5.3 权重加载适配

| Patch 文件 | 作用 |
|-----------|------|
| `patch_weight_utils.py` | DS V2 C8 量化 KV scale 映射 |
| `patch_draft_quarot.py` | Eagle3 drafter 量化应用 |

### 5.4 采样器替换

| Patch 文件 | 替换内容 |
|-----------|---------|
| `patch_rejection_sampler.py` | top_k_top_p、expand_batch、rejection_sample 的 NPU Triton 内核 |

### 5.5 v2 版本 Patch

| Patch 文件 | 作用 |
|-----------|------|
| `patch_v2/patch_uva.py` | 虚拟 UVA Buffer（NPU 不支持 UVA） |
| `patch_v2/patch_input_batch.py` | AscendInputBatch 替换 |
| `patch_v2/patch_block_table.py` | int32 slot mapping |
| `patch_v2/patch_model_state.py` | AscendModelState 替换 |
| `patch_v2/patch_attn_utils.py` | Ascend Attention metadata |

---

## 6. NPUPlatform 核心类

`NPUPlatform` 是 vLLM 与 Ascend NPU 之间的桥梁，继承自 vLLM 的 `Platform` 基类。

```python
# vllm_ascend/platform.py

class NPUPlatform(Platform):

    def pre_register_and_update(self):
        """初始化前置步骤（引擎初始化前调用）"""
        # 1. 应用平台级 Patch
        adapt_patch(is_global_patch=True)
        # 2. 注册量化方法
        # 3. 导入 Ascend 量化配置
        # 4. 配置废弃日志

    def check_and_update_config(self):
        """配置检查和更新（解析配置后调用）"""
        # 1. 自动检测量化
        # 2. 初始化 Ascend 配置（从 additional_config）
        # 3. 选择 Worker 类（NPUWorker / NPUWorker310 / XliteWorker）
        # 4. 配置编译模式（CUDAGraph / ACL Graph / eager）
        # 5. 设置 Custom Ops、调度器覆写、内存分配器等

    def get_attn_backend_cls(self, ...):
        """Attention 后端路由"""
        backend_map = {
            (False, False, False): "AscendAttentionBackend",      # 标准
            (True,  False, False): "AscendMLABackend",            # MLA
            (True,  True,  False): "AscendSFABackend",            # Sparse
            (True,  False, True):  "AscendDSABackend",            # DSA
        }
        return backend_map[(use_mla, use_sparse, use_compress)]

    def get_compile_backend(self):
        """编译后端"""
        return AscendCompiler
```

---

## 7. Attention 后端架构

vLLM-Ascend 实现了 6 种 Attention 后端，通过 `NPUPlatform.get_attn_backend_cls()` 路由：

```
Attention Backend Selection
    │
    ├─ FLASH_ATTN + batch_invariant → AscendFABackend (fa3_v1.py)
    │
    ├─ use_mla=True, not sparse, not compress → AscendMLABackend (mla_v1.py)
    │
    ├─ use_mla=True + use_sparse=True → AscendSFABackend (sfa_v1.py)
    │
    ├─ use_mla=True + use_compress=True → AscendDSABackend (dsa_v1.py)
    │
    ├─ 默认 → AscendAttentionBackend (attention_v1.py)
    │
    └─ 310P → AscendAttentionBackend310 (_310p/)
```

每个后端继承自 `AttentionBackend` 基类，通过 `@register_backend` 注册：

```python
# vllm_ascend/attention/attention_v1.py
@register_backend(AttentionBackendEnum.CUSTOM, "ASCEND")
class AscendAttentionBackend(AttentionBackend):
    ...
```

---

## 8. 模型级覆盖

对于需要特殊硬件优化的模型，vLLM-Ascend 提供了完整的模型实现替换。

### 8.1 DS V4 覆盖

```python
# vllm_ascend/models/__init__.py
def register_model():
    ModelRegistry.register_model(
        "DeepseekV4ForCausalLM",
        "vllm_ascend.models.ds_v4",
        "AscendDeepseekV4ForCausalLM",
    )
    ModelRegistry.register_model(
        "DeepSeekV4MTPModel",
        "vllm_ascend.models.ds_v4_mtp",
        "DSV4MTP",
    )
```

DS V4 是唯一需要完整模型替换的架构（DSA 稀疏 Attention + Compressor），文件位于：

```
vllm_ascend/models/
├── ds_v4.py          # AscendDeepseekV4ForCausalLM（DSA 架构）
├── ds_v4_mtp.py      # DSV4MTP（Multi-Token Predictor）
└── layer/
    ├── compressor.py       # Compressor 层
    ├── dsa_attention.py    # DSA Attention
    └── ...
```

---

## 9. 编译与图融合

vLLM-Ascend 提供了自定义编译后端 `AscendCompiler`，实现 NPU 上的图融合优化：

```
AscendCompiler
    │
    ├─ GraphFusionPassManager
    │   ├─ sequence_parallelism.py        → FlashComm v1
    │   ├─ sequence_parallelism_moe.py    → MoE 序列并行
    │   ├─ allreduce_rmsnorm_fusion.py    → all-reduce + rms_norm 融合
    │   ├─ norm_quant_fusion.py           → 归一化 + 量化 融合
    │   ├─ qknorm_rope_fusion.py          → QK Norm + RoPE 融合
    │   ├─ allgather_chunk_noop.py        → 消除空 all-gather
    │   └─ ...
    │
    └─ ACLGraphWrapper                    → 分段图捕获
```

---

## 10. 环境变量控制

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `VLLM_ASCEND_APPLY_DSV4_PATCH` | 0 | 启用 DeepSeek V4 Patch |
| `VLLM_ASCEND_ENABLE_FLASHCOMM1` | 0 | FlashComm v1 序列并行 |
| `VLLM_ASCEND_ENABLE_FUSED_MC2` | 0 | Fused MC2 通信优化 |
| `VLLM_ASCEND_ENABLE_CONTEXT_PARALLEL` | 0 | 上下文并行 |
| `VLLM_ASCEND_ENABLE_MLAPO` | 1 | MLA 优化（DeepSeek W8A8） |
| `VLLM_ASCEND_ENABLE_NZ` | 1 | FRACTAL_NZ 格式转换 |
| `DYNAMIC_EPLB` | false | 动态专家负载均衡 |
| `SOC_VERSION` | auto | 芯片类型（ascend910b1 / ascend310p1） |

---

## 11. 架构总结

```
vLLM Plugin System
    │
    ├─ entry_points (setup.py)
    │   ├─ vllm.platform_plugins → NPUPlatform
    │   └─ vllm.general_plugins → 注册函数
    │
    ├─ NPUPlatform (platform.py)
    │   ├─ pre_register_and_update()
    │   │   └─ adapt_patch(is_global_patch=True)
    │   │       └─ patch/platform/ → 20+ Monkey Patches
    │   ├─ check_and_update_config()
    │   ├─ get_attn_backend_cls()
    │   └─ get_compile_backend() → AscendCompiler
    │
    ├─ Worker (worker.py)
    │   ├─ __init__()
    │   │   └─ adapt_patch(is_global_patch=False)
    │   │       └─ patch/worker/ → 25+ Monkey Patches
    │   └─ NPUModelRunner → 模型执行
    │
    └─ 自定义组件:
        ├─ attention/     (6 种后端)
        ├─ ops/           (融合 MoE、Triton、LayerNorm)
        ├─ compilation/   (图融合、ACL Graph)
        ├─ quantization/  (ModelSlim、压缩张量)
        ├─ models/        (DS V4 覆盖)
        ├─ spec_decode/   (Eagle、Medusa)
        ├─ sample/        (拒绝采样)
        ├─ lora/          (PunicaWrapperNPU)
        ├─ distributed/   (NPUCommunicator、KV Transfer)
        ├─ device_allocator/ (CaMem)
        └─ _310p/         (310P 专用)
```

---

## 12. 关键设计原则

1. **两阶段 Patch**：平台 Patch 修改全局行为，Worker Patch 修改计算逻辑——分阶段避免启动顺序依赖
2. **导入时执行**：Monkey Patch 在模块导入时自动生效，无需显式调用
3. **条件激活**：通过环境变量控制特定 Patch 是否启用（如 `VLLM_ASCEND_APPLY_DSV4_PATCH`）
4. **最小侵入**：尽量使用 Patch 而非 fork 上游代码，降低维护成本
5. **分层路由**：Attention 后端、编译后端、量化方法都通过 `NPUPlatform` 路由，可扩展性强
6. **渐进优化**：从通用 Patch → 模型特定 Patch → 完整模型替换，适配成本递增、收益递增

---

*文档版本：v1.0*
*创建日期：2026-06-27*
*基于 vLLM-Ascend 源码分析*