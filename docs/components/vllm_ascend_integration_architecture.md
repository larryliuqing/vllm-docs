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

### 3.1 分阶段 Patch

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