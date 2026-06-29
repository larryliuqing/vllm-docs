# DeepSeek V4 模型实现详解与跨平台差异分析

> 本文档详细介绍 DeepSeek V4 模型在 vLLM（CUDA/AMD/XPU）和 vLLM-Ascend（NPU）两个框架中的代码实现，以及两者在架构设计、算子实现和硬件适配上的差异。

---

## 一、架构总览

### 1.1 代码规模

| 指标 | vLLM（三平台） | vLLM-Ascend |
|------|---------------|-------------|
| 模型文件数 | **36** 个（含 nvidia/amd/xpu 子目录） | **2** 个（ds_v4.py + ds_v4_mtp.py） |
| 总代码行 | **~15,500** 行 | **~1,860** 行 |
| 核心模型层 | ~1,340 行（nvidia/model.py） | ~1,354 行（ds_v4.py） |
| 附加算子层 | 8 个 shared ops + 5 个 nvidia ops + 2 个 xpu ops | 2 个 ascendor ops（dsa.py + rope_dsv4.py） |
| 量化配置 | quant_config.py（160 行） | 复用 vLLM FP8 配置 |

**核心差异**：vLLM 对每个硬件平台都有独立的 ops 实现，文件分散；vLLM-Ascend 以单体文件 + 少量自定义算子为主，大量复用 vLLM 的共享层。

### 1.2 模型入口

**vLLM（vllm/models/deepseek_v4/__init__.py）**：按平台分发

```python
# 根据 current_platform 分发到不同子目录
if current_platform.is_rocm():
    from .amd.model import DeepseekV4ForCausalLM
elif current_platform.is_xpu():
    from .xpu.model import DeepseekV4ForCausalLM
else:
    from .nvidia.model import DeepseekV4ForCausalLM
```

**vLLM-Ascend（vllm_ascend/models/__init__.py）**：统一注册

```python
ModelRegistry.register_model("DeepseekV4ForCausalLM",
    "vllm_ascend.models.deepseek_v4:AscendDeepseekV4ForCausalLM")
```

---

## 二、核心架构设计

### 2.1 模型整体架构

```
DeepSeek V4 ForCausalLM
├── DeepseekV4Model（主干网络）
│   ├── VocabParallelEmbedding（词嵌入）
│   ├── DeepseekV4DecoderLayer × N（解码器层）
│   │   ├── RMSNorm（LayerNorm）
│   │   ├── 稀疏 MLA 注意力（核心差异所在）
│   │   │   ├── vLLM: DeepseekV4FlashMLAAttention / FlashInfer
│   │   │   └── vLLM-Ascend: AscendDeepseekSparseAttention (DSA)
│   │   ├── Hybrid Chunk (HC) 残差融合
│   │   └── DeepseekV4MoE（混合专家层）
│   │       ├── Shared Experts（共享专家）
│   │       └── Routed Experts（路由专家，含 MegaMoE）
│   └── RMSNorm
├── ParallelLMHead（lm_head）
└── LogitsProcessor
```

### 2.2 支持的接口

| 接口 | vLLM（NVIDIA） | vLLM（AMD） | vLLM-Ascend |
|------|---------------|------------|-------------|
| `SupportsPP` | ✅ | ✅ | ✅ |
| `SupportsEagle` | ❌ | ❌ | ✅ |
| `SupportsLoRA` | ❌ | ❌ | ✅ |
| `SupportsEPLB` | ✅（通过 `MixtureOfExperts`） | ❌ | ✅（通过 `DeepseekV2MixtureOfExperts`） |

---

## 三、核心实现对比

### 3.1 注意力机制（MLA）

这是两个框架差异最大的部分。

| 维度 | vLLM（NVIDIA） | vLLM-Ascend |
|------|---------------|-------------|
| **实现文件** | `nvidia/flashmla.py` + `sparse_mla.py`（~800 行） | `ops/dsa.py`（272 行） |
| **后端选择** | 运行时可选 FlashMLA / FlashInfer 后端 | 编译时固定的 DSA（Ascend Deepseek Sparse Attention） |
| **类名** | `DeepseekV4FlashMLAAttention` | `AscendDeepseekSparseAttention` |
| **继承链** | ← `DeepseekV4Attention`（ABC，在 `attention.py`） | ← `MultiHeadLatentAttentionWrapper` |
| **CUDA/NPU 算子** | FlashMLA kernel（NVIDIA 闭源）或 FlashInfer TRTLLM-gen | `torch_npu` 原生算子 + DSA 模块 |
| **Indexer Cache** | `DeepseekV4IndexerCache`（在 `attention.py`） | 直接 import vLLM 的 IndexerCache |
| **稀疏注意力** | `_select_dsv4_attn_cls()` 决定 | `AscendDeepseekSparseAttention` 内部集成 |

**vLLM 的注意力选择逻辑**：
```python
def _select_dsv4_attn_cls(vllm_config) -> type[DeepseekV4Attention]:
    if backend == AttentionBackendEnum.FLASHINFER_MLA_SPARSE_DSV4:
        return DeepseekV4FlashInferMLAAttention  # FlashInfer TRTLLM-gen 路径
    return DeepseekV4FlashMLAAttention           # FlashMLA 路径（默认）
```

**vLLM-Ascend 的集成方式**：
```python
# 在 DecoderLayer 中构造 DSA 注意力
dsa_modules = DSAModules(
    rope_emb=self.rope_emb,
    indexer_cache=self.indexer_cache,
    ...
)
self.dsa_attn = AscendDeepseekSparseAttention(
    config=config,
    dsa_modules=dsa_modules,
)
# forward 时直接调用
hidden_states = self.dsa_attn(hidden_states=hidden_states, positions=positions, ...)
```

### 3.2 DecoderLayer 前向传播

**vLLM（NVIDIA）**：使用 `tilelang` 实现的 `mhc_pre_tilelang` / `mhc_fused_post_pre_tilelang`

```python
def forward(self, x, positions, input_ids, post_mix, res_mix, residual):
    if residual is None:
        # 第一层：独立 mhc_pre
        post_mix, res_mix, x = mhc_pre_tilelang(x, self.hc_attn_fn, ...)
    else:
        # 后续层：融合 mhc_fused_post_pre（融合前一层 post 和当前层 pre）
        residual, post_mix, res_mix, x = mhc_fused_post_pre_tilelang(
            x, residual, self.hc_attn_fn, self.hc_post_alpha, ...)
    
    # 注意：attn_norm 已融合进 mhc_*_tilelang 中
    hidden_states = self.attn(x, positions, ...)
    
    # FFN 同样复合 mhc_fused_post_pre
    residual, post_mix, res_mix, x = mhc_fused_post_pre_tilelang(
        hidden_states, residual, self.hc_ffn_fn, self.hc_post_alpha, ...)
    hidden_states = self.ffn(x, ...)
```

**vLLM-Ascend**：使用 `torch.ops._C_ascend.npu_hc_pre` / `npu_hc_post`

```python
def forward(self, positions, hidden_states, residual):
    residual = hidden_states.clone()
    # HC Pre（Ascend 原生算子）
    hidden_states, post, comb = self.hc_pre(hidden_states, self.hc_attn_fn, ...)
    hidden_states = self.input_layernorm(hidden_states)
    # DSA 注意力
    hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states, ...)
    # HC Post
    hidden_states = self.hc_post(hidden_states, residual, post, comb)
    
    residual = hidden_states.clone()
    hidden_states, post, comb = self.hc_pre(hidden_states, self.hc_ffn_fn, ...)
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = self.hc_post(hidden_states, residual, post, comb)
```

**关键差异**：

| 特征 | vLLM（NVIDIA） | vLLM-Ascend |
|------|---------------|-------------|
| HC 算子 | `mhc_pre_tilelang` / `mhc_fused_post_pre_tilelang`（tilelang 编译） | `npu_hc_pre` / `npu_hc_post`（Ascend 自定义算子） |
| Norm 融合 | attn_norm 融合在 mhc_pre 中 | 独立的 `input_layernorm` / `post_attention_layernorm` |
| 残差管理 | 三态 `(residual, post_mix, res_mix)` | 两态 `(residual, post + comb)` |
| 编译装饰器 | 无 | `@support_torch_compile` |
| 签名参数 | `(x, positions, input_ids, post_mix, res_mix, residual)` + `llama_4_scaling` | `(positions, hidden_states, residual, llama_4_scaling)` |

### 3.3 MoE（混合专家层）

| 维度 | vLLM（NVIDIA） | vLLM-Ascend |
|------|---------------|-------------|
| MoE 实现 | `FusedMoE` + `prepare_megamoe_inputs` | `FusedMoE` + Ascend 的 `mix_placement` |
| Shared Experts | 标准实现 | 支持 `mix_placement`（shared experts 放到不同设备） |
| MegaMoE | `DeepseekV4MegaMoEExperts` | 无独立类（复用 MoE） |
| 专家量化 | FP4 / FP8 Expert（`quant_config.py`） | 通过 `get_ascend_config()` 配置 |

**Ascend 独有的 `mix_placement`**：
```python
self.is_fusion_moe_shared_experts_enabled = getattr(
    get_ascend_config(), "mix_placement", False
)
if config.n_shared_experts is None or self.is_fusion_moe_shared_experts_enabled:
    self.shared_experts = None  # shared experts 由 FusedMoE 统一管理
```

当 `mix_placement=True` 时，Ascend 将 shared experts 和 routed experts 在硬件层面混合放置，可以更灵活地利用昇腾集群的内存带宽。NVIDIA 版本则是标准的 TP MoE，shared 和 routed 专家在同一设备上。

### 3.4 MTP（Multi-Token Prediction）

| 维度 | vLLM（NVIDIA） | vLLM-Ascend |
|------|---------------|-------------|
| 实现文件 | `nvidia/mtp.py` + `model_executor/models/deepseek_mtp.py` | `ds_v4_mtp.py` |
| 类名 | `DeepSeekV4MTP` | `DeepSeekV4MTP` |
| 注册名 | —（通过 model_executor 注册） | `DeepSeekV4MTPModel` |
| SharedHead | 使用 `ParallelLMHead` | 自定义 `SharedHead`（含 RMSNorm） |
| Decoder 复用 | 独立的 MTP decoder 层 | 复用 `DeepseekV2DecoderLayer` + `DeepseekV4MoE` |
| 权重加载 | 标准 `load_weights` | 自定义 `_remap_weight_name` |

### 3.5 RoPE（旋转位置编码）

| 维度 | vLLM | vLLM-Ascend |
|------|------|-------------|
| 文件 | `common/rope.py`（36 行） | `ops/rope_dsv4.py`（237 行） |
| 功能 | 辅助函数 `build_deepseek_v4_rope()` | 完整 `ComplexExpRotaryEmbedding` 类 |
| 缓存 | 无 | `RopeDataProxy` 支持 cos/sin 缓存 |
| 与 DSA 集成 | 无 | `get_cos_and_sin_dsa()` 专为 DSA 优化 |

Ascend 的 RoPE 实现远比 vLLM 的通用版本复杂，因为：
1. 昇腾 NPU 的复数运算需要特定优化
2. 需要为 DSA 注意力提供定制的 cos/sin 值
3. 通过 `RopeDataProxy` 实现了 cos/sin 缓存以降低延迟

### 3.6 量化配置

| 维度 | vLLM | vLLM-Ascend |
|------|------|-------------|
| 类 | `DeepseekV4FP8Config`（继承 `Fp8Config`） | 无独立 DSV4 量化配置 |
| 专家类型 | 支持 FP4 / FP8 专家 | 复用 vLLM 的 FP8 量化 |
| 权重映射 | `_make_deepseek_v4_weights_mapper(expert_dtype)` | 标准 default_weight_loader |

**vLLM 的量化路由**：
```python
# 在 config.py 中自动重写 quant_method
if model_type == "deepseek_v4":
    quant_config["quant_method"] = "deepseek_v4_fp8"
```
vLLM-Ascend 没有这个路由——DS V4 在昇腾上直接使用 vLLM 的 FP8 量化方法。

### 3.7 DSA 上下文并行

**vLLM-Ascend 独有**（vLLM 无对应实现）：
```python
self.enable_dsa_cp = enable_dsa_cp()
attn_sink_heads = self.n_heads if self.enable_dsa_cp else self.n_local_heads
wq_b_cls = ReplicatedLinear if self.enable_dsa_cp else ColumnParallelLinear
```

当启用 DSA Context Parallel 时：
- 注意力头不再按 TP 切分（`n_heads` 而非 `n_local_heads`）
- Q 投影变为复制而非切分（`ReplicatedLinear` 而非 `ColumnParallelLinear`）

---

## 四、详细文件映射

### 4.1 vLLM DS V4 文件结构

| 目录/文件 | 行数 | 作用 |
|----------|------|------|
| **`__init__.py`** | 31 | 入口，按平台分发 |
| **`quant_config.py`** | 160 | FP4/FP8 专家量化配置 |
| **`attention.py`** | 800 | MLA 注意力基类 + IndexerCache |
| **`compressor.py`** | 399 | KV cache 压缩器（CompressorStateCache） |
| **`sparse_mla.py`** | 416 | 稀疏 MLA AttentionBackend |
| **`common/rope.py`** | 36 | RoPE 辅助函数 |
| **`common/ops/`**（8 个文件） | ~2,700 | 共享算子（cache_utils, fused_compress_quant, fused_indexer_q 等） |
| **`nvidia/model.py`** | ~1,340 | NVIDIA 模型主文件（MLP, MoE, DecoderLayer, ForCausalLM） |
| **`nvidia/mtp.py`** | ~300 | NVIDIA MTP 实现 |
| **`nvidia/ops/`**（6 个文件） | ~3,400 | NVIDIA 专用算子（cutedsl, flashmla, flashinfer_sparse） |
| **`amd/model.py`** | 818 | AMD ROCm 模型实现 |
| **`xpu/model.py`** | ~1,370 | Intel XPU 模型实现 |
| **`model_executor/models/ds_mtp.py`** | 516 | MTP 模型注册与权重加载 |

### 4.2 vLLM-Ascend DS V4 文件结构

| 文件 | 行数 | 作用 |
|------|------|------|
| **`models/ds_v4.py`** | 1,354 | 完整的模型实现（MLP, MoE, DecoderLayer, ForCausalLM） |
| **`models/ds_v4_mtp.py`** | 506 | MTP 实现（复用主模型层） |
| **`ops/dsa.py`** | 272 | Ascend Deepseek Sparse Attention |
| **`ops/rope_dsv4.py`** | 237 | DS V4 专用的 RoPE 实现 |

---

## 五、总结

| 维度 | vLLM 策略 | vLLM-Ascend 策略 |
|------|-----------|-----------------|
| **注意力实现** | FlashMLA / FlashInfer TRTLLM-gen（CUDA 生态） | AscendDeepseekSparseAttention（NPU 原生 DSA） |
| **Hybrid Chunk** | `mhc_pre_tilelang`（tilelang 编译） | `npu_hc_pre` / `npu_hc_post`（Ascend 自定义算子） |
| **MoE 并行** | 标准 TP MoE | 支持 mix_placement 混合放置 |
| **RoPE** | 36 行辅助函数 | 237 行完整类 + 缓存 + DSA 集成 |
| **量化** | DeepseekV4FP8Config（FP4/FP8 专家） | 复用 vLLM 通用 FP8 |
| **接口支持** | SupportsPP | SupportsPP + SupportsEagle + SupportsLoRA |
| **代码风格** | 硬件隔离子目录（36 文件） | 单体文件 + 少量自定义 ops（2 文件 + 2 ops） |

**一句话总结**：vLLM 的 DS V4 适配是**全栈自研 + CUDA 生态依赖**模式（FlashMLA、FlashInfer、cutedsl、tilelang），vLLM-Ascend 是**复用共享层 + 替换关键路径为 NPU 原生算子**模式——复用 vLLM 的 IndexerCache、Compressor、FusedMoE，替换注意力（DSA）、RoPE（ComplexExpRotaryEmbedding）、HC（npu_hc_pre/post）为昇腾 NPU 实现。两者都适配了 DS V4 的核心架构（MLA + MoE + MTP + HC），只是底层硬件算子不同。

---

**文档版本**: v1.0  
**创建时间**: 2026-06-27  
**基于源码**: `vllm/vllm/models/deepseek_v4/` + `vllm-ascend/vllm_ascend/models/ds_v4.py`