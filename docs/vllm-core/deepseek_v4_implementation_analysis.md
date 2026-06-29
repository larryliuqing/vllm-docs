# DS V4 模型实现详解与跨平台差异分析

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
    "vllm_ascend.models.ds_v4:AscendDeepseekV4ForCausalLM")
```

### 1.3 模型注册

**vLLM** 在 `vllm/model_executor/models/registry.py` 注册：
```python
"DeepseekV4ForCausalLM": ("vllm.models.ds_v4", "DeepseekV4ForCausalLM"),
```

**vLLM-Ascend** 通过 `vllm_ascend/models/__init__.py` 注册：
```python
ModelRegistry.register_model("DeepseekV4ForCausalLM", 
    "vllm_ascend.models.deepseek_v4:AscendDeepseekV4ForCausalLM")
```

vLLM 的配置验证在 `config.py` 中有特殊路由：
```python
# 自动将 fp8 量化为 ds_v4_fp8
if model_type == "ds_v4":
    quant_config["quant_method"] = "ds_v4_fp8"
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

## 三、vLLM 源码解读（NVIDIA 路径）

### 3.1 模块功能一览

| 模块 | 类/文件 | 行数 | 核心功能 |
|------|---------|------|---------|
| **MLP** | `DeepseekV4MLP` in `nvidia/model.py` | 70-118 | SwiGLU 前馈网络（MeragedColumnParallelLinear + RowParallelLinear + SiLU） |
| **MoE（门控 + 路由）** | `DeepseekV4MoE` in `nvidia/model.py` | 478-724 | 双后端路由（MegaMoE/FusedMoE）、shared experts、GateLinear sqrtsoftplus 评分 |
| **MoE（FP4 专家）** | `DeepseekV4MegaMoEExperts` in `nvidia/model.py` | 140-473 | SM100 专用 FP4 专家、DeepGEMM 权重变换、EPLB 负载均衡、对称缓冲区 |
| **注意力（基类）** | `DeepseekV4Attention` in `attention.py` | 98-618 | MLA 抽象基类、融合 W_Q_A+W_KV 投影、Q/KV 低秩分解、Indexer 管理 |
| **注意力（FlashMLA）** | `DeepseekV4FlashMLAAttention` in `nvidia/flashmla.py` | 33-320 | FlashMLA kernel 预填充/解码、FlashMLA FP8 KV cache、fp8_o_proj |
| **注意力（FlashInfer）** | `DeepseekV4FlashInferMLAAttention` in `nvidia/flashinfer_sparse.py` | ~80 | FlashInfer TRTLLM-gen 路径、bf16/FP8 KV cache |
| **DecoderLayer** | `DeepseekV4DecoderLayer` in `nvidia/model.py` | 740-885 | tilelang HC（mhc_pre/fused_post_pre）、三态残差流、attn_norm 融合 |
| **Model** | `DeepseekV4Model` in `nvidia/model.py` | 888-1057 | Embed → HC 扩展 → DecoderLayers → HC Head（tilelang）、MTP buffer |
| **ForCausalLM** | `DeepseekV4ForCausalLM` in `nvidia/model.py` | 1251-1337 | 顶层模型、权重映射器（_make_deepseek_v4_weights_mapper）、MegaMoE 权重转换 |
| **MTP** | `DSV4MTP` in `nvidia/mtp.py` | 260-380 | Multi-Token Prediction、e_proj/h_proj 分离、hc_head 延迟到 compute_logits |
| **Indexer** | `DeepseekV4Indexer` in `attention.py` | 662-800 | C4 稀疏索引、topk 选择、压缩分数计算 |
| **Sparse MLA Backend** | `DeepseekV4FlashMLABackend` in `sparse_mla.py` | 416 | AttentionBackend 接口、KV cache 格式定义、Metadata 构建 |
| **Compressor** | `CompressorStateCache` in `compressor.py` | 399 | KV cache 压缩状态缓存 |
| **RoPE** | `build_deepseek_v4_rope` in `common/rope.py` | 36 | RoPE 辅助函数 |
| **共享算子** | 8 个文件 in `common/ops/` | ~2,700 | Triton 实现的 KV cache 管理、融合算子 |
| **NV 专用算子** | 6 个文件 in `nvidia/ops/` | ~3,400 | CuteDSL 实现 sparse_attn、o_proj、dequant、indexer、megamoe |
| **量化配置** | `DeepseekV4FP8Config` in `quant_config.py` | 160 | FP4/FP8 专家量化、NVFP4 (SM100) 配置 |

### 3.2 修改文件一览

所有模型代码位于 `vllm/models/deepseek_v4/` 目录下，按硬件平台分三个子目录：

| 目录/文件 | 行数 | 作用 |
|----------|------|------|
| `__init__.py` | 31 | 入口，按平台分发 |
| `quant_config.py` | 160 | `DeepseekV4FP8Config`：FP8/MXFP4 专家量化配置 |
| `attention.py` | 800 | `DeepseekV4Attention`：MLA 注意力基类 + `DeepseekV4IndexerCache` |
| `compressor.py` | 399 | `CompressorStateCache`：KV cache 压缩器 |
| `sparse_mla.py` | 416 | `DeepseekV4FlashMLABackend`：稀疏 MLA AttentionBackend |
| `common/rope.py` | 36 | `build_deepseek_v4_rope()`：RoPE 辅助函数 |
| `common/ops/`（8 个文件） | ~2,700 | 共享 Triton 算子 |
| `nvidia/model.py` | ~1,340 | NVIDIA 模型主文件 |
| `nvidia/mtp.py` | ~400 | NVIDIA MTP 实现 |
| `nvidia/flashmla.py` | ~320 | FlashMLA 注意力实现 |
| `nvidia/flashinfer_sparse.py` | ~80 | FlashInfer 稀疏注意力 |
| `nvidia/ops/`（6 个文件） | ~3,400 | CuteDSL 算子（sparse_attn, o_proj, dequant, indexer, megamoe） |
| `amd/` | 2 个文件 ~900 | ROCm 适配 |
| `xpu/` | 4 个文件 ~1,800 | Intel XPU 适配 |

### 3.2 模型层次结构

```
DeepseekV4ForCausalLM                     # 顶层：模型注册 + 权重管理
  └─ DeepseekV4Model                      # 主干：Embed + DecoderLayers + Norm + HC Head
       ├─ embed_tokens (VocabParallelEmbedding)
       ├─ DeepseekV4DecoderLayer × N      # 核心计算层
       │    ├─ attn_norm / ffn_norm (RMSNorm)
       │    ├─ attn: DeepseekV4Attention   # MLA 注意力（平台特有子类）
       │    └─ ffn: DeepseekV4MoE          # MoE 混合专家
       └─ norm (RMSNorm)
```

### 3.3 详解：`DeepseekV4MLP`（前馈网络）

```
vllm/models/deepseek_v4/nvidia/model.py`, lines 70-118
```

标准 SwiGLU MLP，用于 shared experts：

```python
class DeepseekV4MLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size, hidden_act, ...):
        # gate_up_proj: 门控和上投影合并 (MergedColumnParallelLinear)
        self.gate_up_proj = MergedColumnParallelLinear(hidden_size, [intermediate_size] * 2)
        # down_proj: 下投影
        self.down_proj = RowParallelLinear(intermediate_size, hidden_size)
        self.act_fn = SiluAndMul()  # SiLU 激活

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)       # SiLU(gate) * up
        x, _ = self.down_proj(x)
        return x
```

**特点**：
- 使用 `MergedColumnParallelLinear` 将 gate 和 up 两个投影合并为一个矩阵乘法
- 支持 TP（张量并行），`disable_tp` 参数控制是否切分
- 支持量化（通过 `quant_config` 参数传递）

---

### 3.4 详解：`DeepseekV4MoE`（混合专家层）

```
nvidia/model.py`, lines 478-724
```

DS V4 的 MoE 层支持两种后端：
1. **MegaMoE**（`deep_gemm_mega_moe`）：基于 DeepGEMM 的 NVidia SM100 专用实现
2. **FusedMoE**：通用 FusedMoE 实现（所有 GPU）

```python
class DeepseekV4MoE(nn.Module):
    def __init__(self, vllm_config, prefix=""):
        self.use_mega_moe = (vllm_config.kernel_config.moe_backend == "deep_gemm_mega_moe")
        
        # Gate: 线性层 + routed scoring
        self.gate = GateLinear(...)
        self.gate.e_score_correction_bias = None
        self.gate.tid2eid = None  # hash MoE 的 token->expert 映射
        
        # Shared experts
        if config.n_shared_experts is None:
            self.shared_experts = None
        else:
            self.shared_experts = DeepseekV4MLP(...)
        
        if self.use_mega_moe:
            self._init_mega_moe_experts(...)
        else:
            self._init_fused_moe_experts(...)
```

**MegaMoE 初始化**（`_init_mega_moe_experts`，lines 571-611）：

```python
def _init_mega_moe_experts(self, vllm_config, config, prefix):
    # Expert Parallel (EP) 设置
    self.ep_group = get_ep_group()
    self.ep_size = self.ep_group.world_size
    
    # EPLB: 逻辑专家 → 物理副本映射
    self.n_redundant_experts = eplb_config.num_redundant_experts
    self.n_physical_experts = self.n_logical_experts + self.n_redundant_experts
    
    # 每个 rank 的物理专家数量
    self.n_local_physical_experts = self.n_physical_experts // self.ep_size
    
    # DeepseekV4MegaMoEExperts: 以 uint8 存储 FP4 权重的专家
    self.experts = DeepseekV4MegaMoEExperts(
        vllm_config,
        num_experts=self.n_physical_experts,
        num_local_experts=self.n_local_physical_experts,
        ...
    )
```

**MegaMoE forward**（lines 659-698）：

```
1. Gate 计算 router logits
2. fused_topk_bias 计算 topk_weights 和 topk_ids
3. EPLB：将逻辑 expert ID 映射到物理副本（load balancing）
4. prepare_megamoe_inputs：准备 DeepGEMM 输入
5. deep_gemm.fp8_fp4_mega_moe：调用 DeepGEMM 的 FP4 MegaMoE kernel
6. Shared experts 输出叠加
```

**FusedMoE 初始化**（`_init_fused_moe_experts`，lines 613-657）：

```python
def _init_fused_moe_experts(self, vllm_config, config, quant_config, prefix):
    # 按 TP 切分专家
    self.n_local_physical_experts = self.n_physical_experts // self.tp_size
    self.experts_start_idx = self.tp_rank * self.n_local_experts
    
    self.experts = FusedMoE(
        shared_experts=self.shared_experts,
        gate=self.gate,
        num_experts=config.n_routed_experts,
        top_k=config.num_experts_per_tok,
        ...
    )
```

**forward 路由**（lines 659-698）：`use_mega_moe` 为 True 时走 MegaMoE 路径，否则走 FusedMoE 路径。

---

#### 3.4.1 详解：`DeepseekV4MegaMoEExperts`

```
nvidia/model.py`, lines 140-473
```

专门为 SM100（Blackwell）架构设计的 FP4 专家实现。

**权重存储**（lines 173-218）：
```python
# w13: gate+up 合并权重 (uint8, 实际是 FP4)
self.w13_weight = nn.Parameter(
    torch.zeros(num_local_experts, 2 * intermediate_size, hidden_size // 2, dtype=torch.uint8)
)
self.w13_weight_scale = nn.Parameter(
    torch.zeros(num_local_experts, 2 * intermediate_size, hidden_size // 32, dtype=torch.uint8)
)
```

**权重转换**（`finalize_weights`，lines 296-334）：
```python
def finalize_weights(self):
    # 使用 DeepGEMM 将权重转换为 MegaMoE 需要的布局
    deep_gemm = _import_deep_gemm()
    self._transformed_l1_weights, self._transformed_l2_weights = (
        deep_gemm.transform_weights_for_mega_moe(
            (w13_weight, w13_scale), (w2_weight, w2_scale)
        )
    )
    # 释放原始权重（节省显存）
    self.w13_weight = None
    self.w2_weight = None
```

**forward**（lines 411-472）：
```python
def forward(self, hidden_states, topk_weights, topk_ids, ...):
    # 1. EPLB 映射
    topk_ids = eplb_map_to_physical_and_record(...)
    
    # 2. 准备对称缓冲区
    prepare_megamoe_inputs(hidden_states, topk_weights, topk_ids, symm_buffer, ...)
    
    # 3. 调用 DeepGEMM kernel
    deep_gemm.fp8_fp4_mega_moe(y, self._transformed_l1_weights, ...)
    return y
```

**EPLB 支持**（lines 365-406）：通过 `get_expert_weights()` 返回 EPLB 需要的权重视图。

---

### 3.5 详解：注意力机制

#### 3.5.1 注意力基类 `DeepseekV4Attention`

```
attention.py`, lines 98-618
```

MLA（Multi-head Latent Attention）的抽象基类，定义了所有平台共享的权重和结构：

```python
class DeepseekV4Attention(nn.Module, AttentionLayerBase, ABC):
    # 平台子类需提供的属性
    backend_cls: ClassVar[type[AttentionBackend]]      # AttentionBackend
    use_flashmla_fp8_layout: ClassVar[bool] = True     # KV cache 布局
    
    @abstractmethod
    def get_padded_num_q_heads(cls, num_heads): ...    # Q head padding
    @abstractmethod
    def forward_mqa(self, q, kv, positions, output): ...  # 稀疏 MLA 前向
    @abstractmethod
    def _o_proj(self, o, positions): ...               # 输出投影
```

**初始化**（lines 148-296）：

```python
def __init__(self, vllm_config, prefix, topk_indices_buffer, aux_stream_list):
    config = vllm_config.model_config.hf_config
    # MLA 投影
    self.fused_wqa_wkv = MergedColumnParallelLinear(     # W_Q_A & W_KV 融合
        hidden_size, [q_lora_rank, head_dim], ...
    )
    self.q_norm = RMSNorm(q_lora_rank, eps)
    self.wq_b = ColumnParallelLinear(q_lora_rank, n_heads * head_dim, ...)  # W_Q_B
    self.kv_norm = RMSNorm(head_dim, eps)
    
    # 输出投影
    self.wo_a = ColumnParallelLinear(n_heads * head_dim // n_groups, ...)  # W_O_A
    self.wo_b = RowParallelLinear(n_groups * o_lora_rank, hidden_size, ...)  # W_O_B
    
    # RoPE
    self.rotary_emb = build_deepseek_v4_rope(config, ...)
    
    # Indexer（C4 压缩率时启用）
    if self.compress_ratio == 4:
        self.indexer = DeepseekV4Indexer(...)
    
    # KV cache
    self.swa_cache_layer = DeepseekV4SWACache(...)
```

**forward 调度**（lines 318-618）：

```python
def forward(self, positions, hidden_states, ...):
    # 1. 输入投影
    q, kv = self.fused_wqa_wkv(hidden_states)  # 融合 Q 和 KV 投影
    q_normed = self.q_norm(q)
    kv_normed = self.kv_norm(kv)
    q = self.wq_b(q_normed)                     # Q 升维到 n_heads * head_dim
    
    # 2. 调用平台子类的稀疏 MLA 前向
    self.forward_mqa(q, kv_normed, positions, output)  # 抽象方法
    
    # 3. 输出投影（含 inv-RoPE + wo_a + wo_b）
    return self._o_proj(output, positions)       # 抽象方法
```

MLA（Multi-head Latent Attention）的关键：Q 和 KV 先通过低秩投影（`q_lora_rank`，`head_dim`）压缩，再通过 Wq_b 升维到完整的注意力头数。这比标准 MHA 的参数和计算量都更少。

#### 3.5.2 `DeepseekV4FlashMLAAttention`（FlashMLA 实现）

```
nvidia/flashmla.py`, lines 33-320
```

```python
class DeepseekV4FlashMLAAttention(DeepseekV4Attention):
    backend_cls = DeepseekV4FlashMLABackend  # 来自 sparse_mla.py
    
    def forward_mqa(self, q, kv, positions, output):
        # 获取 forward context 中的注意力元数据
        attn_metadata = get_forward_context().attn_metadata
        
        if attn_metadata is None:
            # Warmup dummy run：预分配 workspace
            ...
        elif attn_metadata.is_prefill:
            # Prefill：使用 dequantize_and_gather + flash_mla_sparse_fwd
            k = dequantize_and_gather_k_cache(self.swa_cache_layer, ...)
            flash_mla_sparse_fwd(q, k, ..., output, ...)
        else:
            # Decode：使用 flash_mla_with_kvcache + 稀疏 topk
            topk_lens, topk_indices = compute_global_topk_indices_and_lens(...)
            flash_mla_with_kvcache(q, k_cache, topk_indices, ..., output)
    
    def _o_proj(self, o, positions):
        # 使用 deep_gemm_fp8_o_proj：融合 inv-RoPE + wo_a + wo_b
        return deep_gemm_fp8_o_proj(o, positions, ...)
```

**AttentionBackend**：`DeepseekV4FlashMLABackend`（`sparse_mla.py`）管理 KV cache 格式、Metadata 构建和 workspace 分配。

#### 3.5.3 注意力后端选择

```python
# nvidia/model.py:726
def _select_dsv4_attn_cls(vllm_config):
    if backend == AttentionBackendEnum.FLASHINFER_MLA_SPARSE_DSV4:
        return DeepseekV4FlashInferMLAAttention  # FlashInfer 路径
    return DeepseekV4FlashMLAAttention           # FlashMLA 路径（默认）
```

- **FlashMLA 路径**：使用 FlashMLA kernel + UE8M0 block-scaled FP8 KV cache
- **FlashInfer 路径**：使用 FlashInfer TRTLLM-gen kernel + plain bf16/FP8 KV cache

#### 3.5.4 `DeepseekV4Indexer`（稀疏索引器）

```
attention.py`, lines 662-800
```

负责压缩注意力中的 topk 索引管理：
```python
class DeepseekV4Indexer(nn.Module):
    def __init__(self, ...):
        self.compressor = ColumnParallelLinear(hidden_size, head_dim, ...)
        self.index_score = ReplicatedLinear(hidden_size, n_heads, ...)
        self.weights_proj = ReplicatedLinear(num_layers, n_heads, ...)
    
    def forward(self, hidden_states, positions):
        # 1. 计算压缩后的 KV 分数
        # 2. 根据 topk 选择重要的 KV 位置
        # 3. 返回稀疏注意力需要的索引
```

---

### 3.6 详解：`DeepseekV4DecoderLayer`（解码器层）

```
nvidia/model.py`, lines 740-885
```

```python
class DeepseekV4DecoderLayer(nn.Module):
    def __init__(self, vllm_config, prefix, topk_indices_buffer, aux_stream_list):
        # 注意力（平台特异子类）
        self.attn = _select_dsv4_attn_cls(vllm_config)(vllm_config, prefix, ...)
        # MoE
        self.ffn = DeepseekV4MoE(vllm_config, prefix)
        # LayerNorm
        self.attn_norm = RMSNorm(hidden_size, eps)
        self.ffn_norm = RMSNorm(hidden_size, eps)
        # HC（Hybrid Chunk）参数
        self.hc_attn_fn = nn.Parameter(...)  # 注意力 HC 融合矩阵
        self.hc_ffn_fn = nn.Parameter(...)   # FFN HC 融合矩阵
    
    def forward(self, x, positions, input_ids, post_mix, res_mix, residual):
        if residual is None:
            # 第一层：独立 mhc_pre
            post_mix, res_mix, x = mhc_pre_tilelang(x, hc_attn_fn, ..., norm_weight=...)
        else:
            # 后续层：融合（前一层 post_pre + 当前层 pre）
            residual, post_mix, res_mix, x = mhc_fused_post_pre_tilelang(
                x, residual, post_mix, res_mix, hc_attn_fn, ...)
        
        x = self.attn(positions, x, None)  # 注意：attn_norm 已融合在 mhc_pre 中
        
        # FFN 部分同理
        residual, post_mix, res_mix, x = mhc_fused_post_pre_tilelang(
            x, residual, post_mix, res_mix, hc_ffn_fn, ...)
        x = self.ffn(x, input_ids)
        
        return x, residual, post_mix, res_mix
```

**HC（Hybrid Chunk）**：DeepSeek V4 特有的残差流处理机制——
- 将输入 `x` 通过 `hc_mult` 倍扩展为 `(T, hc_mult, D)` 的多流表示
- 每层在注意力/FFN 前后通过 `hc_pre` / `hc_post` 进行跨流融合
- 减少层间通信，提高计算效率

---

### 3.7 详解：`DeepseekV4Model`（主干网络）

```
nvidia/model.py`, lines 888-1057
```

```python
class DeepseekV4Model(nn.Module):
    def __init__(self, *, vllm_config, prefix=""):
        self.use_mega_moe = (vllm_config.kernel_config.moe_backend == "deep_gemm_mega_moe")
        
        # Embed
        self.embed_tokens = VocabParallelEmbedding(...)
        
        # Decoder layers（通过 make_layers 支持 PP）
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: DeepseekV4DecoderLayer(vllm_config, prefix, ...),
        )
        
        self.norm = RMSNorm(hidden_size, eps)
        
        # HC Head 参数（MTP 也用到）
        self.hc_head_fn = nn.Parameter(...)
        
        # MTP hidden states buffer
        self._mtp_hidden_buffer = torch.empty(max_tokens, hc_dim, ...)
    
    def forward(self, input_ids, positions, intermediate_tensors, inputs_embeds):
        # 1. Embed + HC 扩展 (T, D) → (T, hc_mult, D)
        hidden_states = self.embed_input_ids(input_ids)
        hidden_states = hidden_states.unsqueeze(-2).repeat(1, self.hc_mult, 1)
        
        # 2. 逐层 DecoderLayer
        for layer in self.layers:
            hidden_states, residual, post_mix, res_mix = layer(...)
        hidden_states = mhc_post_tilelang(hidden_states, residual, post_mix, res_mix)
        
        # 3. 保存 MTP 目标的 pre-hc_head 残差
        self._mtp_hidden_buffer[:num_tokens].copy_(hidden_states.flatten(1))
        
        # 4. HC Head：将 (T, hc_mult, D) 压缩回 (T, D)
        hidden_states = hc_head_fused_kernel_tilelang(hidden_states, ...)
        hidden_states = self.norm(hidden_states)
        return hidden_states
```

**关键设计**：
- **`_mtp_hidden_buffer`**：保存 pre-hc_head 的残差流状态，供 MTP draft 模型使用
- **PP 支持**：通过 `make_layers` + `intermediate_tensors` 实现 Pipeline Parallelism

---

### 3.8 详解：`DeepseekV4ForCausalLM`（顶层模型）

```
nvidia/model.py`, lines 1251-1337
```

```python
class DeepseekV4ForCausalLM(nn.Module, SupportsPP, DeepseekV4MixtureOfExperts):
    model_cls = DeepseekV4Model
    hf_to_vllm_mapper = _make_deepseek_v4_weights_mapper("fp4")
    
    def __init__(self, *, vllm_config, prefix=""):
        # 根据 expert_dtype 选择权重映射器（FP4 vs FP8）
        expert_dtype = getattr(config, "expert_dtype", "fp4")
        if expert_dtype != "fp4":
            self.hf_to_vllm_mapper = _make_deepseek_v4_weights_mapper(expert_dtype)
        
        self.model = self.model_cls(vllm_config=vllm_config, prefix=...)
        self.lm_head = ParallelLMHead(...)  # 语言模型头
        self.logits_processor = LogitsProcessor(...)
        self.set_moe_parameters()
```

**权重映射器**（`_make_deepseek_v4_weights_mapper`，lines 1177-1211）：
```python
def _make_deepseek_v4_weights_mapper(expert_dtype):
    scale_regex = {
        re.compile(r"(\.experts\.\d+\.w[123])\.scale$"): r"\1.weight_scale",
        re.compile(r"\.scale$"): ".weight_scale_inv",
    }
    return WeightsMapper(
        orig_to_new_prefix={
            "layers.": "model.layers.",
            "hc_head": "model.hc_head",
            "mtp.": "model.mtp.",
        },
        orig_to_new_regex=scale_regex,
        orig_to_new_suffix={
            "head.weight": "lm_head.weight",
            "embed.weight": "embed_tokens.weight",
            ".ffn.gate.bias": ".ffn.gate.e_score_correction_bias",
        },
        orig_to_new_substr={
            ".shared_experts.w2": ".shared_experts.down_proj",
        },
    )
```

**权重加载**（`load_weights`，lines 1333-1337）：
```python
def load_weights(self, weights):
    loader = AutoWeightsLoader(self, skip_substrs=["mtp."])
    loaded_params = loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
    self.model.finalize_mega_moe_weights()  # 转换 MegaMoE 权重布局
    return loaded_params
```

**`DeepseekV4MixtureOfExperts`**（lines 1214-1249）：管理 MoE 层的元数据，供 EPLB 调度使用。

---

### 3.9 详解：MTP（Multi-Token Prediction）

**文件**：`nvidia/mtp.py`

**`DeepSeekV4MTP`**（lines 260-380）：
```
DeepSeekV4MTP
  └─ DSV4MultiTokenPredictor
       └─ DSV4MultiTokenPredictorLayer × N
            ├─ enorm / hnorm (RMSNorm)
            ├─ e_proj / h_proj (ReplicatedLinear, separate in V4)
            ├─ shared_head (SharedHead with RMSNorm + ParallelLMHead)
            └─ mtp_block (DeepseekV4DecoderLayer, 复用主模型层)
```

**MTP forward**（`DeepSeekV4MultiTokenPredictorLayer.forward`，lines 132-167）：

```python
def forward(self, input_ids, positions, previous_hidden_states, inputs_embeds, spec_step_idx):
    # 1. 将 target 的 pre-hc_head 残差 reshape 为 3D
    previous_hidden_states = previous_hidden_states.view(-1, self.hc_mult, H)
    
    # 2. 融合：enorm（位置0掩码）+ hnorm
    inputs_embeds, previous_hidden_states = fused_mtp_input_rmsnorm(
        inputs_embeds, positions, previous_hidden_states, ...
    )
    
    # 3. e_proj + h_proj 融合
    hidden_states = self.h_proj(previous_hidden_states) + self.e_proj(inputs_embeds).unsqueeze(-2)
    
    # 4. 复用 DeepseekV4DecoderLayer
    hidden_states, residual, post_mix, res_mix = self.mtp_block(
        positions=positions, x=hidden_states, input_ids=None)
    
    # 5. HC post + 展平（hc_head 延迟到 compute_logits 执行）
    hidden_states = mhc_post_tilelang(hidden_states, residual, post_mix, res_mix)
    return hidden_states.flatten(1)
```

**`compute_logits`**（lines 231-257）：
```python
def compute_logits(self, hidden_states, spec_step_idx):
    # 1. hc_head：将 (T, hc_mult, D) 压缩回 (T, D)
    hidden_states = hc_head_fused_kernel_tilelang(hidden_states, ...)
    # 2. shared_head RMSNorm
    hidden_states = mtp_shared_head_rmsnorm(hidden_states, ...)
    # 3. 语言模型头
    logits = self.logits_processor(shared_head.head, hidden_states)
    return logits
```

**V4 与 V3 的 MTP 差异**（注释明确说明）：
- V4 使用分离的 `e_proj` / `h_proj`（带 fp8 量化），V3 使用融合的 `eh_proj`
- V4 增加 `hc_head` 超压缩词汇投影
- V4 的 DecoderLayer 有独立的 aux-stream 管理
- V4 checkpoint 的权重名称需要 remapping

---

### 3.10 详解：共享算子库（common/ops）

`vllm/models/deepseek_v4/common/ops/` 包含 8 个 Triton 实现的算子，被所有硬件平台共享：

| 文件 | 行数 | 功能 |
|------|------|------|
| `cache_utils.py` | 899 | KV cache 量化/反量化、topk 索引、稀疏索引预计算 |
| `fused_compress_quant_cache.py` | 666 | 压缩、归约、RoPE、FP8 量化融合 kernel |
| `fused_indexer_q.py` | 438 | Indexer Q 投影 + RoPE + 量化融合 |
| `fused_inv_rope_fp8_quant.py` | 318 | 逆 RoPE + FP8 量化融合 |
| `fused_mtp_input_rmsnorm.py` | 203 | MTP 输入 RMSNorm 融合（enorm + hnorm） |
| `fused_qk_rmsnorm.py` | 96 | Q/KV RMSNorm 融合 |
| `save_partial_states.py` | 101 | 保存部分状态（用于 PP） |

这些算子全部使用 **Triton** 实现，在 vLLM-Ascend 中也能被复用（通过 import 方式）。

---

### 3.11 详解：Attention 后端（sparse_mla.py + compressor.py）

**sparse_mla.py**（416 行）：

- `DeepseekV4FlashMLABackend`：实现 `AttentionBackend` 接口
  - KV cache 格式定义
  - Metadata 构建
  - Workspace 管理
- `DeepseekV4FlashMLAMetadata`：注意力元数据结构

```python
class DeepseekV4FlashMLABackend(AttentionBackend):
    @classmethod
    def get_kv_cache_shape(cls, ...):
        # UE8M0 block-scaled FP8 格式
        # block_size=64, packed as uint8
        ...
    
    @classmethod
    def is_mla(cls) -> bool: return True
    @classmethod
    def is_sparse(cls) -> bool: return True
```

**compressor.py**（399 行）：

- `CompressorStateCache`：KV cache 压缩状态的缓存层
- `CompressorBackend`：压缩器的 AttentionBackend

```python
class CompressorStateCache(torch.nn.Module, AttentionLayerBase):
    def __init__(self, ...):
        # 压缩器 + 状态缓存
        self.compressor = DeepseekV4Compressor(...)
        self.state_cache = {}
```

---

### 3.12 详解：量化配置

`quant_config.py`（160 行）：

```python
class DeepseekV4FP8Config(Fp8Config):
    @property
    def expert_dtype(self) -> str:
        return getattr(self, "_expert_dtype", "fp8")
    
    @property
    def is_scale_e8m0(self) -> bool:
        # FP8 使用 E8M0 缩放因子
        return True
    
    def moe_quant_algo(self) -> str:
        return "fp8_block_quant"  # 或 "fp4_mxfp4"
    
    def _get_nvfp4_config(self) -> ModelOptNvFp4Config:
        # NVFP4 配置（SM100 Blackwell）
        ...
```

**两种专家量化模式**：
1. **FP4（MXFP4）**：默认模式，使用 `Mxfp4MoEMethod`，权重为 uint8（实际是 FP4）
2. **FP8**：使用 `Fp8MoEMethod`（block_quant=True），权重为 FP8

权重映射器会根据 `expert_dtype` 选择不同的 scale 重命名规则。

---

## 四、vLLM-Ascend 源码详解

> vLLM-Ascend 的 DS V4 模型全部实现在 `vllm_ascend/models/deepseek_v4.py`（1,355 行）这一个文件中，外加两个自定义算子文件（`ops/dsa.py` 272 行 + `ops/rope_dsv4.py` 238 行），结构远比 vLLM 的 36 个文件精简。

### 4.0 模块功能一览

| 模块 | 类/文件 | 行数 | 核心功能 |
|------|---------|------|---------|
| **MLP** | `DeepseekV2MLP` in `deepseek_v4.py` | 184-226 | SwiGLU 前馈网络，与 NVIDIA 实现一致 |
| **MoE** | `DeepseekV4MoE` in `deepseek_v4.py` | 229-385 | `mix_placement` 混合放置、FusedMoE 统一管理 routed+shared、`muls_add_triton` 融合 |
| **Indexer** | `Indexer` in `deepseek_v4.py` | 404-468 | 空 forward、参数/缓存管理、KV cache 类型按设备区分（A5: fp8 vs int8） |
| **Compressor** | `Compressor` in `deepseek_v4.py` | 471-578 | WKV/Wgate 分离、overlap_transform、`npu_rotary_mul` 硬件加速、state cache |
| **Attention（DSA 封装）** | `DeepseekV4Attention` in `ds_v4.py` | 581-767 | DSA 封装层、W_Q_A/W_KV 分离、ComplexExpRotaryEmbedding、skip_topk 策略 |
| **DSA 注意力核心** | `AscendDeepseekSparseAttention` in `ops/dsa.py` | 60-272 | C++ DSA custom op、6-tuple KV cache、ACL graph capture、warmup |
| **DecoderLayer** | `DeepseekV2DecoderLayer` in `ds_v4.py` | 770-860 | `npu_hc_pre/post` 自定义算子、显式 LayerNorm、两态残差流 |
| **Model** | `DeepseekV4Model` in `deepseek_v4.py` | 864-1006 | `@support_torch_compile`、纯 PyTorch hc_head、FlashComm1 all_gather |
| **ForCausalLM** | `AscendDeepseekV4ForCausalLM` in `ds_v4.py` | 1049-1354 | 多继承（SupportsPP+LoRA+Eagle）、mix_placement 权重加载、大量名称 remap |
| **MTP** | `DSV4MTP` in `deepseek_v4_mtp.py` | 506 | 复用主模型 DeepseekV2DecoderLayer、纯 PyTorch hc_head |
| **RoPE** | `ComplexExpRotaryEmbedding` in `ops/rope_dsv4.py` | 104-237 | `RopeGlobalState` 全局缓存、`npu_rotary_mul` 硬件加速、多组 RoPE 支持 |

### 4.1 文件结构与层次

```
AscendDeepseekV4ForCausalLM               # 顶层，多继承 SupportsPP + SupportsLoRA + SupportsEagle
  └─ DeepseekV4Model                      # 主干网络（@support_torch_compile）
       ├─ embed_tokens (VocabParallelEmbedding)
       ├─ DeepseekV2DecoderLayer × N      # 注意：命名是 V2，实际是 V4 DecoderLayer
       │    ├─ self_attn: DeepseekV4Attention  # 封装了 AscendDeepseekSparseAttention
       │    └─ mlp: DeepseekV4MoE
       └─ norm (RMSNorm)
```

**类命名差异**：vLLM-Ascend 大量使用 `DeepseekV2*` 前缀（如 `DeepseekV2DecoderLayer`、`DeepseekV2MLP`、`DeepseekV2MixtureOfExperts`），这是因为 Ascend 代码从 V2 版本演进而来，但实际上是 V4 架构。

### 4.2 详解：辅助函数

```
deepseek_v4.py`, lines 94-183
```

**`hadamard_transform_ref`**（lines 94-108）：Hadamard 变换，用于激活值旋转
```python
def hadamard_transform_ref(x, scale=1.0):
    from scipy.linalg import hadamard
    # 将输入 x 通过 Hadamard 矩阵变换
    # SC20 论文：用 Hadamard 变换替代 LayerNorm 的部分功能
```

**`precompute_freqs_cis_cpu`**（lines 116-151）：YLRC（YaRN Linear Ramp Correction）RoPE 频率预计算
```python
def precompute_freqs_cis_cpu(dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow):
    # 1. 计算 base → freqs (θ_i = 1/(base^(2i/d)))
    # 2. Yarn: 对超出 original_seq_len 的维度做 NTK-aware scaling
    # 3. 用 torch.polar 生成复数 cis 值
```

**`apply_rotary_emb`**（lines 154-175）：通过复数乘法应用 RoPE
```python
def apply_rotary_emb(x, freqs_cis, inverse=False):
    x = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if inverse:
        freqs_cis = freqs_cis.conj()  # 逆 RoPE（用于输出投影）
    x = torch.view_as_real(x * freqs_cis.to(x.device)).flatten(-2)
```

**`get_spec_layer_idx_from_weight_name`**（lines 178-183）：从权重名解析 MTP 层索引，用于权重加载时跳过 MTP 层。

---

### 4.3 详解：`DeepseekV2MLP`（前馈网络）

```
deepseek_v4.py`, lines 184-226
```

与 vLLM NVIDIA 的 `DeepseekV4MLP` 几乎相同——SwiGLU 结构，`MergedColumnParallelLinear` + `RowParallelLinear`。

```python
class DeepseekV2MLP(nn.Module):
    def __init__(self, ..., is_sequence_parallel=False, ...):
        self.gate_up_proj = MergedColumnParallelLinear(hidden_size, [intermediate_size] * 2, ...)
        self.down_proj = RowParallelLinear(intermediate_size, hidden_size, ...)
        self.act_fn = SiluAndMul()
    
    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x
```

---

### 4.4 详解：`DeepseekV4MoE`（混合专家层）

```
deepseek_v4.py`, lines 229-385
```

**初始化**（lines 230-333）：

```python
class DeepseekV4MoE(nn.Module):
    def __init__(self, config, parallel_config, quant_config, prefix, is_draft_layer=False):
        # EP 设置（Ascend 默认开启 Expert Parallel）
        self.ep_group = get_ep_group().device_group
        self.ep_rank = get_ep_group().rank_in_group
        self.ep_size = self.ep_group.size()
        
        # mix_placement: Ascend 独有功能——shared experts 与 routed experts 混合放置
        self.is_fusion_moe_shared_experts_enabled = getattr(get_ascend_config(), "mix_placement", False)
        if config.n_shared_experts is None or self.is_fusion_moe_shared_experts_enabled:
            self.shared_experts = None  # 由 FusedMoE 内部管理
        else:
            self.shared_experts = DeepseekV2MLP(...)
        
        # Gate（路由层）
        self.gate = ReplicatedLinear(hidden_size, n_routed_experts, ...)
        self.gate.precast_fp32_weight = True  # Ascend 特有: gate 权重强制 fp32
        
        # FusedMoE（统一管理 routed + shared experts）
        self.experts = FusedMoE(
            shared_experts=self.shared_experts,
            gate=self.gate,
            use_grouped_topk=True,               # 分组 topk（V4 特性）
            num_expert_group=config.n_group,
            topk_group=config.topk_group,
            scoring_func="softmax",
            enable_eplb=self.enable_eplb,
            is_sequence_parallel=self.is_sequence_parallel,
            n_shared_experts=config.n_shared_experts if mix_placement else 0,
            ...
        )
```

**Ascend vs NVIDIA 的 MoE 关键区别**：
| 维度 | NVIDIA | Ascend |
|------|--------|--------|
| 专家并行 | `get_ep_group()`（MegaMoE 必需） | `get_ep_group()`（默认开启） |
| Gate 精度 | `GateLinear` + `out_dtype=torch.float32` | `ReplicatedLinear` + `precast_fp32_weight=True` |
| Router | `fused_topk_bias` + `sqrtsoftplus` | `F.linear` + `softmax`（FusedMoE 内部） |
| Shared experts | `DeepseekV4MLP` 独立模块 | 支持 `mix_placement` 集成到 FusedMoE |
| 专家量化 | 支持 FP4（`DeepseekV4MegaMoEExperts`） | 仅 FP8（FusedMoE 泛化支持） |
| Grouped topk | 非显式 | `use_grouped_topk=True` |
| Sequence parallel | 条件生效 | 支持 `is_sequence_parallel` + chunk/all_gather |

**forward**（lines 335-385）：

```python
def forward(self, hidden_states, input_ids=None):
    # 1. Sequence parallel chunk（若开启）
    if self.is_sequence_parallel:
        hidden_states = sequence_parallel_chunk(hidden_states)
    
    # 2. Router（gate 或 FusedMoE 内部）
    if self.experts.is_internal_router:
        fused_moe_out = self.experts(hidden_states=hidden_states, router_logits=hidden_states)
    else:
        router_logits = F.linear(hidden_states.float(), self.gate.weight)
        fused_moe_out = self.experts(hidden_states=hidden_states, router_logits=router_logits)
    
    # 3. Shared experts 融合（FusedMoE 包返回 tuple 时）
    if fused_moe_out_is_tuple:
        shared_output, final_hidden_states = fused_moe_out
        if self.shared_experts is not None:
            final_hidden_states = muls_add_triton(final_hidden_states, shared_output, ...)
        else:
            final_hidden_states *= self.routed_scaling_factor
    
    # 4. All gather（sequence parallel 后）
    if self.is_sequence_parallel:
        final_hidden_states = tensor_model_parallel_all_gather(final_hidden_states, 0)
    
    return final_hidden_states
```

使用 `muls_add_triton`（Ascend 自定义 Triton 算子）替代 NVIDIA 的简单加法，支持乘以 scale factor。

---

### 4.5 详解：`Indexer` 和 `Compressor`

#### 4.5.1 Indexer（索引器）

```
deepseek_v4.py`, lines 404-468
```

```python
class Indexer(nn.Module):
    def __init__(self, vllm_config, config, compress_ratio, quant_config, cache_config, prefix):
        self.wq_b = ReplicatedLinear(q_lora_rank, n_heads * head_dim, ...)
        self.weights_proj = ReplicatedLinear(hidden_size, n_heads, ...)
        
        # KV cache 类型（按设备区分）
        k_dtype = torch.float8_e4m3fn if A5 else torch.int8
        self.k_cache = DeepseekV4IndexerCache(head_dim, dtype=k_dtype, ...)
        
        # Compressor（压缩比 >1 时启用）
        if compress_ratio > 1:
            self.compressor = Compressor(vllm_config, config, compress_ratio, ...)
    
    def forward(self, hidden_states, qr, positions, rotary_emb):
        return  # 空 forward，实际逻辑集成在 DSA 内部
```

**与 NVIDIA 的关键差异**：NVIDIA 使用 `DeepseekV4Indexer`（`attention.py:662`），有完整的 `forward` 实现；Ascend 的 `Indexer` 仅负责参数管理和缓存，实际计算在 DSA 内部完成。

#### 4.5.2 Compressor（压缩器）

```
deepseek_v4.py`, lines 471-578
```

```python
class Compressor(nn.Module):
    def __init__(self, ..., compress_ratio=4, head_dim=512, rotate=False, ...):
        # Absolute Position Embedding
        self.ape = nn.Parameter(torch.empty(compress_ratio, coff * head_dim, ...))
        
        # WKV + Wgate（V4 将 wkv 和 wgate 分开）
        self.wkv = ReplicatedLinear(dim, coff * head_dim, ...)
        self.wgate = ReplicatedLinear(dim, coff * head_dim, ...)
        
        # State cache（按压缩比区分）
        if compress_ratio == 4:
            self.state_cache = CompressorStateCache(state_dim=2 * coff * head_dim, ..., block_size=8)
        elif compress_ratio == 128:
            self.state_cache = CompressorStateCache(state_dim=2 * head_dim, ..., block_size=32)
```

**`overlap_transform`**（lines 538-543）：V4 特有的 overlap 变换
```python
def overlap_transform(self, tensor, value=0):
    # tensor: (b, s, 2, d) → 展开为 (b, s, 2*ratio, d)
    # 将前半部分重叠到下一时间步，实现窗口滑动
```

**`rope_single`**（lines 555-578）：使用 `torch_npu.npu_rotary_mul` 硬件加速的 RoPE
```python
def rope_single(self, x, cos, sin, inverse=False):
    x_rot = torch_npu.npu_rotary_mul(
        x.reshape(num_tokens, num_heads, 1, rotary_dim).to(torch.float32),
        cos, sin, rotary_mode="interleave"
    )
```

---

### 4.6 详解：`DeepseekV4Attention`（注意力层）

```
ds_v4.py`, lines 581-767
```

Ascend 的注意力层是一个**封装层**，内部将实际计算委托给 `AscendDeepseekSparseAttention`（DSA）。

**初始化**（lines 582-758）：

```python
class DeepseekV4Attention(nn.Module):
    def __init__(self, vllm_config, config, max_position_embeddings, cache_config, quant_config, prefix, topk_indices_buffer):
        # MLA 投影
        self.wq_a = ReplicatedLinear(dim, q_lora_rank, ...)          # W_Q_A（不同于 NVIDIA 的 fused_wqa_wkv）
        self.q_norm = RMSNorm(q_lora_rank, ...)
        wq_b_cls = ReplicatedLinear if self.enable_dsa_cp else ColumnParallelLinear
        self.wq_b = wq_b_cls(q_lora_rank, n_heads * head_dim, ...)   # W_Q_B
        
        self.wkv = ReplicatedLinear(dim, head_dim, ...)               # W_KV（与 W_Q_A 分离！）
        self.kv_norm = RMSNorm(head_dim, ...)
        
        self.wo_a = ColumnParallelLinear(n_heads * head_dim // n_groups, ...)  # W_O_A
        self.wo_b = RowParallelLinear(n_groups * o_lora_rank, dim, ...)       # W_O_B
        
        # RoPE（使用 Ascend 专用的 ComplexExpRotaryEmbedding）
        self.rotary_emb = ComplexExpRotaryEmbedding(
            vllm_config=vllm_config, layername=..., head_size=rope_head_dim, ...
        )
        
        # Compressor + Indexer（压缩比 >1 时启用）
        if compress_ratio > 1:
            self.compressor = Compressor(...)
            if compress_ratio == 4:
                self.indexer = Indexer(...)
        
        # DSA 模块聚合（将所有子模块打包传入 DSA）
        dsa_modules = DSAModules(
            wq_a=self.wq_a, q_norm=self.q_norm, wq_b=self.wq_b,
            wkv=self.wkv, kv_norm=self.kv_norm,
            wo_a=self.wo_a, wo_b=self.wo_b,
            attn_sink=self.attn_sink,
            indexer=self.indexer, compressor=self.compressor,
            topk_indices_buffer=topk_indices_buffer,
            skip_topk=skip_topk,
        )
        
        # 实例化 DSA 注意力（真正执行计算的类）
        self.dsa_attn = AscendDeepseekSparseAttention(
            dim=self.dim, n_heads=self.n_heads, scale=self.scale,
            dsa_modules=dsa_modules, cache_config=cache_config, ...
        )
```

**forward**（lines 761-767）：

```python
def forward(self, positions, hidden_states, llama_4_scaling):
    return self.dsa_attn(positions, hidden_states, llama_4_scaling)
```

**与 NVIDIA 的 Attention 关键区别**：

| 维度 | NVIDIA | Ascend |
|------|--------|--------|
| W_Q_A + W_KV | `fused_wqa_wkv` 合并为线性层 | `wq_a` + `wkv` 分离为两个线性层 |
| `wq_b` 切分 | `ColumnParallelLinear`（TP 切分） | `ReplicatedLinear` 或 `ColumnParallelLinear`（由 `enable_dsa_cp` 决定） |
| 稀疏索引 | vLLM AttentionBackend 管理 | DSA 内部集成（`skip_topk` 策略） |
| RoPE | `build_deepseek_v4_rope()` 36 行辅助 | `ComplexExpRotaryEmbedding` 238 行完整类 |
| NLP 注意力 | `npu_rotary_mul` 硬件加速 | 复数乘法 |

---

### 4.7 详解：DSA 注意力核心（`AscendDeepseekSparseAttention`）

```
ops/dsa.py`, lines 60-272
```

```python
class AscendDeepseekSparseAttention(MultiHeadLatentAttentionWrapper):
    def __init__(self, ..., dsa_modules, ...):
        # 从 DSAModules 中获取所有子模块
        self.wq_a = dsa_modules.wq_a
        self.q_norm = dsa_modules.q_norm
        ...
        self.swa_cache_layer = DeepseekV4SWACache(...)  # SWA 窗口缓存
        
        # C++ 实现的 DSA 注意力（关键的 NPU 算子）
        self.dsa_attn = DSAAttention(dim=dim, n_heads=n_heads, ..., cache_config=..., ...)
    
    def forward(self, positions, hidden_states, kv_cache=None, attn_metadata=None):
        # 所有 DSA 前向路径都通过 dsa_forward custom op 执行
        # 这是为了 ACL graph capture（NPU 计算图捕获）的需要
        torch.ops.vllm.dsa_forward(hidden_states, need_gather_q_kv, output, self.prefix)
        return output
```

**`dsa_forward` custom op**（lines 183-224）：

```python
def dsa_forward(hidden_states, need_gather_q_kv, output, layer_name):
    # 从 forward context 获取当前层的 self
    self = forward_context.no_compile_layers[layer_name]
    
    if attn_metadata is None:
        # Warmup: 预分配 workspace
        self.dsa_attn.impl.dsa_warmup_with_multistream(hidden_states)
        output.fill_(0)
    else:
        # 构建 KV cache 六元组
        kv_cache = _build_kv_cache(self, forward_context)
        # 调用 C++ DSA forward
        self.dsa_attn.impl.forward(self.dsa_attn.layer_name, hidden_states, kv_cache, ...)
```

**`_build_kv_cache`**（lines 232-266）：

```python
def _build_kv_cache(self, forward_context):
    """构建 DSA forward 需要的 6-tuple KV cache"""
    return tuple([
        compress_kv_cache,      # 压缩 KV cache
        swa_kv_cache,           # SWA 窗口 cache
        state_cache,            # Compressor 状态 cache
        indexer_state_cache,    # Indexer 状态 cache
        indexer_k_cache,        # Indexer K cache
        indexer_scale_cache,    # Indexer scale cache
    ])
```

DSA 的核心是 `DSAAttention`（C++ 实现，`vllm_ascend/models/layer/attention/layer.py`），它将整个 MLA 的 forward 路径——包括 Q 投影、RoPE、KV 压缩、稀疏 attention、输出投影——全部封装在一个 custom op 中，实现 NPU 上的计算图捕获优化。

---

### 4.8 详解：`DeepseekV2DecoderLayer`（解码器层）

```
deepseek_v4.py`, lines 770-860
```

```python
class DeepseekV2DecoderLayer(nn.Module):
    def __init__(self, vllm_config, prefix, config, topk_indices_buffer, is_draft_layer=False):
        self.self_attn = DeepseekV4Attention(vllm_config, ...)  # 封装 DSA
        self.mlp = DeepseekV4MoE(config, parallel_config, ...)
        
        # 独立的 LayerNorm（不同于 NVIDIA 的融合版本）
        self.input_layernorm = RMSNorm(hidden_size, eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps)
        
        # HC 参数
        self.hc_attn_fn = nn.Parameter(...)
        self.hc_ffn_fn = nn.Parameter(...)
    
    def hc_pre(self, x, hc_fn, hc_scale, hc_base):
        # 调用 Ascend 自定义算子
        y = torch.ops._C_ascend.npu_hc_pre(x, hc_fn, hc_scale, hc_base, ...)
        return y
    
    def hc_post(self, x, residual, post, comb):
        # 调用 Ascend 自定义算子（注意 unsqueeze/squeeze）
        y = torch.ops._C_ascend.npu_hc_post(
            x.unsqueeze(dim=0), residual.unsqueeze(dim=0), ...)
        return y.squeeze(dim=0)
    
    def forward(self, positions, hidden_states, residual, llama_4_scaling=None):
        # ---- Attention 部分 ----
        residual = hidden_states.clone()
        hidden_states, post, comb = self.hc_pre(hidden_states, self.hc_attn_fn, ...)
        hidden_states = self.input_layernorm(hidden_states)         # 显式 LayerNorm！
        hidden_states = self.self_attn(positions, hidden_states, llama_4_scaling)
        hidden_states = self.hc_post(hidden_states, residual, post, comb)
        
        # ---- FFN 部分 ----
        residual = hidden_states.clone()
        hidden_states, post, comb = self.hc_pre(hidden_states, self.hc_ffn_fn, ...)
        hidden_states = self.post_attention_layernorm(hidden_states)  # 显式 LayerNorm！
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.hc_post(hidden_states, residual, post, comb)
        
        return hidden_states, residual
```

**NVIDIA vs Ascend DecoderLayer 对照**：

| 步骤 | NVIDIA | Ascend |
|------|--------|--------|
| 1 | `mhc_pre`（含 attn_norm 权重） | `npu_hc_pre` + 显式 `input_layernorm` |
| 2 | `self.attn(positions, x, None)` | `self.self_attn(positions, hidden_states, llama_4_scaling)` |
| 3 | `mhc_fused_post_pre_tilelang`（融合 post + FFN pre） | `npu_hc_post` + 显式 `post_attention_layernorm` |
| 4 | `self.ffn(x, input_ids)` | `self.mlp(hidden_states)` |
| 5 | 返回 `(x, residual, post_mix, res_mix)` 三态残差 | 返回 `(hidden_states, residual)` 两态残差 |
| 签名 | `(x, positions, input_ids, post_mix, res_mix, residual)` | `(positions, hidden_states, residual, llama_4_scaling)` |

---

### 4.9 详解：`DeepseekV4Model`（主干网络）

```
deepseek_v4.py`, lines 864-1006
```

```python
@support_torch_compile
class DeepseekV4Model(nn.Module):
    def __init__(self, *, vllm_config, prefix=""):
        # Embed
        self.embed_tokens = VocabParallelEmbedding(...)
        
        # Decoder layers
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: DeepseekV2DecoderLayer(vllm_config, prefix, ...),
        )
        self.norm = RMSNorm(...)
        
        # HC Head
        self.hc_head_fn = nn.Parameter(...)
        
        # MTP hidden buffer
        self._mtp_hidden_buffer = torch.empty(max_tokens, hc_dim, ...)
    
    def hc_head(self, x, hc_fn, hc_scale, hc_base):
        """Ascend 的 hc_head 实现（纯 PyTorch，vs NVIDIA 的 tilelang kernel）"""
        # 1. RMSNorm
        rsqrt = torch.rsqrt(x.flatten(1).float().square().mean(-1, keepdim=True) + norm_eps)
        # 2. Linear projection + sigmoid
        mixes = torch.sigmoid(F.linear(x_flat, hc_fn) * rsqrt * hc_scale + hc_base) + hc_eps
        # 3. Weighted sum over hc_mult streams
        y = torch.sum(mixes.unsqueeze(-1) * x.view(shape), dim=1)
        return y.to(dtype)
    
    def forward(self, input_ids, positions, intermediate_tensors, inputs_embeds):
        # 1. Embed
        hidden_states = self.embed_input_ids(input_ids)
        residual = None
        
        # 2. HC expand: (T, D) → (T, hc_mult, D)
        hidden_states = hidden_states.unsqueeze(1).repeat(1, self.hc_mult, 1)
        
        # 3. Decoder layers
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual, llama_4_scaling)
        
        # 4. MTP target hidden states（FlashComm1 时需 all_gather）
        if flash_comm_v1_enabled:
            h_states_flat = tensor_model_parallel_all_gather(hidden_states.flatten(1), dim=0)
        else:
            h_states_flat = hidden_states.flatten(1)
        self._mtp_hidden_buffer[:num_tokens].copy_(h_states_flat)
        
        # 5. HC Head（纯 PyTorch）
        hidden_states = self.hc_head(hidden_states, self.hc_head_fn, ...)
        hidden_states = self.norm(hidden_states)
        return hidden_states
```

**与 NVIDIA 的 Model 关键区别**：

| 维度 | NVIDIA | Ascend |
|------|--------|--------|
| `hc_head` | `hc_head_fused_kernel_tilelang`（tilelang kernel） | 纯 PyTorch：RMSNorm + Linear + sigmoid |
| PP 中间张量 | `IntermediateTensors({"hidden_states": 3D})` | `IntermediateTensors({"hidden_states", "residual"})` |
| 编译装饰器 | 无 | `@support_torch_compile` |
| MTP buffer 管理 | 直接 copy_ | FlashComm1 时 all_gather + pad 处理 |
| `make_empty_intermediate_tensors` | 自定义：`torch.zeros(batch, hc_mult, D)` | `make_empty_intermediate_tensors_factory(["hidden_states", "residual"], ...)` |

---

### 4.10 详解：`AscendDeepseekV4ForCausalLM`（顶层模型）

```
deepseek_v4.py`, lines 1049-1354
```

```python
class AscendDeepseekV4ForCausalLM(nn.Module, SupportsPP, DeepseekV2MixtureOfExperts, SupportsLoRA, SupportsEagle):
    packed_modules_mapping = {"gate_up_proj": ["gate_proj", "up_proj"]}
    model_cls = DeepseekV4Model
    
    def __init__(self, *, vllm_config, prefix=""):
        self.model = self.model_cls(vllm_config=vllm_config, prefix=...)
        self.lm_head = ParallelLMHead(...)
        self.logits_processor = LogitsProcessor(...)
        self.set_moe_parameters()
    
    def forward(self, input_ids, positions, intermediate_tensors, inputs_embeds):
        hidden_states = self.model(input_ids, positions, intermediate_tensors, inputs_embeds)
        return hidden_states
```

**权重加载**（`load_weights`，lines 1138-1354）：

```python
def load_weights(self, weights):
    # 1. 参数映射：V4 checkpoint 名称 → 模型参数名称
    stacked_params_mapping = [
        ("gate_up_proj", "gate_proj", 0),   # gate/up 合并
        ("gate_up_proj", "up_proj", 1),
    ]
    expert_params_mapping = FusedMoE.make_expert_params_mapping(...)
    
    for name, loaded_weight in weights:
        # 2. 跳过 MTP 层
        if get_spec_layer_idx_from_weight_name(config, name):
            continue
        
        # 3. 名称重映射（大量 replace）
        name = name.replace(".w1.", ".gate_proj.")
        name = name.replace(".w2.", ".down_proj.")
        name = name.replace(".w3.", ".up_proj.")
        name = name.replace(".ffn.", ".mlp.")
        name = name.replace(".attn.", ".self_attn.")
        name = name.replace("model.head.", "lm_head.")
        ...
        
        # 4. Attn sink 特殊处理（DSA CP 时全量复制）
        if "sink" in name:
            if enable_dsa_cp():
                param.data.copy_(loaded_weight)
            else:
                narrow_weight = loaded_weight.narrow(0, head_start, heads_per_rank)
        
        # 5. mix_placement 时的 shared experts 处理
        if is_fusion_moe_shared_experts_layer:
            # 将 shared expert 权重按 n_shared_experts 分割
            for j in range(num_chunks):
                chunk_name = name.replace("mlp.shared_experts", f"mlp.experts.{n_routed_experts + j}")
                ...
```

**`get_expert_mapping`**（lines 1119-1129）：当 `mix_placement=True` 时，专家总数会包含 shared experts：
```python
def get_expert_mapping(self):
    return FusedMoE.make_expert_params_mapping(
        self.model,
        num_experts=self.config.n_routed_experts
            + (self.config.n_shared_experts if mix_placement else 0),
        ...
    )
```

---

### 4.11 详解：`ComplexExpRotaryEmbedding`（Ascend 专用 RoPE）

```
ops/rope_dsv4.py`, lines 104-237
```

**全局状态**（lines 13-21）：
```python
class RopeGlobalState:
    def __init__(self):
        self.static_cache: dict = {}           # cos/sin 静态缓存
        self.runtime_buffer: dict = {}          # 运行时 buffer
        self.layer_info: dict = {}              # 层 → 配置映射
        self.registry_summary: dict = {}        # 配置 → group 映射
```

**初始化**（`__init__`，lines 117-166）：
```python
class ComplexExpRotaryEmbedding(nn.Module):
    def __init__(self, vllm_config, layername, head_size, ...):
        # 1. 生成全局唯一的配置 key
        config_key = f"rotary_dim{rotary_dim}_base{base}_scaling_factor{scaling_factor}_..."
        
        # 2. 注册此层的配置信息
        _ROPE_STATE.layer_info[layername] = (config_key, rope_groups)
        
        # 3. 静态缓存：预计算所有位置的 cos/sin
        if config_key not in _ROPE_STATE.static_cache:
            inv_freq = self.precompute_freqs_cis(...)  # YaRN NTK scaling
            freqs = torch.einsum("i,j -> ij", t, inv_freq)
            cos = freqs.cos().repeat_interleave(2, dim=-1)
            sin = freqs.sin().repeat_interleave(2, dim=-1)
            _ROPE_STATE.static_cache[config_key] = (cos, sin)
        
        # 4. 运行时 buffer（避免每次 forward 重新分配）
        if config_key not in _ROPE_STATE.runtime_buffer:
            buf_cos = torch.ones(max_batch, 1, 1, rotary_dim, ...)
            buf_sin = torch.zeros(max_batch, 1, 1, rotary_dim, ...)
            _ROPE_STATE.runtime_buffer[config_key][grp] = (buf_cos, buf_sin)
```

**forward**（lines 217-234）：
```python
def forward(self, x, cos, sin):
    # 使用 torch_npu 硬件加速的 rotary_mul
    x = torch_npu.npu_rotary_mul(x, cos, sin, rotary_mode="interleave")
    return x
```

**`get_cos_and_sin_dsa`**（lines 63-101）：为 DSA 注意力提供 cos/sin 值
```python
def get_cos_and_sin_dsa(positions, use_cache=False):
    # 从静态查表获取当前 positions 对应的 cos/sin
    curr_cos = static_cos[pos_tensor]
    curr_sin = static_sin[pos_tensor]
    
    if use_cache:
        # 复制到运行时 buffer（避免分配新张量）
        buf_cos[:num_tokens].copy_(curr_cos)
        return (buf_cos, buf_sin)
    return (curr_cos, curr_sin)
```

**RoPE 功能总结**：
- 支持多组 RoPE（default + c4/c128），每组有自己的 cos/sin
- 通过 `RopeGlobalState` 全局共享缓存，避免重复计算
- 使用 `RopeDataProxy` 实现按层名或配置的矢量化索引
- 底层使用 `torch_npu.npu_rotary_mul` 硬件加速

---

### 4.12 详解：MTP（Multi-Token Prediction）—— Ascend 版本

**文件**：`models/deepseek_v4_mtp.py`

由于篇幅限制，这里仅概括与 vLLM NVIDIA 的差异：

| 维度 | NVIDIA MTP | Ascend MTP |
|------|-----------|------------|
| Decoder 复用 | `DeepseekV4DecoderLayer`（V4 特有） | `DeepseekV2DecoderLayer` + `DeepseekV4MoE` |
| SharedHead | 从 `vllm/model_executor/models/ds_mtp.py` 导入 | 自定义 `SharedHead`（含 RMSNorm） |
| HC Head | `hc_head_fused_kernel_tilelang` | 纯 PyTorch `hc_head` 方法 |
| Key 操作 | `fused_mtp_input_rmsnorm`（Triton） | 同样使用 `fused_mtp_input_rmsnorm` 但内部调用 NPU 算子 |
| `compute_logits` | `hc_head` + `mtp_shared_head_rmsnorm` | 类似逻辑，但 hc_head 是纯 PyTorch |
| 权重加载 | `_remap_weight_name` + `get_spec_layer_idx` | `get_spec_layer_idx_from_weight_name`（相同逻辑） |

---

## 五、核心实现对比

### 5.1 注意力机制（MLA）

| 维度 | vLLM（NVIDIA） | vLLM-Ascend |
|------|---------------|-------------|
| **实现位置** | `nvidia/flashmla.py` + `sparse_mla.py`（~800 行） | `ops/dsa.py`（272 行） |
| **后端选择** | 运行时可选 FlashMLA / FlashInfer | 编译时固定的 DSA |
| **类名** | `DeepseekV4FlashMLAAttention` | `AscendDeepseekSparseAttention` |
| **继承链** | ← `DeepseekV4Attention`（ABC） | ← `MultiHeadLatentAttentionWrapper` |
| **CUDA/NPU 算子** | FlashMLA kernel 或 FlashInfer TRTLLM-gen | `torch_npu` 原生算子 + DSA C++ 模块 |
| **KV cache 格式** | UE8M0 block-scaled FP8 (uint8) / bf16 | DSA 原生格式 |
| **W_Q_A + W_KV** | `fused_wqa_wkv` 合并 | `wq_a` + `wkv` 分离 |
| **Indexer Cache** | `DeepseekV4IndexerCache`（`attention.py`） | 直接 import vLLM 的 IndexerCache |

### 5.2 DecoderLayer 前向传播

| 特征 | vLLM（NVIDIA） | vLLM-Ascend |
|------|---------------|-------------|
| HC 算子 | `mhc_pre_tilelang`（tilelang 编译） | `npu_hc_pre` / `npu_hc_post`（Ascend 自定义算子） |
| Norm | attn_norm 融合在 mhc_pre 中 | 独立的 `input_layernorm` / `post_attention_layernorm` |
| 残差管理 | 三态 `(residual, post_mix, res_mix)` | 两态 `(residual, post+comb)` |
| 编译装饰器 | 无 | `@support_torch_compile` |
| 签名 | `(x, positions, input_ids, post_mix, res_mix, residual)` | `(positions, hidden_states, residual, llama_4_scaling)` |

### 5.3 MoE（混合专家层）

| 维度 | vLLM（NVIDIA） | vLLM-Ascend |
|------|---------------|-------------|
| MoE 实现 | `FusedMoE` + `DeepseekV4MegaMoEExperts` | `FusedMoE` + `mix_placement` |
| Shared Experts | 独立 `DeepseekV4MLP` | 支持 `mix_placement`（集成到 FusedMoE） |
| MegaMoE | `DeepseekV4MegaMoEExperts`（FP4, SM100 专用） | 无 |
| Gate 精度 | `GateLinear` + `out_dtype=torch.float32` | `ReplicatedLinear` + `precast_fp32_weight=True` |
| 专家量化 | FP4（`DeepseekV4MegaMoEExperts`）或 FP8 | FP8 泛化（FusedMoE） |
| Grouped topk | 隐式 | 显式 `use_grouped_topk=True` |

### 5.4 RoPE

| 维度 | vLLM | vLLM-Ascend |
|------|------|-------------|
| 文件 | `common/rope.py`（36 行） | `ops/rope_dsv4.py`（237 行） |
| 实现 | 辅助函数 `build_deepseek_v4_rope()` | 完整 `ComplexExpRotaryEmbedding` 类 |
| 缓存 | 无 | `RopeDataProxy` 支持静态 + 运行时缓存 |
| 硬件加速 | CUDA kernel 内部处理 | `torch_npu.npu_rotary_mul` |

### 5.5 HC Head

| 维度 | vLLM（NVIDIA） | vLLM-Ascend |
|------|---------------|-------------|
| 实现 | `hc_head_fused_kernel_tilelang`（tilelang） | 纯 PyTorch（`rsqrt` + `sigmoid` + `linear`） |

### 5.6 其他差异

| 维度 | vLLM | vLLM-Ascend |
|------|------|-------------|
| 量化 | `DeepseekV4FP8Config`（FP4/FP8） | 复用 vLLM 通用 FP8 |
| DSA Context Parallel | 无 | 独有 `enable_dsa_cp()` |
| 接口支持 | `SupportsPP` + `DeepseekV4MixtureOfExperts` | `SupportsPP` + `SupportsEagle` + `SupportsLoRA` |

---

## 六、总结

| 维度 | vLLM 策略 | vLLM-Ascend 策略 |
|------|-----------|-----------------|
| **代码组织** | 36 个文件，硬件隔离子目录 | 2 个主文件 + 2 个 ops，单体结构 |
| **注意力** | FlashMLA / FlashInfer（CUDA 生态） | AscendDeepseekSparseAttention（NPU 原生 DSA） |
| **Hybrid Chunk** | `mhc_pre_tilelang`（tilelang 编译，Norm 融合） | `npu_hc_pre` / `npu_hc_post`（Ascend 自定义算子，显式 Norm） |
| **MoE** | 标准 TP + DeepseekV4MegaMoE（SM100） | 标准 EP + mix_placement |
| **RoPE** | 36 行辅助函数 | 237 行完整类 + 缓存 + DSA 集成 |
| **HC Head** | `hc_head_fused_kernel_tilelang`（tilelang kernel） | 纯 PyTorch 实现 |
| **量化** | `DeepseekV4FP8Config`（FP4/FP8 专用配置） | 复用 vLLM 通用 FP8 |
| **接口** | SupportsPP + DeepseekV4MixtureOfExperts | SupportsPP + SupportsEagle + SupportsLoRA |
| **编译优化** | CUDA graph | `@support_torch_compile` + DSA custom op graph capture |

**一句话总结**：vLLM 的 DS V4 适配是**全栈自研 + CUDA 生态依赖**模式（FlashMLA、FlashInfer、cutedsl、tilelang），vLLM-Ascend 是**复用共享层 + 替换关键路径为 NPU 原生算子**模式——复用 vLLM 的 IndexerCache、Compressor、FusedMoE，替换注意力（DSA）、RoPE（ComplexExpRotaryEmbedding）、HC（npu_hc_pre/post）、HC Head（纯 PyTorch）为昇腾 NPU 实现。两者都适配了 DS V4 的核心架构（MLA + MoE + MTP + HC），只是底层硬件算子不同。

---

**文档版本**: v3.0  
**创建时间**: 2026-06-27  
**基于源码**: `vllm/vllm/models/deepseek_v4/` + `vllm-ascend/vllm_ascend/models/deepseek_v4.py` + `vllm-ascend/vllm_ascend/ops/dsa.py` + `vllm-ascend/vllm_ascend/ops/rope_dsv4.py`

---

## 附：文件结构清单

### vLLM DS V4 文件结构

| 目录/文件 | 行数 | 作用 |
|----------|------|------|
| **`__init__.py`** | 31 | 入口，按平台分发 |
| **`quant_config.py`** | 160 | FP4/FP8 专家量化配置 |
| **`attention.py`** | 800 | MLA 注意力基类 + IndexerCache |
| **`compressor.py`** | 399 | KV cache 压缩器（CompressorStateCache） |
| **`sparse_mla.py`** | 416 | 稀疏 MLA AttentionBackend |
| **`common/rope.py`** | 36 | RoPE 辅助函数 |
| **`common/ops/`**（8 个文件） | ~2,700 | 共享 Triton 算子 |
| **`nvidia/model.py`** | ~1,340 | NVIDIA 模型主文件 |
| **`nvidia/mtp.py`** | ~400 | NVIDIA MTP 实现 |
| **`nvidia/flashmla.py`** | ~320 | FlashMLA 注意力 |
| **`nvidia/flashinfer_sparse.py`** | ~80 | FlashInfer 稀疏注意力 |
| **`nvidia/ops/`**（6 个） | ~3,400 | NVIDIA 专用 CuteDSL 算子 |
| **`amd/model.py`** | 818 | AMD ROCm 模型实现 |
| **`amd/mtp.py`** | — | AMD MTP 实现 |
| **`xpu/model.py`** | ~1,370 | Intel XPU 模型实现 |
| **`xpu/mtp.py`** | — | Intel XPU MTP 实现 |
| **`model_executor/models/ds_mtp.py`** | 516 | MTP 模型注册与权重加载 |

### vLLM-Ascend DS V4 文件结构

| 文件 | 行数 | 作用 |
|------|------|------|
| **`models/deepseek_v4.py`** | 1,354 | 完整的模型实现（MLP, MoE, DecoderLayer, ForCausalLM） |
| **`models/deepseek_v4_mtp.py`** | 506 | MTP 实现（复用主模型层） |
| **`ops/dsa.py`** | 272 | Ascend Deepseek Sparse Attention |
| **`ops/rope_dsv4.py`** | 237 | DS V4 专用的 RoPE 实现 |

---

## 六、总结

| 维度 | vLLM 策略 | vLLM-Ascend 策略 |
|------|-----------|-----------------|
| **代码规模** | 36 个文件，分散的子目录结构 | 2+2 个文件，单体结构 |
| **注意力实现** | FlashMLA / FlashInfer（CUDA 生态） | AscendDeepseekSparseAttention（NPU 原生 DSA） |
| **Hybrid Chunk** | `mhc_pre_tilelang`（tilelang 编译） | `npu_hc_pre` / `npu_hc_post`（Ascend 自定义算子） |
| **MoE** | 标准 TP + DeepseekV4MegaMoE（SM100） | 标准 TP + mix_placement |
| **RoPE** | 36 行辅助函数 | 237 行完整类 + 缓存 + DSA 集成 |
| **量化** | DeepseekV4FP8Config（FP4/FP8 专家） | 复用 vLLM 通用 FP8 |
| **接口支持** | SupportsPP + DeepseekV4MixtureOfExperts | SupportsPP + SupportsEagle + SupportsLoRA |
| **代码风格** | 硬件隔离子目录（36 文件） | 单体文件 + 少量自定义 ops（2 文件 + 2 ops） |

**一句话总结**：vLLM 的 DS V4 适配是**全栈自研 + CUDA 生态依赖**模式（FlashMLA、FlashInfer、cutedsl、tilelang），vLLM-Ascend 是**复用共享层 + 替换关键路径为 NPU 原生算子**模式——复用 vLLM 的 IndexerCache、Compressor、FusedMoE，替换注意力（DSA）、RoPE（ComplexExpRotaryEmbedding）、HC（npu_hc_pre/post）为昇腾 NPU 实现。两者都适配了 DS V4 的核心架构（MLA + MoE + MTP + HC），只是底层硬件算子不同。

---

**文档版本**: v3.0  
**创建时间**: 2026-06-27  
**基于源码**: `vllm/vllm/models/deepseek_v4/` + `vllm-ascend/vllm_ascend/models/deepseek_v4.py` + `vllm-ascend/vllm_ascend/ops/dsa.py` + `vllm-ascend/vllm_ascend/ops/rope_dsv4.py`