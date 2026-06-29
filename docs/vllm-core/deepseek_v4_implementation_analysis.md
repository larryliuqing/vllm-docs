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

### 3.1 修改文件一览

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

## 四、核心实现对比（vLLM vs vLLM-Ascend）

### 4.1 注意力机制（MLA）

| 维度 | vLLM（NVIDIA） | vLLM-Ascend |
|------|---------------|-------------|
| **实现位置** | `nvidia/flashmla.py` + `sparse_mla.py`（~800 行） | `ops/dsa.py`（272 行） |
| **后端选择** | 运行时可选 FlashMLA / FlashInfer | 编译时固定的 DSA |
| **类名** | `DeepseekV4FlashMLAAttention` | `AscendDeepseekSparseAttention` |
| **继承链** | ← `DeepseekV4Attention`（ABC） | ← `MultiHeadLatentAttentionWrapper` |
| **CUDA/NPU 算子** | FlashMLA kernel 或 FlashInfer TRTLLM-gen | `torch_npu` 原生算子 + DSA 模块 |
| **KV cache 格式** | UE8M0 block-scaled FP8 (uint8) / bf16 | DSA 原生格式 |
| **Indexer Cache** | `DeepseekV4IndexerCache`（在 `attention.py`） | 直接 import vLLM 的 IndexerCache |

**vLLM 的注意力选择逻辑**：
```python
# nvidia/model.py:726
def _select_dsv4_attn_cls(vllm_config):
    if backend == AttentionBackendEnum.FLASHINFER_MLA_SPARSE_DSV4:
        return DeepseekV4FlashInferMLAAttention  # FlashInfer TRTLLM-gen 路径
    return DeepseekV4FlashMLAAttention           # FlashMLA 路径（默认）
```

**vLLM-Ascend 的集成方式**：
```python
# ds_v4.py
dsa_modules = DSAModules(
    rope_emb=self.rope_emb,
    indexer_cache=self.indexer_cache,
)
self.dsa_attn = AscendDeepseekSparseAttention(
    config=config,
    dsa_modules=dsa_modules,
)
# forward 时直接调用
hidden_states = self.dsa_attn(hidden_states=hidden_states, positions=positions, ...)
```

### 4.2 DecoderLayer 前向传播

**vLLM（NVIDIA）**：使用 `tilelang` 实现的 `mhc_pre_tilelang` / `mhc_fused_post_pre_tilelang`
```python
# 第一层：独立 mhc_pre → 后续层：融合 mhc_fused_post_pre
# attn_norm 已融合在 mhc_pre/mhc_fused_post_pre 中
# forward 签名: (x, positions, input_ids, post_mix, res_mix, residual)
```

**vLLM-Ascend**：使用 `torch.ops._C_ascend.npu_hc_pre` / `npu_hc_post`
```python
# 显式的 input_layernorm + post_attention_layernorm（未融合）
# forward 签名: (positions, hidden_states, residual, llama_4_scaling)
```

| 特征 | vLLM（NVIDIA） | vLLM-Ascend |
|------|---------------|-------------|
| HC 算子 | `mhc_pre_tilelang`（tilelang 编译） | `npu_hc_pre` / `npu_hc_post`（Ascend 自定义算子） |
| Norm 融合 | attn_norm 融合在 mhc_pre 中 | 独立的 RMSNorm layer |
| 残差管理 | 三态 `(residual, post_mix, res_mix)` | 两态 `(residual, post + comb)` |
| 编译装饰器 | 无 | `@support_torch_compile` |
| 签名参数 | 6 参数（含 post_mix/res_mix） | 4 参数（更简洁） |

### 4.3 MoE（混合专家层）

| 维度 | vLLM（NVIDIA） | vLLM-Ascend |
|------|---------------|-------------|
| MoE 实现 | `FusedMoE` + `DeepseekV4MegaMoEExperts` | `FusedMoE` + `mix_placement` |
| Shared Experts | 标准实现（DeepseekV4MLP） | 支持 `mix_placement`（混合放置） |
| MegaMoE | `DeepseekV4MegaMoEExperts`（FP4, SM100 专用） | 无 |
| 专家并行 | 支持 EPLB | 支持 EPLB |

**Ascend 独有的 `mix_placement`**：
```python
self.is_fusion_moe_shared_experts_enabled = getattr(
    get_ascend_config(), "mix_placement", False
)
if config.n_shared_experts is None or self.is_fusion_moe_shared_experts_enabled:
    self.shared_experts = None  # 由 FusedMoE 统一管理
```

### 4.4 MTP（Multi-Token Prediction）

| 维度 | vLLM（NVIDIA） | vLLM-Ascend |
|------|---------------|-------------|
| 实现文件 | `nvidia/mtp.py` + `model_executor/models/ds_mtp.py` | `ds_v4_mtp.py` |
| 类名 | `DSV4MTP` | `DeepSeekV4MTP` |
| 注册名 | — | `DeepSeekV4MTPModel` |
| SharedHead | 使用 `SharedHead`（从 `ds_mtp.py` 引入） | 自定义 `SharedHead`（含 RMSNorm） |
| Decoder 复用 | `DeepseekV4DecoderLayer` | `DeepseekV2DecoderLayer` + `DeepseekV4MoE` |
| 关键操作 | `fused_mtp_input_rmsnorm`（Triton 算子） | 类似逻辑（复用 vLLM 版本） |

### 4.5 RoPE（旋转位置编码）

| 维度 | vLLM | vLLM-Ascend |
|------|------|-------------|
| 文件 | `common/rope.py`（36 行） | `ops/rope_dsv4.py`（237 行） |
| 功能 | 辅助函数 `build_deepseek_v4_rope()` | 完整 `ComplexExpRotaryEmbedding` 类 |
| 缓存 | 无 | `RopeDataProxy` 支持 cos/sin 缓存 |
| 与注意力集成 | FlashMLA 内部处理 | DSA 专用的 `get_cos_and_sin_dsa()` |

### 4.6 其他差异

| 维度 | vLLM | vLLM-Ascend |
|------|------|-------------|
| 量化 | `DeepseekV4FP8Config`（FP4/FP8） | 复用 vLLM 通用 FP8 |
| DSA Context Parallel | 无 | 独有 `enable_dsa_cp()` |
| 量化路由 | `config.py` 中自动重写 `quant_method` | 无特殊路由 |
| 权重映射器 | `_make_deepseek_v4_weights_mapper(expert_dtype)` | 标准 `default_weight_loader` |

---

## 五、详细文件映射

### 5.1 vLLM DS V4 文件结构

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

### 5.2 vLLM-Ascend DS V4 文件结构

| 文件 | 行数 | 作用 |
|------|------|------|
| **`models/ds_v4.py`** | 1,354 | 完整的模型实现（MLP, MoE, DecoderLayer, ForCausalLM） |
| **`models/ds_v4_mtp.py`** | 506 | MTP 实现（复用主模型层） |
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

**文档版本**: v2.0  
**创建时间**: 2026-06-27  
**基于源码**: `vllm/vllm/models/deepseek_v4/` + `vllm-ascend/vllm_ascend/models/ds_v4.py`