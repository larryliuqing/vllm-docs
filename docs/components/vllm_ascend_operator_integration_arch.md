# vLLM-Ascend 算子集成架构详解

> 本文档详细解答 vLLM 与 vLLM-Ascend 之间的算子调用关系，阐述 vLLM-Ascend 如何通过 **Patch 机制**、**PluggableLayer/CustomOp OOT 注册机制** 来替换 vLLM 中的算子，实现昇腾 NPU 适配。

---

## 1. 核心结论

**Q: vLLM 中的算子是否都需要在 vLLM-Ascend 中重新实现？**

**A: 不是！主要通过 4 种机制实现适配：**

| 机制 | 说明 | 替换方式 | 数量 |
|------|------|---------|------|
| **CustomOp.register_oot** | 继承上游类，替换 `forward_oot` | 类级别替换 | 30+ 算子 |
| **Patch 机制** | Monkey Patch 直接替换 | 函数/方法级别替换 | 48 个 Patch |
| **direct_register_custom_op** | 注册 `torch.ops.vllm.xxx` 原始算子 | 算子级新增 | 10+ 个 |
| **量化方法继承** | 继承 `LinearMethodBase` / `FusedMoEMethodBase` | 配置驱动 | 15+ 个 |

**Q: vLLM 如何调用 vLLM-Ascend 中的算子？**

**A: 对上游代码完全透明**：通过 PluggableLayer/CustomOp 的 `__new__` 方法自动返回 Ascend 替换类，vLLM 代码无需任何修改。

---

## 2. CustomOp OOT 注册机制（核心）

### 2.1 原理：`__new__` 替换

vLLM 的 `CustomOp` 和 `PluggableLayer` 基类在 `__new__` 中拦截实例化，将上游类透明替换为 OOT 注册的 Ascend 类：

```python
# vllm/model_executor/custom_op.py (上游 vLLM)

class CustomOp(nn.Module):
    def __new__(cls, *args, **kwargs):
        op_name = cls.__name__

        # 检查是否有 OOT 替换
        if op_name in op_registry_oot:
            return super().__new__(op_registry_oot[op_name])
        return super().__new__(cls)

    def forward(self, *args, **kwargs):
        return self._forward_method(*args, **kwargs)

    def forward_cuda(self, *args, **kwargs):
        raise NotImplementedError

    def forward_oot(self, *args, **kwargs):
        return self.forward_native(*args, **kwargs)  # CPU 回退
```

**工作流程**：当 vLLM 代码执行 `layer = SiluAndMul()` 时：

```
1. SiluAndMul.__new__(SiluAndMul, ...)
2. op_registry_oot 中存在 "SiluAndMul" → AscendSiluAndMul
3. 返回 AscendSiluAndMul 实例（不是 SiluAndMul 实例）
4. forward() 自动路由到 forward_oot()
```

### 2.2 统一注册入口

所有 OOT 注册集中在 `vllm_ascend/utils.py` 的 `register_ascend_customop()` 函数中：

```python
# vllm_ascend/utils.py

_ASCEND_CUSTOMOP_IS_REIGISTERED = False

def register_ascend_customop():
    global _ASCEND_CUSTOMOP_IS_REIGISTERED
    if _ASCEND_CUSTOMOP_IS_REIGISTERED:
        return

    # 1. 导入所有 Ascend 算子类（触发类定义）
    import vllm_ascend.ops.activation        # 定义 AscendSiluAndMul 等
    import vllm_ascend.ops.layernorm         # 定义 AscendRMSNorm 等
    import vllm_ascend.ops.rotary_embedding  # 定义 AscendRotaryEmbedding 等
    import vllm_ascend.ops.linear            # 定义 AscendLinearBase 等
    import vllm_ascend.ops.fused_moe         # 定义 AscendFusedMoE
    import vllm_ascend.ops.mla               # 定义 AscendMultiHeadLatentAttention
    import vllm_ascend.ops.dsa               # 定义 AscendDeepseekSparseAttention
    ...

    # 2. 构建注册映射表
    REGISTERED_ASCEND_OPS = {
        # 激活函数
        "QuickGELU": AscendQuickGELU,
        "SiluAndMul": AscendSiluAndMul,
        # LayerNorm
        "RMSNorm": AscendRMSNorm,
        "GemmaRMSNorm": AscendGemmaRMSNorm,
        "RMSNormGated": AscendRMSNormGated,
        # RoPE
        "RotaryEmbedding": AscendRotaryEmbedding,
        "MRotaryEmbedding": AscendMRotaryEmbedding,
        "YaRNScalingRotaryEmbedding": AscendYaRNRotaryEmbedding,
        "DeepseekScalingRotaryEmbedding": AscendDeepseekScalingRotaryEmbedding,
        "ApplyRotaryEmb": AscendApplyRotaryEmb,
        # Linear
        "ColumnParallelLinear": AscendColumnParallelLinear,
        "RowParallelLinear": AscendRowParallelLinear,
        "MergedColumnParallelLinear": AscendMergedColumnParallelLinear,
        "QKVParallelLinear": AscendQKVParallelLinear,
        "ReplicatedLinear": AscendReplicatedLinear,
        # MoE
        "FusedMoE": AscendFusedMoE,
        # Embedding
        "VocabParallelEmbedding": AscendVocabParallelEmbedding,
        "ParallelLMHead": AscendParallelLMHead,
        "LogitsProcessor": AscendLogitsProcessor,
        # Attention
        "MultiHeadLatentAttentionWrapper": AscendMultiHeadLatentAttention,
        "GatedDeltaNetAttention": AscendGatedDeltaNetAttention,
        "BailingMoELinearAttention": AscendBailingMoELinearAttention,
        "Conv3dLayer": AscendConv3dLayer,
        "MMEncoderAttention": AscendMMEncoderAttention,
        "RelPosAttention": AscendRelPosAttention,
        "CustomQwen2Decoder": AscendCustomQwen2Decoder,
    }

    # 3. 条件注册（DS MLA 模型）
    if config.use_mla:
        REGISTERED_ASCEND_OPS["GateLinear"] = AscendGateLinear

    # 4. 310P 设备替换（覆盖部分注册）
    if is_310p_device():
        REGISTERED_ASCEND_OPS.update({
            "RMSNorm": AscendRMSNorm310,
            "RotaryEmbedding": AscendRotaryEmbedding310,
            ...
        })

    # 5. 执行注册
    for name, op_cls in REGISTERED_ASCEND_OPS.items():
        CustomOp.register_oot(_decorated_op_cls=op_cls, name=name)

    _ASCEND_CUSTOMOP_IS_REIGISTERED = True
```

---

## 3. 算子适配详解（按业务分类）

### 3.1 激活函数（Activation）

**替换模式**：`CustomOp.register_oot` — 继承上游类，重写 `forward_oot`

```python
# vllm_ascend/ops/activation.py

class AscendQuickGELU(QuickGELU):
    """替换 vLLM 的 QuickGELU
    GPU 实现: x * sigmoid(1.702 * x)
    NPU 实现: npu_fast_gelu（硬件加速）
    """
    def forward_oot(self, x):
        return torch_npu.npu_fast_gelu(x)


class AscendSiluAndMul(SiluAndMul):
    """替换 vLLM 的 SiluAndMul
    GPU 实现: silu(gate) * up
    NPU 实现: npu_swiglu（硬件融合算子）
    """
    def forward_oot(self, x):
        # 一次调用完成 silu(gate) * up 的融合计算
        return torch_npu.npu_swiglu(x)
```

**调用链路**：
```
vLLM 模型代码: self.act_fn = SiluAndMul()
    → SiluAndMul.__new__() 发现 OOT 注册
    → 实际实例化 AscendSiluAndMul
    → forward() → forward_oot() → torch_npu.npu_swiglu(x)
```

---

### 3.2 LayerNorm

**替换模式**：`CustomOp.register_oot` — 继承上游类，重写 `forward_oot`

```python
# vllm_ascend/ops/layernorm.py

class AscendRMSNorm(RMSNorm):
    """替换 vLLM 的 RMSNorm
    GPU 实现: x * rsqrt(mean(x^2) + eps) * weight
    NPU 实现: 使用 npu_rms_norm 硬件算子
    """
    def forward_oot(self, x, residual=None):
        if residual is not None:
            # 融合 add + rms_norm（减少一次 kernel launch）
            y, _ = torch.ops._C_ascend.npu_add_rms_norm_bias(
                x, residual, self.weight, None, self.variance_epsilon
            )
            return y
        # 单 RMSNorm
        return torch_npu.npu_rms_norm(x, self.weight, self.variance_epsilon)[0]


class AscendGemmaRMSNorm(GemmaRMSNorm):
    """替换 vLLM 的 GemmaRMSNorm（Gemma 使用 1 + weight 的变体）"""
    def forward_oot(self, x, residual=None):
        # Gemma 特殊处理：weight 加 1 偏置
        return torch.ops._C_ascend.npu_gemma_rms_norm(
            x, self.weight, self.variance_epsilon
        )
```

---

### 3.3 Rotary Embedding（RoPE）

**替换模式**：`CustomOp.register_oot` + `direct_register_custom_op`（底层算子）

```python
# vllm_ascend/ops/rotary_embedding.py

class AscendRotaryEmbedding(RotaryEmbedding):
    """替换 vLLM 的 RotaryEmbedding
    GPU: cos/sin 查表 → 逐元素乘加
    NPU: 融合算子 torch.ops.vllm.npu_rotary_embedding
    """
    def forward_oot(self, positions, query, key, offsets=None):
        # 调用注册的原始算子
        return torch.ops.vllm.npu_rotary_embedding(
            positions, query, key, self.head_size,
            self.cos_sin_cache, self.is_neox_style,
        )


class AscendMRotaryEmbedding(MRotaryEmbedding):
    """替换 vLLM 的 MRotaryEmbedding（多模态 RoPE，Qwen2-VL 使用）
    多模态 RoPE 需要处理 3D 位置编码（temporal + spatial）
    """
    def forward_oot(self, positions, query, key, offsets=None):
        # 使用 NPU 融合的多模态 RoPE Triton 算子
        return torch.ops.vllm.triton_split_qkv_rmsnorm_mrope(
            qkv=qkv, q_weight=self.q_norm.weight,
            k_weight=self.k_norm.weight, cos_sin=cos_sin,
            num_q_heads=self.num_heads, num_kv_heads=self.num_kv_heads,
            head_size=self.head_dim, eps=self.q_norm.variance_epsilon,
            mrope_section=self.rotary_emb.mrope_section,
            ...
        )


class AscendDeepseekScalingRotaryEmbedding(DeepseekScalingRotaryEmbedding):
    """替换 vLLM 的 DeepseekScalingRotaryEmbedding（DS V3/V4 使用）
    注意：这里重写的是 forward 而非 forward_oot
    因为 DeepSeek 的 RoPE 有特殊的 Yarn scaling 逻辑
    """
    def forward(self, positions, query, key, offsets=None):
        # 计算 DS 特有的缩放因子
        # 使用 npu_rotary_embedding 硬件加速
        return torch.ops.vllm.npu_rotary_embedding(...)
```

**底层算子注册**：

```python
# vllm_ascend/ops/register_custom_ops.py
from vllm.utils.torch_utils import direct_register_custom_op

direct_register_custom_op(
    op_name="npu_rotary_embedding",
    op_func=rope_forward_oot,
    fake_impl=_rope_forward_oot_impl_fake,
    dispatch_key="PrivateUse1",  # NPU 设备分发键
)
```

---

### 3.4 线性层（Linear）

**替换模式**：`CustomOp.register_oot` — 替换所有线性层变体

```python
# vllm_ascend/ops/linear.py

class AscendColumnParallelLinear(ColumnParallelLinear):
    """替换 vLLM 的 ColumnParallelLinear
    支持：自定义 TP group、FlashComm2 O-shard、MatmulAllReduce 融合
    """
    def __init__(self, ...):
        super().__init__(...)
        # 根据配置选择 custom_op（不同的前向策略）
        self.custom_op = get_parallel_op(
            "column", self.quant_method, input_size,
            self.world_size, prefix=self.prefix,
        )

    def forward(self, input_):
        if self.custom_op:
            # 使用自定义的前向策略
            return self.custom_op.apply(input_, self.weight, ...)
        return super().forward(input_)


class AscendQKVParallelLinear(QKVParallelLinear):
    """替换 vLLM 的 QKVParallelLinear
    QKV 投影在 TP 下有特殊的分片需求
    """
    ...

class AscendRowParallelLinear(RowParallelLinear):
    """替换 vLLM 的 RowParallelLinear"""
    ...

class AscendMergedColumnParallelLinear(MergedColumnParallelLinear):
    """替换 vLLM 的 MergedColumnParallelLinear（gate_up_proj 使用）"""
    ...
```

**自定义前向策略**（`linear_op.py`）：

```python
# vllm_ascend/ops/linear_op.py

# 根据不同前缀选择不同的 TP 通信策略
class MLPColumnParallelOp(CustomColumnParallelOp):
    """MLP 的 column parallel：gate_up_proj 使用"""
    ...

class OProjRowParallelOp(CustomRowParallelOp):
    """output_proj 的 row parallel：使用自定义 TP group"""
    ...

class Flashcomm2OshardQKVParallelOp(CustomColumnParallelOp):
    """FlashComm2 的 QKV parallel：O-shard 切分"""
    ...

class MatmulAllreduceRowParallelOp(CustomRowParallelOp):
    """MatmulAllReduce 融合：matmul 和 all-reduce 合并"""
    ...
```

---

### 3.5 Fused MoE

**替换模式**：`CustomOp.register_oot` — 替换 vLLM 的 `FusedMoE` 类

```python
# vllm_ascend/ops/fused_moe/fused_moe.py

class AscendFusedMoE(FusedMoE):
    """替换 vLLM 的 FusedMoE
    支持：
    - 多种通信方法（AllGather, MC2, All2All）
    - 权重预取（weight prefetch 减少 HBM 延迟）
    - 共享专家 DP 复制
    - 各种量化方法
    """

    def __init__(self, ...):
        super().__init__(...)
        # 选择 MoE 通信方法
        self.comm_method = get_moe_comm_method(...)
        # 初始化权重预取
        self.weight_prefetch = WeightPrefetchMethod(...)


class AscendMoERunner(MoERunner):
    """替换 vLLM 的 MoERunner
    关键差异：
    - use_dp_chunking: DP chunk 支持
    - _fused_output_is_reduced: fused output 减少模式
    - _maybe_reduce_shared_expert_output: 共享专家输出减少
    """
    ...

    def _fused_output_is_reduced(self, hidden_states, ...):
        """NPU 上 fused output 在 chunk 级别完成 reduce"""
        ...


class AscendGateLinear(GateLinear):
    """替换 vLLM 的 GateLinear（路由器门控）
    在 DeepSeek MLA 模型中，路由器需要在 fp32 精度下计算
    """
    def forward_oot(self, x):
        # 强制 fp32 精度
        x = x.float()
        return torch.ops.vllm.npu_gate_forward(x, self.weight, ...)
```

**通信方法选择**（`moe_comm_method.py`）：

```python
def get_moe_comm_method(comm_method: str, ...):
    methods = {
        "MC2": MC2Method,          # Matmul + Collective 通信优化
        "All2All": All2AllMethod,  # All-to-All（专家并行）
        "AllGather": AllGatherMethod,  # AllGather（默认）
    }
    return methods[comm_method](...)
```

---

### 3.6 Attention 后端

**替换模式**：`CustomOp.register_oot` + `direct_register_custom_op`（原始算子）

```python
# vllm_ascend/ops/mla.py

class AscendMultiHeadLatentAttention(MultiHeadLatentAttentionWrapper):
    """替换 vLLM 的 MultiHeadLatentAttentionWrapper
    MLA（Multi-Head Latent Attention）：
    - DeepSeek V2/V3 使用的 KV 压缩 Attention
    - 将 KV 压缩到低维 latent space
    """

    def forward(self, ...):
        # 调用注册的 MLA 原始算子
        torch.ops.vllm.mla_forward(
            query, key_rope, value, ...,
            output=output,
        )
        return output


# 底层算子注册
direct_register_custom_op(
    op_name="mla_forward",
    op_func=mla_forward,          # NPU 实现
    mutates_args=["output"],      # output in-place 修改
    fake_impl=mla_forward_fake,   # 图捕获时的 fake 实现
    dispatch_key="PrivateUse1",   # NPU 分发键
)
```

```python
# vllm_ascend/ops/dsa.py

class AscendDeepseekSparseAttention(MultiHeadLatentAttentionWrapper):
    """替换 vLLM 的 DeepseekSparseAttention
    DSA（DS Attention）：
    - DS V4 的动态稀疏 Attention
    - Compressor 压缩 KV Cache
    - 稀疏选择机制
    """

    def forward(self, ...):
        torch.ops.vllm.dsa_forward(
            query, key, value, ...,
            output=output,
        )
        return output


direct_register_custom_op(
    op_name="dsa_forward",
    op_func=dsa_forward,
    mutates_args=["output"],
    fake_impl=dsa_forward_fake,
    dispatch_key="PrivateUse1",
)
```

---

### 3.7 Embedding 与 LM Head

**替换模式**：`CustomOp.register_oot`

```python
# vllm_ascend/ops/vocab_parallel_embedding.py

class AscendVocabParallelEmbedding(VocabParallelEmbedding):
    """替换 vLLM 的 VocabParallelEmbedding
    支持自定义 TP group（embed_tp, lmhead_tp）
    """
    def forward(self, input_):
        if self.tp_size > 1:
            # TP 切分下的 embedding 查找
            input_ = input_.view(-1)
            masked_input = input_ - self.tp_rank * self.num_embeddings_per_partition
            mask = (input_ >= self.tp_rank * self.num_embeddings_per_partition) & \
                   (input_ < (self.tp_rank + 1) * self.num_embeddings_per_partition)
            masked_input = torch.where(mask, masked_input, torch.zeros_like(masked_input))
            output = torch.nn.functional.embedding(masked_input, self.weight, ...)
            # All-Reduce 同步
            output = tensor_model_parallel_all_reduce(output)
            return output
        return super().forward(input_)


class AscendParallelLMHead(ParallelLMHead):
    """替换 vLLM 的 ParallelLMHead（语言模型头）
    继承自 AscendVocabParallelEmbedding
    """
    ...


class AscendLogitsProcessor(LogitsProcessor):
    """替换 vLLM 的 LogitsProcessor
    支持自定义 TP group 和 NPU 优化
    """
    ...
```

---

### 3.8 GDN（Gated Delta Net）

**替换模式**：`CustomOp.register_oot`

```python
# vllm_ascend/ops/gdn.py

class AscendGatedDeltaNetAttention(GatedDeltaNetAttention):
    """替换 vLLM 的 GatedDeltaNetAttention
    Qwen3.5 使用的 Gated Delta Net Attention：
    - 线性 Attention 变体
    - 使用 NPU Triton 算子加速 chunk 计算
    - 融合 gating 操作
    """

    def forward(self, ...):
        # 1. NPU 优化的 causal conv1d
        k = causal_conv1d_fn(k, ...)

        # 2. NPU Triton chunked GDN 计算
        o = chunk_gdn(q, k, v, ...)

        # 3. 融合 gating
        o = fused_gdn_gating(o, gating, ...)
        return o
```

---

### 3.9 Triton 算子

Triton 算子不属于 `CustomOp` 体系。它们通过两种方式使用：

**方式1：通过 Patch 替换 vLLM 的 Triton 算子调用**

```python
# vllm_ascend/patch/worker/patch_triton.py

import vllm.model_executor.layers.mamba.ops.causal_conv1d
from vllm_ascend.ops.triton.mamba.causal_conv1d import causal_conv1d_fn

# 用 Ascend Triton 实现替换 vLLM 的 CUDA Triton 实现
vllm.model_executor.layers.mamba.ops.causal_conv1d.causal_conv1d_fn = causal_conv1d_fn
```

**方式2：通过 `direct_register_custom_op` 注册为原始算子**

```python
# vllm_ascend/ops/register_custom_ops.py

# Triton fused multiply-add → 注册为 torch.ops.vllm.muls_add
from vllm_ascend.ops.triton.muls_add import muls_add_triton

direct_register_custom_op(
    op_name="muls_add",
    op_func=muls_add_triton,
    fake_impl=_muls_add_impl_fake,
    dispatch_key="PrivateUse1",
)
```

**Triton 算子目录结构**：

```
vllm_ascend/ops/triton/
├── mamba/
│   ├── causal_conv1d.py     # Mamba 因果卷积 1D（通过 Patch 替换）
│   └── lightning_attn.py    # Lightning Attention
├── fla/                      # Flash Linear Attention（GDN 使用）
│   ├── chunk.py             # 核心 chunk 计算
│   ├── chunk_o.py           # chunk output
│   ├── chunk_delta_h.py     # chunk delta hidden state
│   ├── wy_fast.py           # WY 快速变换
│   └── ...
├── spec_decode/
│   └── utils.py             # 投机解码辅助
├── linearnorm/              # 融合 QKV + RMSNorm + RoPE（注册为 torch.ops.vllm）
│   ├── split_qkv_rmsnorm_rope.py
│   ├── split_qkv_rmsnorm_mrope.py
│   └── split_qkv_tp_rmsnorm_rope.py
├── batch_invariant/         # 图捕获兼容的实现
│   ├── matmul.py
│   ├── rmsnorm.py
│   └── softmax.py
├── activation/
│   └── swiglu_quant.py      # SwiGLU + 量化
├── reject_sample.py         # 拒绝采样 NPU Triton
├── penalty.py               # 惩罚项 NPU Triton
├── rms_norm.py              # RMSNorm NPU Triton
└── rope.py                  # RoPE NPU Triton
```

---

### 3.10 量化方法

量化方法不通过 `CustomOp.register_oot` 注册，而是通过继承 `LinearMethodBase` / `FusedMoEMethodBase` 并由配置驱动选择：

```python
# vllm_ascend/quantization/methods/w4a8.py

class AscendW4A8DynamicLinearMethod(LinearMethodBase):
    """W4A8 动态量化线性层方法
    - 权重：FP4 量化
    - 激活：INT8 动态量化
    - 使用 NPU 硬件加速的矩阵乘法
    """
    def create_weights(self, layer, ...):
        # 创建 W4A8 格式的权重
        ...

    def apply(self, weights, x, bias=None):
        # NPU 硬件加速的 W4A8 矩阵乘法
        return torch.ops.vllm.w4a8_gemm(x, weights, ...)
```

支持的所有量化方法：

| 方法 | 文件 | 说明 |
|------|------|------|
| W8A8 Dynamic | `methods/w8a8_dynamic.py` | INT8 动态量化 |
| W8A8 Static | `methods/w8a8_static.py` | INT8 静态量化 |
| W8A8 MXFP8 | `methods/w8a8_mxfp8.py` | MXFP8 格式 |
| W8A16 | `methods/w8a16.py` | 权重 INT8，激活 FP16 |
| W4A16 | `methods/w4a16.py` | 权重 INT4，激活 FP16 |
| **W4A8** | `methods/w4a8.py` | **权重 FP4，激活 INT8（DS V4 Flash 使用）** |
| W4A4 MXFP4 | `methods/w4a4_mxfp4.py` | MXFP4 格式 |
| W4A4 FlatQuant | `methods/w4a4_flatquant.py` | FlatQuant 量化 |
| W8A8 PDMix | `methods/w8a8_pdmix.py` | PD Mix 量化 |
| KV C8 | `methods/kv_c8.py` | KV Cache C8 量化 |

---

### 3.11 原始算子（direct_register_custom_op）

`register_custom_ops.py` 注册的 10 个原始算子：

```python
# vllm_ascend/ops/register_custom_ops.py

# 序列并行（SP）相关
direct_register_custom_op("maybe_chunk_residual", ...)
direct_register_custom_op("maybe_all_gather_and_maybe_unpad", ...)
direct_register_custom_op("maybe_pad_and_reduce", ...)

# 权重预取
direct_register_custom_op("prefetch_preprocess", ...)
direct_register_custom_op("prefetch_postprocess", ...)

# 通信优化
direct_register_custom_op("maybe_all_reduce_tensor_model_parallel", ...)
direct_register_custom_op("matmul_and_reduce", ...)  # Matmul + Reduce 融合

# 量化
direct_register_custom_op("quantize", ...)

# 计算融合
direct_register_custom_op("npu_rotary_embedding", ...)
direct_register_custom_op("muls_add", ...)  # fused multiply-add

# MLA / DSA 相关（在各自的文件中注册）
direct_register_custom_op("mla_forward", ...)   # ops/mla.py
direct_register_custom_op("dsa_forward", ...)    # ops/dsa.py

# 融合线性层（在 linearnorm/ 中注册）
direct_register_custom_op("split_qkv_rmsnorm_rope", ...)    # triton/linearnorm/
direct_register_custom_op("split_qkv_rmsnorm_mrope", ...)   # triton/linearnorm/
direct_register_custom_op("split_qkv_tp_rmsnorm_rope", ...) # triton/linearnorm/
```

所有原始算子使用 `dispatch_key="PrivateUse1"`，确保只在 NPU 设备上生效，并都提供了 `fake_impl` 用于图捕获。

---

## 4. 总结

### 算子适配全景图

```
上游 vLLM 算子
    │
    ├─ 激活函数: SiluAndMul, QuickGELU
    │   └─ OOT → npu_swiglu, npu_fast_gelu
    │
    ├─ LayerNorm: RMSNorm, GemmaRMSNorm
    │   └─ OOT → npu_rms_norm, npu_add_rms_norm
    │
    ├─ RoPE: RotaryEmbedding, MRotaryEmbedding, DeepseekScalingRotaryEmbedding
    │   └─ OOT + CustomOp → npu_rotary_embedding, triton_mrope
    │
    ├─ Linear: ColumnParallel, RowParallel, QKVParallel, MergedColumnParallel
    │   └─ OOT → 自定义 TP group + FlashComm2/MatmulAllReduce
    │
    ├─ MoE: FusedMoE, GateLinear
    │   └─ OOT → MC2/All2All + weight prefetch
    │
    ├─ Attention: MLA, DSA, GDN, BailingMoE, MMEncoder, RelPos
    │   └─ OOT + CustomOp → mla_forward, dsa_forward, triton GDN
    │
    ├─ Embedding: VocabParallelEmbedding, ParallelLMHead, LogitsProcessor
    │   └─ OOT → 自定义 TP group
    │
    ├─ Quantization: W4A8, W8A8, W4A4, ...
    │   └─ MethodBase 继承 → 配置驱动
    │
    └─ Raw Ops: SP, prefetch, matmul+reduce, fused ops
        └─ direct_register_custom_op → torch.ops.vllm.xxx
```

### 关键设计原则

1. **透明替换**：`CustomOp.__new__` 拦截实例化，上游代码零修改
2. **融合优化**：NPU 上减少 kernel launch（如 QKV split + QK-Norm + RoPE 融合为一个 Triton 算子）
3. **硬件加速**：优先使用 `torch_npu.npu_xxx` 硬件算子，其次使用 Triton，最后 CPU 回退
4. **图捕获兼容**：所有 `direct_register_custom_op` 都提供 `fake_impl`
5. **量化扩展**：通过 `LinearMethodBase` 体系支持 10+ 种量化方法

---

*文档版本：v2.0*
*创建日期：2026-06-27*
*基于 vLLM-Ascend 源码分析*