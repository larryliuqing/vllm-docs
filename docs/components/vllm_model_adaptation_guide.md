# vLLM模型适配指南

## 概述

本文档详细介绍如何将一个新模型适配到vLLM，包括适配原理、详细步骤以及在Ascend NPU上的特殊考虑。

## 1. 模型适配核心问题

### 1.1 模型适配是在哪里进行的？

**答案：模型适配主要在vLLM项目中，vLLM-Ascend只处理硬件特定的优化。**

| 项目 | 模型数量 | 职责 |
|-----|---------|------|
| **vLLM** | 290+个 | 模型架构实现、权重加载、推理逻辑 |
| **vLLM-Ascend** | 3个 | 特定硬件优化（DeepSeek V4 DSA架构等） |

### 1.2 是否需要适配？

```
新模型是否需要适配？
    │
    ├─ 是否已有类似模型实现？
    │   │
    │   ├─ 是 → 可能只需配置文件
    │   │        (如Llama变体)
    │   │
    │   └─ 否 → 需要完整实现
    │           ↓
    │
    ├─ 模型架构是否特殊？
    │   │
    │   ├─ 特殊架构 → vLLM + vLLM-Ascend适配
    │   │               (如 DS V4 的 DSA)
    │   │
    │   └─ 常见架构 → 仅vLLM适配
    │                  (如Transformer变体)
    │
    ▼
评估适配工作量
```

## 2. 模型适配架构

### 2.1 vLLM模型层架构

```
vllm/model_executor/
├── layers/              # 基础算子层
│   ├── activation.py
│   ├── attention.py
│   ├── linear.py
│   ├── layernorm.py
│   ├── rotary_embedding.py
│   ├── fused_moe/
│   ├── quantization/
│   └── ...
│
├── models/              # 模型实现（290+个）
│   ├── llama.py         # Llama系列
│   ├── deepseek_v2.py   # DeepSeek系列
│   ├── qwen.py          # Qwen系列
│   └── ...              # 其他模型
│
└── weight_loader.py     # 权重加载
```

### 2.2 vLLM-Ascend模型层架构

```
vllm_ascend/
├── ops/                 # NPU算子
│   ├── activation.py
│   ├── linear.py
│   ├── attention/       # Attention后端
│   ├── fused_moe/
│   └── triton/
│
├── models/              # 仅3个特殊模型
│   ├── deepseek_v4.py    # DS V4（DSA架构）
│   ├── deepseek_v4_mtp.py # DS V4 MTP
│   └── layer/           # 特殊层实现
│
└── patch/               # Monkey Patch
    ├── platform/        # 全局Patch
    └── worker/          # Worker Patch
```

### 2.3 模型注册机制

**vLLM模型注册流程：**

```python
# 1. 在models目录创建模型文件
# vllm/model_executor/models/my_model.py
# 通过 registry.py 中的静态映射注册（推荐）

# 2. 或在 registry.py 中添加映射：
# vllm/model_executor/models/registry.py
# 在 _VLLM_MODEL_MAP 字典中注册:
#     "MyModelForCausalLM": ("my_model", "MyModelForCausalLM"),

# 3. vLLM自动识别架构
# 通过HuggingFace config.json中的architectures字段
```

**注册表示例（registry.py）：**
```python
# vllm/model_executor/models/registry.py

_VLLM_MODEL_MAP = {
    # "架构名": ("模块文件名", "类名")
    "LlamaForCausalLM": ("llama", "LlamaForCausalLM"),
    "Qwen2ForCausalLM": ("qwen2", "Qwen2ForCausalLM"),
    "DeepseekV2ForCausalLM": ("deepseek_v2", "DeepseekV2ForCausalLM"),
    # 新增模型:
    "MyModelForCausalLM": ("my_model", "MyModelForCausalLM"),
}
```

**模型注册表位置：**
- `vllm/model_executor/models/registry.py` — 静态映射表 `_VLLM_MODEL_MAP`

## 3. 模型适配详细步骤

### 步骤1：分析模型架构

**需要分析的内容：**

```python
# 1. 查看HuggingFace模型配置
config.json:
{
  "architectures": ["MyModelForCausalLM"],
  "model_type": "my_model",
  "hidden_size": 4096,
  "num_attention_heads": 32,
  "num_hidden_layers": 28,
  ...
}

# 2. 分析模型文件
# transformers/models/my_model/modeling_my_model.py
- Attention类型（标准、MLA、GQA等）
- 是否使用MoE
- 位置编码类型（RoPE、ALiBi等）
- 激活函数（SwiGLU、GeGLU等）
- 是否有特殊层（如DeepSeek的Compressor）
```

**架构类型判断：**

| 架构特征 | vLLM实现复杂度 | vLLM-Ascend适配需求 |
|---------|---------------|-------------------|
| 标准Transformer | 低 | 可能不需要 |
| GQA (Grouped Query Attention) | 中 | Attention后端适配 |
| MLA (Multi-Head Latent Attention) | 高 | 需要NPU MLA后端 |
| MoE (Mixture of Experts) | 高 | 需要NPU MoE算子 |
| 新型位置编码 | 中 | 可能需要Custom Op |
| 新型激活函数 | 低 | 可能不需要 |

### 步骤2：创建模型文件

**基础模板：**

```python
# vllm/model_executor/models/my_model.py

"""MyModel模型实现"""
from collections.abc import Iterable
from typing import Optional, Tuple

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention import (
    Attention,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
    QKVParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    is_pp_missing_parameter,
    make_layers,
)
from vllm.sequence import IntermediateTensors
from vllm.v1.attention.backend import AttentionType

from .interfaces import SupportsPP


class MyModelAttention(nn.Module):
    """MyModel的Attention层"""

    def __init__(
        self,
        vllm_config: VllmConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int,
        rope_theta: float,
        cache_config: CacheConfig,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.num_kv_heads = num_kv_heads

        # QKV投影（使用vLLM的QKVParallelLinear自动处理TP切分）
        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.num_heads,
            self.num_kv_heads,
            bias=False,
        )

        # 输出投影
        self.o_proj = RowParallelLinear(
            num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )

        # RoPE位置编码
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
        )

        # Attention层（使用vLLM抽象层，NPU自动路由到对应的backend）
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.num_kv_heads,
            cache_config=cache_config,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: "AttentionMetadata",
    ) -> torch.Tensor:
        # 1. QKV投影
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([...], dim=-1)

        # 2. 应用RoPE
        q, k = self.rotary_emb(positions, q, k)

        # 3. Attention计算
        attn_output = self.attn(q, k, v, kv_cache, attn_metadata)

        # 4. 输出投影
        output, _ = self.o_proj(attn_output)
        return output


class MyModelMLP(nn.Module):
    """MyModel的MLP层"""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
    ):
        super().__init__()
        self.gate_up_proj = ColumnParallelLinear(
            hidden_size,
            intermediate_size * 2,
            bias=False,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )
        self.act_fn = SiluAndMul()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(hidden_states)
        gate_up = self.act_fn(gate_up)
        output, _ = self.down_proj(gate_up)
        return output


class MyModelDecoderLayer(nn.Module):
    """单个Transformer层"""

    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.hidden_size = config.hidden_size

        # Attention
        self.self_attn = MyModelAttention(
            vllm_config=vllm_config,
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rope_theta=config.rope_theta,
            cache_config=vllm_config.cache_config,
            prefix=f"{prefix}.self_attn",
        )

        # MLP
        self.mlp = MyModelMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
        )

        # LayerNorm
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        residual: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Pre-Attention Norm
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Attention
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            positions=positions,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
        )

        # Post-Attention residual
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        # MLP
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, residual


class MyModelModel(nn.Module):
    """Transformer主体"""

    def __init__(
        self,
        config: ModelConfig,
        cache_config: CacheConfig,
        lora_config: Optional[LoRAConfig] = None,
    ):
        super().__init__()
        self.config = config

        # Embedding
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            pad_token_id=config.pad_token_id,
        )

        # Transformer Layers
        # make_layers 自动处理PP（流水线并行）的层切分
        self.layers = make_layers(
            config.num_hidden_layers,
            lambda i: MyModelDecoderLayer(vllm_config, prefix=f"model.layers.{i}"),
            prefix="model.layers",
        )

        # Final Norm
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: list[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> torch.Tensor:
        # Embedding
        hidden_states = self.embed_tokens(input_ids)

        # Transformer Layers
        for i, layer in enumerate(self.layers):
            hidden_states, residual = layer(
                hidden_states=hidden_states,
                positions=positions,
                kv_cache=kv_caches[i],
                attn_metadata=attn_metadata,
                residual=None if i == 0 else residual,
            )

        # Final Norm
        hidden_states = self.norm(hidden_states)
        return hidden_states


class MyModelForCausalLM(
    # 继承Mixin以支持vLLM的高级特性
    SupportsPP,  # 支持流水线并行
    nn.Module,
):
    """完整的CausalLM模型
    vLLM中的模型类遵循以下约定：
    - __init__ 接收 vllm_config 和 prefix 参数
    - compute_logits 接收 hidden_states 返回 logits
    - load_weights 接收 Iterable[tuple[str, Tensor]] 返回 set[str]
    """

    # 定义合并的参数映射（HF格式 -> vLLM格式）
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.model = MyModelModel(vllm_config, prefix=f"{prefix}.model")

        # LM Head（使用ColumnParallelLinear自动处理TP切分）
        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: list[torch.Tensor],
        attn_metadata: "AttentionMetadata",
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            kv_caches=kv_caches,
            attn_metadata=attn_metadata,
            intermediate_tensors=intermediate_tensors,
        )
        logits, _ = self.lm_head(hidden_states)
        return logits

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """计算logits（由vLLM的调度器调用）"""
        # 使用LogitsProcessor处理TP切分和采样
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """权重加载
        vLLM的权重以 Iterable[(name, Tensor)] 形式传入
        AutoWeightsLoader 自动处理：
          - stacked参数合并（如 q_proj + k_proj + v_proj -> qkv_proj）
          - TP切分
          - PP层跳过
        """
        # 使用AutoWeightsLoader简化加载（推荐）
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            # 将q_proj/k_proj/v_proj合并到qkv_proj
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            # 将gate_proj/up_proj合并到gate_up_proj
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        loader = AutoWeightsLoader(
            self,
            stacked_params_mapping=stacked_params_mapping,
        )
        return loader.load_weights(weights)
```

**⚠️ 注意：** 上述 `MyModelForCausalLM` 类定义已经是完整的模板代码。实际使用时，`registry.py` 中的映射就足够了——不需要额外的 `@register_model` 装饰器。

### 步骤3：注册模型

**方式1：在registry.py中添加映射（推荐）**

```python
# vllm/model_executor/models/registry.py

# 在 _VLLM_MODEL_MAP 字典中新增:
_VLLM_MODEL_MAP = {
    ...
    "MyModelForCausalLM": ("my_model", "MyModelForCausalLM"),
}
```

**方式2：通过__init__.py导入（旧方式）**

```python
# vllm/model_executor/models/__init__.py

from .my_model import MyModelForCausalLM  # noqa: F401
```

### 步骤4：权重加载适配

**权重映射策略：**

**推荐方式：使用 AutoWeightsLoader（自动处理stacked参数和TP切分）**

```python
def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
    """使用AutoWeightsLoader自动加载

    AutoWeightsLoader 自动处理：
    - 遍历模块所有参数，匹配权重名称
    - stacked参数合并（q_proj + k_proj + v_proj → qkv_proj）
    - TP切分（ColumnParallelLinear的额外维度）
    - PP跳过不属于本层的参数
    """
    stacked_params_mapping = [
        # (合并后的参数名, HF权重名, shard_id)
        ("qkv_proj", "q_proj", "q"),
        ("qkv_proj", "k_proj", "k"),
        ("qkv_proj", "v_proj", "v"),
        ("gate_up_proj", "gate_proj", 0),
        ("gate_up_proj", "up_proj", 1),
    ]
    loader = AutoWeightsLoader(
        self,
        stacked_params_mapping=stacked_params_mapping,
    )
    return loader.load_weights(weights)
```

**手动方式（理解原理，仅复杂场景需要）：**

```python
def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
    """手动权重加载（演示底层机制）"""
    params_dict = dict(self.named_parameters())
    loaded_params = set()

    for name, loaded_weight in weights:
        # 跳过PP缺失的层
        if is_pp_missing_parameter(name, self):
            continue

        # 查找目标参数名
        param = params_dict.get(name)
        if param is None:
            continue

        # 使用参数的weight_loader（TP切分也在其中完成）
        weight_loader = getattr(param, "weight_loader", default_weight_loader)
        weight_loader(param, loaded_weight)
        loaded_params.add(name)

    return loaded_params
```

**常见权重格式适配：**

| HuggingFace格式 | vLLM格式 | 说明 |
|----------------|---------|------|
| `model.layers.{i}.self_attn.q_proj` | `model.layers.{i}.self_attn.qkv_proj.q` | 合并QKV |
| `model.layers.{i}.mlp.gate_proj` | `model.layers.{i}.mlp.gate_up_proj[0]` | 合并Gate/Up |
| `model.embed_tokens.weight` | `model.embed_tokens.weight` | 直接映射 |

### 步骤5：测试与验证

**基础测试：**

```python
# test_my_model.py

from vllm import LLM, SamplingParams

# 1. 加载模型
llm = LLM(
    model="path/to/my_model",
    trust_remote_code=True,
)

# 2. 测试推理
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=100,
)

outputs = llm.generate(
    ["Hello, how are you?"],
    sampling_params,
)

# 3. 验证输出
for output in outputs:
    print(output.outputs[0].text)
```

**性能测试：**

```python
# 测试吞吐量
from vllm import LLM

llm = LLM(model="my_model", tensor_parallel_size=2)
# 运行benchmark
# python benchmarks/benchmark_throughput.py --model my_model
```

## 4. Ascend NPU适配

### 4.1 是否需要vLLM-Ascend适配？

**判断标准：**

| 模型特征 | 是否需要Ascend适配 | 原因 |
|---------|------------------|------|
| 标准Transformer | 可能不需要 | vLLM抽象层自动路由 |
| 使用MoE | 需要 | 需要NPU MoE算子 |
| 使用MLA | 需要 | 需要NPU MLA后端 |
| 使用新型Attention | 需要 | 需要Custom Op |
| 使用量化 | 可能需要 | 需要NPU量化算子 |

### 4.2 Ascend适配方式

**方式1：利用现有Patch（最常见）**

大多数模型通过vLLM-Ascend现有的48个Patch自动适配：

```
patch/
├── platform/
│   ├── patch_triton.py         # Triton算子替换
│   ├── patch_distributed.py    # 分布式通信
│   └── patch_kv_cache_*.py     # KV Cache适配
│
└── worker/
│   ├── patch_triton.py         # Triton优化
│   ├── patch_qwen3vl.py        # 多模态适配示例
│   ├── patch_minimax_m2.py     # 特殊模型示例
│   └── ...
```

**方式2：添加新的Worker Patch**

如果模型有特殊算子需要NPU优化：

```python
# vllm_ascend/patch/worker/patch_my_model.py

"""MyModel的NPU适配Patch"""

import vllm.model_executor.models.my_model as my_model_module
from vllm_ascend.ops.my_special_op import npu_special_op_forward


def apply_patch():
    """应用Monkey Patch"""
    # 替换特殊算子
    my_model_module.MySpecialOp.forward = npu_special_op_forward

    # 或添加新方法
    my_model_module.MyModelAttention._npu_forward = npu_attention_forward
```

**方式3：创建专用模型文件（极少数情况）**

仅用于全新架构（如 DS V4 的 DSA）：

```python
# vllm_ascend/models/deepseek_v4.py

"""DeepSeek V4专用实现（DSA架构）"""

from vllm_ascend.ops.dsa import DSAOp
from vllm_ascend.attention.dsa_v1 import AscendDSABackend


class DeepSeekV4ForCausalLM(nn.Module):
    """DeepSeek V4在NPU上的专用实现"""

    def __init__(self, config, cache_config, ...):
        super().__init__()
        # 使用NPU专用算子
        self.dsa_attention = DSAOp(...)
        ...
```

### 4.3 Attention后端适配

**NPU支持的Attention后端：**

```python
# vllm_ascend/platform.py

@classmethod
def get_attn_backend_cls(cls, selected_backend, attn_selector_config, ...):
    backend_map = {
        # MLA模型
        (True, False, False): "vllm_ascend.attention.mla_v1.AscendMLABackend",

        # 标准Attention
        (False, False, False): "vllm_ascend.attention.attention_v1.AscendAttentionBackend",

        # Sparse Flash Attention（DeepSeek V4）
        (True, True, False): "vllm_ascend.attention.sfa_v1.AscendSFABackend",

        # DeepSeek Attention
        (True, False, True): "vllm_ascend.attention.dsa_v1.AscendDSABackend",
    }
    return backend_map[(use_mla, use_sparse, use_compress)]
```

**适配新Attention类型：**

1. 在 `vllm_ascend/ops/` 创建新算子
2. 在 `vllm_ascend/attention/` 创建新后端
3. 在 `platform.py` 的 `get_attn_backend_cls` 中添加映射

### 4.4 MoE适配

**vLLM-Ascend的MoE实现：**

```
vllm_ascend/ops/fused_moe/
├── fused_moe.py           # 核心MoE算子
├── moe_mlp.py             # 专家MLP
├── gate_linear.py         # 门控选择
├── token_dispatcher.py    # Token分发
├── moe_comm_method.py     # 通信方法
│   ├── MC2               # 集合通信优化
│   ├── All2All           # 专家并行
│   └── AllReduce         # 张量并行
└── prepare_finalize.py    # 内存管理
```

**MoE模型适配步骤：**

```python
# 1. 在vLLM模型中使用MoE层
from vllm.model_executor.layers.fused_moe import fused_moe

class MyMoEModelDecoderLayer(nn.Module):
    def __init__(self, config, ...):
        # MoE层会自动使用vLLM-Ascend的NPU优化
        self.mlp = fused_moe.FusedMoE(...)

# 2. Worker Patch自动替换（已存在）
# patch/worker/patch_fused_moe.py会自动应用
```

## 5. 特殊模型适配案例

### 5.1 案例：DS V4（DSA架构）

**为什么需要vLLM-Ascend专用实现？**

DeepSeek V4引入了DSA（DeepSeek Attention），这是一种全新的Attention架构：
- 动态稀疏Attention
- Compressor层压缩KV Cache
- 独特的RoPE变体

**适配文件：**

```
vllm-ascend/vllm_ascend/models/
├── deepseek_v4.py          # DeepSeek V4专用实现
├── deepseek_v4_mtp.py      # Multi-Token Predictor
└── layer/
    ├── compressor.py      # Compressor层
    ├── dsa_attention.py   # DSA Attention
    └── ...
```

**关键适配点：**

1. DSA算子实现：`vllm_ascend/ops/dsa.py`
2. Compressor优化：`vllm_ascend/ops/layer_shard_linear.py`
3. 特殊RoPE：`vllm_ascend/ops/rope_dsv4.py`
4. Attention后端：`vllm_ascend/attention/dsa_v1/`

### 5.2 案例：Qwen3-VL（多模态）

**适配方式：使用Worker Patch**

```python
# vllm_ascend/patch/worker/patch_qwen3vl.py

"""Qwen3-VL多模态模型NPU适配"""

import vllm.model_executor.models.qwen3_vl as qwen3vl_module
from vllm_ascend.ops.mm_encoder_attention import npu_mm_encoder_attention


def apply_patch():
    """替换多模态编码器Attention"""
    qwen3vl_module.Qwen3VLForConditionalGeneration._get_deepstack_input_embeds = (
        npu_get_deepstack_input_embeds
    )

    # 支持Flash Comm v1
    qwen3vl_module.Qwen3VLVisionEncoder.attention = npu_mm_encoder_attention
```

### 5.3 案例：MiniMax-M2

**适配方式：多文件Patch**

```
patch/platform/
├── patch_minimax_m2_config.py        # 配置验证
├── patch_minimax_m2_tool_call_parser.py  # 工具调用解析
├── patch_minimax_usage_accounting.py  # 使用统计

patch/worker/
├── patch_minimax_m2.py               # 模型前向
├── patch_minimax_m2_linear_attn.py   # 线性Attention
```

**关键适配：**
- 线性Attention的RMSNorm优化
- fp8权重加载适配
- Eagle3推测解码支持

## 6. 模型适配流程图

```
新模型适配流程
    │
    ├─ 1. 分析模型架构
    │      │
    │      ├─ 查看HuggingFace config
    │      ├─ 分析transformers实现
    │      └─ 判断是否需要适配
    │      │
    │      ▼
    │
    ├─ 2. 在vLLM创建模型文件
    │      │
    │      ├─ 创建layers（Attention, MLP等）
    │      ├─ 实现forward方法
    │      ├─ 实现load_weights方法
    │      └─ 注册模型
    │      │
    │      ▼
    │
    ├─ 3. 测试基础功能
    │      │
    │      ├─ 加载模型
    │      ├─ 测试推理
    │      └─ 验证输出
    │      │
    │      ▼
    │
    ├─ 4. Ascend适配（如需要）
    │      │
    │      ├─ 判断是否需要NPU优化
    │      │     │
    │      │     ├─ 需要特殊算子 → 创建ops文件
    │      │     ├─ 需要特殊后端 → 创建attention backend
    │      │     ├─ 需要Patch → 创建patch文件
    │      │     └─ 不需要 → 利用现有Patch
    │      │
    │      ▼
    │
    ├─ 5. 性能测试与优化
    │      │
    │      ├─ 吞吐量测试
    │      ├─ 内存分析
    │      └─ 优化瓶颈算子
    │      │
    │      ▼
    │
    └─ 6. 提交与维护
           │
           ├─ 提交到vLLM（主实现）
           ├─ 提交到vLLM-Ascend（NPU优化）
           └─ 添加文档和测试
```

## 7. 常见问题与解决方案

### Q1：模型加载失败，提示找不到模型类

**解决方案：**
```python
# 检查注册是否正确
from vllm.model_executor.models import ModelRegistry
registry = ModelRegistry.get_supported_archs()
assert "MyModelForCausalLM" in registry

# 或使用trust_remote_code
llm = LLM(model="my_model", trust_remote_code=True)
```

### Q2：权重加载失败，形状不匹配

**解决方案：**
```python
# 检查权重映射
def load_weights(self, weights):
    # 添加调试信息
    for name, weight in weights.items():
        print(f"{name}: {weight.shape}")

    # 调整stacked_params_mapping
    stacked_params_mapping = [
        ("qkv_proj", "q_proj", "q"),
        ...
    ]
```

### Q3：NPU上性能差，怎么办？

**解决方案：**

1. 检查是否使用了Patch：
```python
# 查看Patch日志
from vllm.logger import logger
# 启动时会打印"Applied patch: patch_xxx.py"
```

2. 分析性能瓶颈：
```bash
# 使用NPU profiling
ASCEND_LAUNCH_BLOCKING=1 python my_test.py
```

3. 添加优化Patch：
```python
# 创建新的Worker Patch
# vllm_ascend/patch/worker/patch_my_model.py
```

### Q4：MoE模型在NPU上报错

**解决方案：**
```python
# 确保使用正确的通信方法
from vllm_ascend.ops.fused_moe import get_moe_comm_method

# 检查配置
# 需要设置：enable_expert_parallel=True (for TP+EP)
# 或使用MC2通信方法
```

## 8. 最佳实践总结

### 8.1 适配原则

1. **最小化原则**：优先利用vLLM抽象层，不重复造轮子
2. **分层原则**：算子在ops/，模型在models/，Patch在patch/
3. **性能优先**：NPU优化仅用于性能关键路径

### 8.2 适配检查清单

| 检查项 | 说明 |
|-------|------|
| 模型架构分析 | 是否有特殊层或算子 |
| vLLM基础实现 | 是否正确注册和加载 |
| 权重映射 | 是否正确处理stacked参数 |
| 基础测试 | 是否能正确推理 |
| NPU适配 | 是否需要特殊算子或Patch |
| 性能测试 | 吞吐量和延迟是否满足要求 |
| 文档和测试 | 是否添加相应文档和测试用例 |

### 8.3 参考资源

- **vLLM模型示例**：`vllm/model_executor/models/llama.py`
- **vLLM-Ascend特殊模型**：`vllm_ascend/models/deepseek_v4.py`
- **Patch示例**：`vllm_ascend/patch/worker/patch_qwen3vl.py`
- **算子示例**：`vllm_ascend/ops/activation.py`

---

*文档版本：v1.0*
*创建日期：2026-06-20*
*基于vLLM和vLLM-Ascend源码分析*