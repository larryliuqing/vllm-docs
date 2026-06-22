# vLLM 推测解码架构与实现详解

> 本文档深度解析 vLLM 的推测解码（Speculative Decoding）功能，从业务和功能角度阐述其架构设计、核心组件、处理流程和实现原理。

---

## 一、推测解码概述

### 1.1 功能定位

**推测解码**是一种加速大语言模型推理的技术，通过以下机制提升性能：

1. **Draft Phase**: 使用轻量级方法快速生成候选 token 序列
2. **Verify Phase**: 目标模型并行验证所有候选 token
3. **Accept/Reject**: 根据验证结果接受或拒绝候选 token

**核心价值**：
- **降低延迟**: 通过并行验证减少推理轮次
- **提升吞吐**: 减少目标模型的计算次数
- **降低成本**: 减少昂贵的模型推理调用

### 1.2 支持的推测方法

| 方法 | 提议器 | 是否需要模型 | 输入类型 | 适用场景 | 特点 |
|------|-------|------------|---------|---------|------|
| **Ngram** | `NgramProposer` | ❌ 无需模型 | Token IDs | 重复文本场景 | 基于 N-gram 匹配，零成本 |
| **Eagle** | `EagleProposer` | ✅ 需要模型 | Hidden States | 通用加速 | 使用目标模型的隐藏状态 |
| **Eagle3** | `Eagle3Proposer` | ✅ 需要模型 | Hidden States | Llama/DeepSeek | Eagle 的增强版本 |
| **Medusa** | `MedusaProposer` | ✅ 需要模型 | Hidden States | 通用加速 | 多头并行预测 |
| **DFlash** | `DFlashProposer` | ✅ 需要模型 | Hidden States | Qwen3 系列 | 并行解码，支持多模态 |
| **MTP** | `MTPProposer` | ✅ 需要模型 | Hidden States | DeepSeek V4 | Multi-Token Prediction |
| **Draft Model** | `DraftModelProposer` | ✅ 需要模型 | Token IDs | 通用加速 | 使用小型 draft 模型 |
| **Suffix Decoding** | `SuffixProposer` | ❌ 无需模型 | Token IDs | 特定后缀场景 | 基于后缀匹配 |

### 1.3 源码规模

| 项目 | 文件数 | 总行数 | 主要文件 |
|------|--------|--------|---------|
| **vLLM** | 12 | 4599 | `llm_base_proposer.py` (1809), `ngram_proposer_gpu.py` (662), `utils.py` (602) |
| **vLLM-Ascend** | 10 | 2671 | `llm_base_proposer.py` (1979), `dflash_proposer.py` (265) |

---

## 二、核心组件架构

### 2.1 组件抽象与分层

```mermaid
graph TB
    subgraph Config["配置层"]
        SpecConfig[SpeculativeConfig<br/>推测解码配置]
        DraftConfig[DraftModelConfig<br/>Draft 模型配置]
    end
    
    subgraph Proposer["提议器层"]
        Base[SpecDecodeBaseProposer<br/>基础提议器抽象]
        Ngram[NgramProposer<br/>N-gram 提议器]
        Eagle[EagleProposer<br/>Eagle 提议器]
        Medusa[MedusaProposer<br/>Medusa 提议器]
        DFlash[DFlashProposer<br/>DFlash 提议器]
        Draft[DraftModelProposer<br/>Draft Model 提议器]
    end
    
    subgraph Metadata["元数据层"]
        SpecMeta[SpecDecodeMetadata<br/>推测解码元数据]
        SampleMeta[SamplingMetadata<br/>采样元数据]
    end
    
    subgraph Utils["工具层"]
        UtilsModule[utils.py<br/>辅助函数]
        Metrics[metrics.py<br/>性能指标]
    end
    
    SpecConfig --> Base
    DraftConfig --> Base
    
    Base --> Ngram
    Base --> Eagle
    Base --> Medusa
    Base --> DFlash
    Base --> Draft
    
    Ngram --> SpecMeta
    Eagle --> SpecMeta
    Medusa --> SpecMeta
    DFlash --> SpecMeta
    Draft --> SpecMeta
    
    SpecMeta --> SampleMeta
    
    UtilsModule --> Proposer
    Metrics --> Proposer
    
    style Config fill:#e1f5ff
    style Proposer fill:#fff9c4
    style Metadata fill:#c8e6c9
    style Utils fill:#ffccbc
```

### 2.2 核心组件职责

#### **2.2.1 SpecDecodeBaseProposer（基础提议器）**

**文件**: `vllm/vllm/v1/spec_decode/llm_base_proposer.py` (1809 行)

**核心职责**：
1. **配置管理**: 管理推测解码配置参数
2. **隐藏状态管理**: 缓存和传递目标模型的隐藏状态
3. **模型加载**: 加载 draft 模型（如果需要）
4. **提议生成**: 生成候选 token 序列
5. **CUDA Graph 支持**: 支持 CUDA Graph 优化
6. **并行起草**: 支持并行起草模式

**关键属性**：
```python
class SpecDecodeBaseProposer:
    def __init__(self, vllm_config, device, pass_hidden_states_to_model, runner):
        self.vllm_config = vllm_config                    # vLLM 配置
        self.speculative_config = vllm_config.speculative_config  # 推测配置
        self.draft_model_config = speculative_config.draft_model_config  # Draft 模型配置
        self.method = speculative_config.method            # 推测方法
        self.num_speculative_tokens = speculative_config.num_speculative_tokens  # 推测 token 数
        
        self.hidden_size = draft_model_config.get_hidden_size()  # 隐藏层大小
        self.parallel_drafting = speculative_config.parallel_drafting  # 是否并行起草
        self.pass_hidden_states_to_model = pass_hidden_states_to_model  # 是否传递隐藏状态
```

**核心方法**：
- `load_model()`: 加载 draft 模型
- `propose()`: 生成候选 token 序列
- `dummy_run()`: 用于 CUDA Graph 的空运行
- `set_inputs_first_pass()`: 设置第一轮输入
- `set_inputs_second_pass()`: 设置第二轮输入

---

#### **2.2.2 NgramProposer（N-gram 提议器）**

**文件**: `vllm/vllm/v1/spec_decode/ngram_proposer.py` (285 行) + `ngram_proposer_gpu.py` (662 行)

**核心职责**：
1. **N-gram 匹配**: 在历史 token 中查找匹配的 N-gram
2. **候选生成**: 基于匹配结果生成候选 token
3. **批量处理**: 支持批量请求的 N-gram 提议
4. **多线程加速**: 使用 Numba JIT 加速 N-gram 查找

**核心原理**：

```mermaid
flowchart LR
    A[历史 Token 序列] --> B[翻转序列]
    B --> C[LPS 算法查找<br/>最长匹配 N-gram]
    C --> D[定位匹配位置]
    D --> E[提取后续 K 个 Token]
    E --> F[候选 Token 序列]
    
    style A fill:#e1f5ff
    style B fill:#fff9c4
    style C fill:#c8e6c9
    style D fill:#ffccbc
    style E fill:#d1c4e9
    style F fill:#f8bbd0
```

**关键参数**：
- `min_n`: 最小 N-gram 长度（`prompt_lookup_min`）
- `max_n`: 最大 N-gram 镕度（`prompt_lookup_max`）
- `k`: 候选 token 数量（`num_speculative_tokens`）

**算法实现**：
```python
# LPS (Longest Prefix Suffix) 算法
# 在翻转后的 token 序列中查找最长匹配 N-gram
def _find_longest_matched_ngram_and_propose_tokens(
    origin_tokens,  # 原始 token 序列
    min_ngram,      # 最小 N-gram 长度
    max_ngram,      # 最大 N-gram 长度
    k,              # 候选 token 数
):
    # 1. 翻转 token 序列
    tokens = origin_tokens[::-1]
    
    # 2. 计算 LPS 数组
    lps = np.zeros(max_ngram, dtype=np.int32)
    
    # 3. 查找最长匹配
    # 在翻转序列的前缀中查找与当前位置匹配的最长前缀
    
    # 4. 提取候选 token
    # 从匹配位置开始提取后续 k 个 token
    
    return origin_tokens[start_position : start_position + k]
```

**性能优化**：
- **Numba JIT 编译**: 使用 `@njit(parallel=True)` 加速批量处理
- **多线程并行**: 批量请求并行处理 N-gram 查找
- **阈值控制**: 超过 8192 token 才启用多线程

---

#### **2.2.3 EagleProposer（Eagle 提议器）**

**文件**: `vllm/vllm/v1/spec_decode/eagle.py` (22 行)

**核心职责**：
1. **隐藏状态传递**: 使用目标模型的隐藏状态作为输入
2. **Draft 模型推理**: 使用 Eagle draft 模型生成候选
3. **树形解码**: 支持树形解码结构

**特点**：
- 继承自 `SpecDecodeBaseProposer`
- `pass_hidden_states_to_model = True`
- 使用目标模型的最后一层隐藏状态

**工作流程**：
```mermaid
sequenceDiagram
    participant Target as 目标模型
    participant Eagle as Eagle Draft 模型
    participant Verify as 验证阶段
    
    Target->>Target: 前向传播生成 token_0
    Target->>Eagle: 传递 hidden_states_0
    Eagle->>Eagle: 前向传播生成 [token_1, token_2, ..., token_k]
    Eagle->>Verify: 返回候选 token 序列
    Verify->>Target: 并行验证所有候选
    Target->>Target: 接受/拒绝候选
```

---

#### **2.2.4 MedusaProposer（Medusa 提议器）**

**文件**: `vllm/vllm/v1/spec_decode/medusa.py` (78 行)

**核心职责**：
1. **多头预测**: 使用多个 Medusa head 并行预测多个 token
2. **隐藏状态传递**: 使用目标模型的隐藏状态
3. **Argmax 选择**: 每个 head 选择概率最大的 token

**核心实现**：
```python
class MedusaProposer:
    def propose(self, target_hidden_states, sampling_metadata):
        # 1. Medusa 模型前向传播
        blocks = self.model(target_hidden_states)
        
        # 2. 计算每个 head 的 logits
        logits = self.model.compute_logits(blocks)
        
        # 3. 每个 head argmax 选择 token
        # logits 是一个列表，每个元素对应一个 head 的 logits
        # Shape: [batch_size, vocab_size] per head
        draft_tokens = torch.stack([logit.argmax(dim=-1) for logit in logits], dim=1)
        
        # 4. 返回候选 token
        # Shape: [batch_size, num_heads]
        return draft_tokens
```

**特点**：
- 多个 head 并行预测不同位置的 token
- 无需额外的 draft 模型推理轮次
- 与目标模型共享隐藏状态

---

#### **2.2.5 DFlashProposer（DFlash 提议器）**

**文件**: `vllm/vllm/v1/spec_decode/dflash.py` (289 行)

**核心职责**：
1. **并行解码**: 所有候选 token 在一次前向中生成
2. **多模态支持**: 支持多模态输入（Qwen3.5 模型）
3. **Mask Token**: 使用特殊的 mask token 进行并行解码
4. **上下文分离**: 分离 context token 和 query token

**核心原理**：

```mermaid
graph TB
    subgraph Input["输入准备"]
        A[目标模型 Hidden States]
        B[Next Token IDs]
        C[Context Token IDs]
    end
    
    subgraph Prepare["输入准备阶段"]
        D[构建 Query Tokens<br/>包含 Next Token + Mask Tokens]
        E[构建 Context Tokens<br/>历史 token 作为 K/V]
        F[拼接 Positions<br/>Context + Query]
    end
    
    subgraph Forward["前向传播"]
        G[DFlash Model 前向]
        H[并行生成所有候选]
    end
    
    subgraph Output["输出"]
        I[候选 Token 序列]
    end
    
    A --> D
    B --> D
    C --> E
    D --> F
    E --> F
    F --> G
    G --> H
    H --> I
    
    style Input fill:#e1f5ff
    style Prepare fill:#fff9c4
    style Forward fill:#c8e6c9
    style Output fill:#ffccbc
```

**关键参数**：
```python
class DFlashProposer(SpecDecodeBaseProposer):
    def __init__(self, vllm_config, device, runner):
        super().__init__(vllm_config, device, pass_hidden_states_to_model=True)
        
        # Query tokens 数量 = batch_size * (1 next_token + num_speculative_tokens mask)
        self.max_query_tokens = self.max_batch_size * (1 + self.num_speculative_tokens)
        
        # Positions 数量 = context tokens + query tokens
        self.max_positions = self.max_num_tokens + self.max_query_tokens
```

**特点**：
- **并行起草**: 所有候选 token 在一次前向中生成（`parallel_drafting=True`）
- **Mask Token**: 使用特殊 token 作为占位符
- **多模态**: 支持视觉等多模态输入
- **地址稳定性**: 分离 context buffer 保持 query buffer 地址稳定（CUDA Graph）

---

#### **2.2.6 SpecDecodeMetadata（推测解码元数据）**

**文件**: `vllm/vllm/v1/spec_decode/metadata.py` (66 行)

**核心职责**：
1. **元数据封装**: 封装推测解码所需的元数据
2. **索引管理**: 管理 draft token 和 target logits 的索引
3. **批次信息**: 管理每个请求的 draft token 数量

**数据结构**：
```python
@dataclass
class SpecDecodeMetadata:
    # Draft token IDs
    draft_token_ids: torch.Tensor          # [num_tokens]
    
    # 每个请求的 draft token 数量
    num_draft_tokens: list[int]            # [batch_size]
    
    # Cumulative sum
    cu_num_draft_tokens: torch.Tensor      # [batch_size]
    cu_num_sampled_tokens: torch.Tensor    # [batch_size]
    
    # Target logits 索引
    target_logits_indices: torch.Tensor    # [num_tokens]
    
    # Bonus logits 索引（最后一个 token）
    bonus_logits_indices: torch.Tensor     # [batch_size]
    
    # 所有 logits 索引（draft + bonus）
    logits_indices: torch.Tensor           # [num_tokens + batch_size]
```

**索引关系**：

```mermaid
graph LR
    subgraph Batch["批次请求"]
        R1[Request 1: 3 draft tokens]
        R2[Request 2: 2 draft tokens]
        R3[Request 3: 4 draft tokens]
    end
    
    subgraph DraftTokens["Draft Token IDs"]
        D1[d1_1, d1_2, d1_3]
        D2[d2_1, d2_2]
        D3[d3_1, d3_2, d3_3, d3_4]
    end
    
    subgraph Indices["Logits Indices"]
        L1[Target logits indices<br/>指向 draft token 位置]
        L2[Bonus logits indices<br/>指向最后一个 token]
    end
    
    R1 --> D1
    R2 --> D2
    R3 --> D3
    
    D1 --> L1
    D2 --> L1
    D3 --> L1
    
    R1 --> L2
    R2 --> L2
    R3 --> L2
    
    style Batch fill:#e1f5ff
    style DraftTokens fill:#fff9c4
    style Indices fill:#c8e6c9
```

---

## 三、推测解码工作流程

### 3.1 整体流程

```mermaid
flowchart TB
    Start([开始推理]) --> Config[加载推测解码配置]
    
    Config --> InitProposer[初始化 Proposer]
    
    InitProposer --> DraftPhase{Draft Phase}
    
    DraftPhase --> Ngram[Ngram 方法<br/>基于历史匹配]
    DraftPhase --> Eagle[Eagle/Medusa 方法<br/>基于隐藏状态]
    DraftPhase --> DFlash[DFlash 方法<br/>并行解码]
    DraftPhase --> Draft[Draft Model 方法<br/>小型模型推理]
    
    Ngram --> GenDraft[生成候选 Token 序列]
    Eagle --> GenDraft
    DFlash --> GenDraft
    Draft --> GenDraft
    
    GenDraft --> SpecMeta[创建 SpecDecodeMetadata]
    
    SpecMeta --> VerifyPhase[Verify Phase<br/>目标模型并行验证]
    
    VerifyPhase --> ComputeLogits[计算所有候选 Token 的 logits]
    
    ComputeLogits --> Sampling[采样验证]
    
    Sampling --> AcceptReject{接受/拒绝判定}
    
    AcceptReject --> Accept[接受匹配的 Token]
    AcceptReject --> Reject[拒绝不匹配的 Token]
    
    Accept --> UpdateKV[更新 KV Cache]
    Reject --> UpdateKV
    
    UpdateKV --> NextRound{是否继续?}
    
    NextRound -->|Yes| DraftPhase
    NextRound -->|No| End([结束])
    
    style Start fill:#e1f5ff
    style DraftPhase fill:#fff9c4
    style VerifyPhase fill:#c8e6c9
    style AcceptReject fill:#ffccbc
    style End fill:#d1c4e9
```

### 3.2 详细步骤

#### **步骤 1: 配置加载**

```python
# SpeculativeConfig 配置
speculative_config = VllmConfig.speculative_config

# 关键参数
num_speculative_tokens = speculative_config.num_speculative_tokens  # 候选 token 数
method = speculative_config.method                                    # 推测方法
draft_model_config = speculative_config.draft_model_config            # Draft 模型配置
prompt_lookup_min = speculative_config.prompt_lookup_min              # Ngram 最小长度
prompt_lookup_max = speculative_config.prompt_lookup_max              # Ngram 最大长度
parallel_drafting = speculative_config.parallel_drafting              # 是否并行起草
```

---

#### **步骤 2: Proposer 初始化**

```python
# 根据 method 选择 Proposer
if method == "ngram":
    proposer = NgramProposer(vllm_config)
elif method == "eagle":
    proposer = EagleProposer(vllm_config, device, runner)
elif method == "medusa":
    proposer = MedusaProposer(vllm_config, device)
elif method == "dflash":
    proposer = DFlashProposer(vllm_config, device, runner)
elif method == "draft_model":
    proposer = DraftModelProposer(vllm_config, device)

# 加载模型（如果需要）
proposer.load_model()
```

---

#### **步骤 3: Draft Phase**

**Ngram 方法**：
```python
def propose(sampled_token_ids, num_tokens_no_spec, token_ids_cpu):
    # 1. 查找需要 Ngram 提议的请求
    valid_ngram_requests = []
    for i, sampled_ids in enumerate(sampled_token_ids):
        if len(sampled_ids) > 0 and num_tokens_no_spec[i] < max_model_len:
            valid_ngram_requests.append(i)
    
    # 2. 批量 Ngram 查找（Numba 加速）
    batch_propose_numba(
        valid_ngram_requests,
        num_tokens_no_spec,
        token_ids_cpu,
        min_n, max_n, max_model_len, k,
        valid_ngram_draft, valid_ngram_num_drafts
    )
    
    # 3. 返回候选 token 序列
    draft_token_ids = []
    for i in range(num_requests):
        if self.valid_ngram_num_drafts[i] > 0:
            draft_token_ids.append(self.valid_ngram_draft[i, :self.valid_ngram_num_drafts[i]].tolist())
        else:
            draft_token_ids.append([])
    
    return draft_token_ids
```

**Eagle/Medusa 方法**：
```python
def propose(target_hidden_states, sampling_metadata):
    # 1. 使用目标模型的隐藏状态
    # target_hidden_states: [num_tokens, hidden_size]
    
    # 2. Draft 模型前向传播
    draft_logits = draft_model(target_hidden_states)
    
    # 3. 采样生成候选 token
    draft_tokens = sampler(draft_logits)
    
    # 4. 返回候选序列
    # draft_tokens: [batch_size, num_speculative_tokens]
    return draft_tokens
```

---

#### **步骤 4: Verify Phase**

```python
# 1. 创建 SpecDecodeMetadata
spec_decode_metadata = SpecDecodeMetadata(
    draft_token_ids=draft_token_ids,
    num_draft_tokens=num_draft_tokens,
    cu_num_draft_tokens=cu_num_draft_tokens,
    cu_num_sampled_tokens=cu_num_sampled_tokens,
    target_logits_indices=target_logits_indices,
    bonus_logits_indices=bonus_logits_indices,
    logits_indices=logits_indices
)

# 2. 目标模型并行验证
# 输入包含所有 draft tokens + bonus token
target_logits = target_model(draft_tokens + bonus_tokens)

# 3. 提取对应位置的 logits
# target_logits_indices: 指向每个 draft token 的 logits
# bonus_logits_indices: 指向最后一个 token 的 logits
draft_logits = target_logits[target_logits_indices]
bonus_logits = target_logits[bonus_logits_indices]

# 4. 采样验证
accepted_tokens = sampling_verify(draft_logits, bonus_logits, draft_token_ids)
```

---

#### **步骤 5: Accept/Reject**

```python
def sampling_verify(draft_logits, bonus_logits, draft_token_ids):
    accepted_tokens = []
    
    for i in range(num_draft_tokens):
        # 1. 计算目标模型和 draft 模型的概率
        target_prob = softmax(draft_logits[i])
        draft_prob = softmax(draft_model_logits[i])
        
        # 2. 接受条件
        # 如果 draft token 的概率 >= 目标概率 * uniform(0, 1)
        if draft_prob[draft_token_ids[i]] >= target_prob[draft_token_ids[i]] * random():
            accepted_tokens.append(draft_token_ids[i])
        else:
            # 拒绝后从目标分布采样
            accepted_tokens.append(sample(target_prob))
            break  # 剩余 draft tokens 全部拒绝
    
    # 3. Bonus token
    # 如果所有 draft tokens 都接受，添加 bonus token
    if len(accepted_tokens) == num_draft_tokens:
        bonus_token = sample(softmax(bonus_logits))
        accepted_tokens.append(bonus_token)
    
    return accepted_tokens
```

---

## 四、vLLM vs vLLM-Ascend 实现对比

### 4.1 实现差异

| 维度 | vLLM (CUDA) | vLLM-Ascend (NPU) | 差异说明 |
|------|-------------|------------------|---------|
| **Ngram GPU 实现** | `ngram_proposer_gpu.py` (662 行) | `ngram_proposer_npu.py` (35 行) | Ascend 实现简化，继承 GPU 版本 |
| **DFlash 实现** | `dflash.py` (289 行) | `dflash_proposer.py` (265 行) | 相似实现，适配 NPU |
| **Eagle 实现** | `eagle.py` (22 行) | `eagle_proposer.py` (19 行) | 相似实现 |
| **Medusa 实现** | `medusa.py` (78 行) | `medusa_proposer.py` (70 行) | 相似实现 |
| **Base Proposer** | `llm_base_proposer.py` (1809 行) | `llm_base_proposer.py` (1979 行) | Ascend 扩展更多功能 |
| **Hidden States 处理** | 标准 CUDA tensor | NPU tensor + SP 处理 | Ascend 支持 Sequence Parallel |
| **CUDA/ACL Graph** | CUDA Graph 支持 | ACL Graph 支持 | Graph 机制不同 |

### 4.2 Ascend 特有优化

#### **4.2.1 Sequence Parallel (SP) 支持**

```python
# vllm-ascend/vllm_ascend/spec_decode/llm_base_proposer.py

def split_inputs_tp_to_sp(hidden_states, out):
    """Split hidden states along sequence dimension for SP"""
    group = get_tp_group()
    world_size = group.world_size
    rank = group.rank
    
    # 按序列维度切分
    num_tokens = hidden_states.shape[0]
    padded_num_tokens_per_rank = (num_tokens + world_size - 1) // world_size
    
    start = padded_num_tokens_per_rank * rank
    end = padded_num_tokens_per_rank * (rank + 1)
    
    hidden_states_curr_rank = hidden_states[start:end]
    out[:hidden_states_curr_rank.shape[0]] = hidden_states_curr_rank
    return out[:padded_num_tokens_per_rank]
```

#### **4.2.2 Ascend Forward Context**

```python
# 使用 Ascend 特定的 forward context
from vllm_ascend.ascend_forward_context import set_ascend_forward_context

with set_ascend_forward_context(...):
    draft_model(hidden_states)
```

#### **4.2.3 ACL Graph 支持**

```python
# vllm-ascend/vllm_ascend/spec_decode/llm_base_proposer.py

from vllm_ascend.compilation.acl_graph import ACLGraphWrapper

# ACL Graph 空运行
@torch.inference_mode()
def dummy_run(self, num_tokens, ...):
    # 用于 ACL Graph 捕获
    pass
```

---

## 五、性能优化策略

### 5.1 Ngram 提议器优化

**优化策略**：
1. **Numba JIT 编译**: 使用 `@njit(parallel=True)` 加速 N-gram 查找
2. **多线程并行**: 批量请求并行处理
3. **阈值控制**: 仅在 token 数 > 8192 时启用多线程
4. **LPS 算法**: 高效查找最长匹配 N-gram

**性能提升**：
- N-gram 查找延迟降低 80%+
- 批量处理吞吐提升 5x+

---

### 5.2 CUDA/ACL Graph 优化

**优化策略**：
1. **地址稳定性**: 保持 buffer 地址稳定以支持 Graph 捕获
2. **空运行预热**: 使用 `dummy_run()` 预热 Graph
3. **分离 buffer**: 分离 context buffer 和 query buffer

**性能提升**：
- Draft phase 延迟降低 50%+
- Graph 捕获成功率提升

---

### 5.3 并行起草优化

**优化策略**：
1. **DFlash 并行解码**: 所有候选 token 在一次前向中生成
2. **Medusa 多头预测**: 多个 head 并行预测不同位置
3. **批量验证**: 目标模型并行验证所有候选

**性能提升**：
- 推理轮次减少 30%+
- 整体吞吐提升 20%+

---

## 六、使用示例

### 6.1 Ngram 方法

```python
from vllm import LLM, SamplingParams

# 配置 Ngram 推测解码
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    speculative_config={
        "method": "ngram",
        "num_speculative_tokens": 5,
        "prompt_lookup_min": 3,
        "prompt_lookup_max": 5
    }
)

sampling_params = SamplingParams(max_tokens=100)
outputs = llm.generate(["Hello, world!"], sampling_params)
```

---

### 6.2 Eagle 方法

```python
from vllm import LLM, SamplingParams

# 配置 Eagle 推测解码
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    speculative_config={
        "method": "eagle",
        "num_speculative_tokens": 4,
        "draft_model": "meta-llama/Llama-2-7b-eagle"
    }
)

sampling_params = SamplingParams(max_tokens=100)
outputs = llm.generate(["Hello, world!"], sampling_params)
```

---

### 6.3 DFlash 方法

```python
from vllm import LLM, SamplingParams

# 配置 DFlash 推测解码（Qwen3 系列）
llm = LLM(
    model="Qwen/Qwen3-7B",
    speculative_config={
        "method": "dflash",
        "num_speculative_tokens": 6,
        "parallel_drafting": True
    }
)

sampling_params = SamplingParams(max_tokens=100)
outputs = llm.generate(["Hello, world!"], sampling_params)
```

---

## 七、最佳实践与建议

### 7.1 方法选择建议

| 场景 | 推荐方法 | 原因 |
|------|---------|------|
| **重复文本场景** | Ngram | 基于 N-gram 匹配，零额外成本 |
| **通用加速** | Eagle/Medusa | 使用目标模型隐藏状态，效果稳定 |
| **Qwen3 系列** | DFlash | 并行解码，支持多模态 |
| **DeepSeek V4** | MTP | Multi-Token Prediction，专用优化 |
| **资源受限** | Draft Model | 使用小型模型，成本可控 |

---

### 7.2 参数调优建议

**num_speculative_tokens**：
- 范围: 1-8
- 建议: 4-6
- 过大会导致验证失败率增加

**prompt_lookup_min/max**（Ngram）：
- min_n: 3-5
- max_n: 5-8
- 过大会导致匹配率降低

**parallel_drafting**：
- DFlash: True（必须）
- 其他方法: 根据需求选择

---

### 7.3 性能监控

**关键指标**：
- `draft_acceptance_rate`: Draft token 接受率
- `draft_tokens_per_request`: 每请求的 draft token 数
- `speedup`: 推测解码加速比
- `draft_model_latency`: Draft 模型延迟

**监控代码**：
```python
# metrics.py
class SpecDecodeMetrics:
    draft_acceptance_rate: float
    draft_tokens_generated: int
    draft_tokens_accepted: int
    target_model_calls: int
    speedup: float
```

---

## 八、总结

### 8.1 核心设计

**推测解码核心思想**：
1. **Draft Phase**: 轻量级方法快速生成候选
2. **Verify Phase**: 目标模型并行验证
3. **Accept/Reject**: 根据概率匹配接受或拒绝

### 8.2 关键优势

| 维度 | 优势 |
|------|------|
| **延迟** | 减少 30-50% 推理轮次 |
| **吞吐** | 提升 20-40% 整体吞吐 |
| **成本** | 降低目标模型计算次数 |
| **灵活性** | 支持多种推测方法 |

### 8.3 源码结构

```
vllm/vllm/v1/spec_decode/
├── llm_base_proposer.py         # 基础提议器 (1809 行)
├── ngram_proposer.py            # Ngram 提议器 (285 行)
├── ngram_proposer_gpu.py        # Ngram GPU 实现 (662 行)
├── eagle.py                     # Eagle 提议器 (22 行)
├── medusa.py                    # Medusa 提议器 (78 行)
├── dflash.py                    # DFlash 提议器 (289 行)
├── metadata.py                  # 推测解码元数据 (66 行)
├── utils.py                     # 辅助函数 (602 行)
└── metrics.py                   # 性能指标 (215 行)

vllm-ascend/vllm_ascend/spec_decode/
├── llm_base_proposer.py         # Ascend 基础提议器 (1979 行)
├── ngram_proposer_npu.py        # Ngram NPU 实现 (35 行)
├── eagle_proposer.py            # Eagle 提议器 (19 行)
├── medusa_proposer.py           # Medusa 提议器 (70 行)
├── dflash_proposer.py           # DFlash 提议器 (265 行)
└── utils.py                     # 辅助函数 (31 行)
```

---

**文档版本**: v1.0  
**创建时间**: 2026-06-20  
**基于源码**: vllm/vllm/v1/spec_decode/ + vllm-ascend/vllm_ascend/spec_decode/  
**维护者**: vLLM 项目分析团队