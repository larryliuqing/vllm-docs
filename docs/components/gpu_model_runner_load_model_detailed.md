# GPUModelRunner.load_model() 模型加载流程详解

## 概述

`GPUModelRunner.load_model()` 是 vLLM 中负责将模型加载到 GPU 的核心方法。整个加载过程涉及多个层次：模型实例化、权重加载、权重处理、以及优化包装。

## 整体流程图

```
GPUModelRunner.load_model()
    │
    ├─> 1. 获取 ModelLoader
    │       └─> get_model_loader(load_config)
    │           └─> 根据 load_format 选择对应的 Loader:
    │               - DefaultModelLoader (默认)
    │               - BitsAndBytesModelLoader
    │               - GGUFModelLoader
    │               - TensorizerLoader
    │               - ShardedStateLoader
    │               - DummyModelLoader
    │
    ├─> 2. BaseModelLoader.load_model()
    │       │
    │       ├─> 2.1 初始化模型结构
    │       │   └─> initialize_model(vllm_config, model_config, prefix)
    │       │       │
    │       │       ├─> get_model_architecture(model_config)
    │       │       │   └─> 从注册表中获取模型类:
    │       │       │       - _TEXT_GENERATION_MODELS (文本生成)
    │       │       │       - _EMBEDDING_MODELS (嵌入模型)
    │       │       │       - _MULTIMODAL_MODELS (多模态模型)
    │       │       │
    │       │       └─> model_class(vllm_config=vllm_config, prefix=prefix)
    │       │           └─> 实例化模型（此时权重未加载）
    │       │
    │       ├─> 2.2 加载权重
    │       │   └─> loader.load_weights(model, model_config)
    │       │       │
    │       │       ├─> get_all_weights(model_config, model)
    │       │       │   ├─> _prepare_weights() - 准备权重文件
    │       │       │   │   ├─> 检查本地是否存在
    │       │       │   │   ├─> 不存在则从 HuggingFace 下载
    │       │       │   │   └─> 确定权重格式 (safetensors/bin/pt)
    │       │       │   │
    │       │       │   └─> _get_weights_iterator(source)
    │       │       │       ├─> safetensors_weights_iterator()
    │       │       │       ├─> pt_weights_iterator()
    │       │       │       └─> multi_thread_safetensors_weights_iterator()
    │       │       │
    │       │       └─> model.load_weights(weights_iterator)
    │       │           └─> 将权重张量加载到模型参数
    │       │
    │       └─> 2.3 权重后处理
    │           └─> process_weights_after_loading(model, model_config, device)
    │               ├─> 量化方法处理
    │               │   └─> quant_method.process_weights_after_loading()
    │               └─> 注意力权重初始化
    │                   └─> Attention.process_weights_after_loading()
    │
    ├─> 3. LoRA 模型加载（如果配置）
    │   └─> load_lora_model(model, vllm_config, device)
    │
    ├─> 4. Drafter 模型加载（推测解码）
    │   └─> drafter.load_model(model)
    │
    ├─> 5. MoE 模型处理
    │   └─> EPLB 状态初始化
    │       └─> eplb_state.add_model(moe_model, model_config)
    │
    ├─> 6. 通信缓冲区准备
    │   └─> prepare_communication_buffer_for_model(model)
    │
    └─> 7. 模型包装（优化）
        ├─> CUDAGraphWrapper (CUDAGraph 优化)
        └─> UBatchWrapper (Micro-batching 优化)
```

## 详细步骤解析

### 1. 模型加载器选择 (Model Loader Selection)

**文件位置**: [vllm/model_executor/model_loader/__init__.py](vllm/model_executor/model_loader/__init__.py)

根据 `load_config.load_format` 选择合适的加载器：

```python
_LOAD_FORMAT_TO_MODEL_LOADER = {
    "auto": DefaultModelLoader,           # 自动选择
    "hf": DefaultModelLoader,             # HuggingFace 格式
    "safetensors": DefaultModelLoader,    # SafeTensors 格式
    "bitsandbytes": BitsAndBytesModelLoader,  # 量化模型
    "gguf": GGUFModelLoader,              # GGUF 格式
    "tensorizer": TensorizerLoader,        # Tensorizer 序列化
    "sharded_state": ShardedStateLoader,   # 分片状态
    "dummy": DummyModelLoader,            # 虚拟权重（测试用）
}
```

**关键代码**:
```python
def get_model_loader(load_config: LoadConfig) -> BaseModelLoader:
    load_format = load_config.load_format
    return _LOAD_FORMAT_TO_MODEL_LOADER[load_format](load_config)
```

### 2. 模型实例化 (Model Initialization)

**文件位置**: [vllm/model_executor/model_loader/utils.py:40-96](vllm/model_executor/model_loader/utils.py#L40-L96)

#### 2.1 获取模型类

```python
def initialize_model(vllm_config, model_config, prefix):
    # 1. 从注册表获取模型类
    model_class, _ = get_model_architecture(model_config)

    # 2. 配置量化（如果需要）
    if vllm_config.quant_config:
        configure_quant_config(vllm_config.quant_config, model_class)

    # 3. 实例化模型
    model = model_class(vllm_config=vllm_config, prefix=prefix)
    return model
```

#### 2.2 模型注册表机制

**文件位置**: [vllm/model_executor/models/registry.py](vllm/model_executor/models/registry.py)

vLLM 维护了一个架构到模型类的映射：

```python
_TEXT_GENERATION_MODELS = {
    "LlamaForCausalLM": ("llama", "LlamaForCausalLM"),
    "Qwen2ForCausalLM": ("qwen2", "Qwen2ForCausalLM"),
    "MistralForCausalLM": ("llama", "MistralForCausalLM"),
    # ... 更多模型
}

_MULTIMODAL_MODELS = {
    "LlavaForConditionalGeneration": ("llava", "LlavaForConditionalGeneration"),
    "Qwen2VLForConditionalGeneration": ("qwen2_vl", "Qwen2VLForConditionalGeneration"),
    # ... 更多多模态模型
}
```

**解析流程**:
```python
def _get_model_architecture(model_config):
    architectures = model_config.hf_config.architectures

    # 从注册表解析模型类
    model_cls, arch = model_config.registry.resolve_model_cls(
        architectures,
        model_config=model_config,
    )

    # 类型转换（嵌入模型/分类模型）
    if model_config.convert_type == "embed":
        model_cls = as_embedding_model(model_cls)
    elif model_config.convert_type == "classify":
        model_cls = as_seq_cls_model(model_cls)

    return model_cls, arch
```

### 3. 权重加载 (Weights Loading)

**文件位置**: [vllm/model_executor/model_loader/default_loader.py:382-413](vllm/model_executor/model_loader/default_loader.py#L382-L413)

#### 3.1 权重准备

```python
def _prepare_weights(model_name_or_path, subfolder, revision, ...):
    # 1. 检查是否为本地路径
    is_local = os.path.isdir(model_name_or_path)

    # 2. 如果不是本地，从 HuggingFace 下载
    if not is_local:
        download_weights_from_hf(model_name_or_path, revision)

    # 3. 确定权重文件格式
    if load_format == "safetensors":
        allow_patterns = ["*.safetensors"]
    elif load_format == "pt":
        allow_patterns = ["*.pt"]
    else:  # auto/hf
        allow_patterns = ["*.safetensors", "*.bin"]

    return model_path, weight_files, use_safetensors
```

#### 3.2 权重迭代器

根据权重格式选择不同的迭代器：

**SafeTensors 格式**:
```python
def safetensors_weights_iterator(weight_files):
    """逐个加载 safetensors 文件中的张量"""
    for weight_file in weight_files:
        with safe_open(weight_file, framework="pt") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                yield key, tensor
```

**PyTorch 格式**:
```python
def pt_weights_iterator(weight_files):
    """加载 .pt 或 .bin 文件"""
    for weight_file in weight_files:
        weights = torch.load(weight_file, map_location="cpu")
        for key, tensor in weights.items():
            yield key, tensor
```

**多线程加载** (提升大模型加载速度):
```python
def multi_thread_safetensors_weights_iterator(weight_files, num_threads=8):
    """使用多线程并行加载权重文件"""
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(load_file, f) for f in weight_files]
        for future in as_completed(futures):
            weights = future.result()
            for key, tensor in weights.items():
                yield key, tensor
```

#### 3.3 权重注入模型

```python
def load_weights(model, model_config):
    # 获取权重迭代器
    weights = self.get_all_weights(model_config, model)

    # 调用模型的 load_weights 方法
    loaded_weights = model.load_weights(weights)

    # 检查是否有未加载的权重
    if enable_weights_track:
        self.track_weights_loading(model, loaded_weights)
```

**模型的 load_weights 实现** (示例: LlamaForCausalLM):

```python
def load_weights(self, weights_iterator):
    loaded_params = set()

    for name, loaded_weight in weights_iterator:
        # 参数名映射
        param_name = name.replace("model.", "")

        # 查找对应的参数
        if param_name in self.state_dict():
            param = self.state_dict()[param_name]

            # 处理张量并行
            if is_tensor_parallel(param_name):
                loaded_weight = shard_weight(loaded_weight, tp_rank)

            # 加载权重
            param.data.copy_(loaded_weight)
            loaded_params.add(param_name)

    return loaded_params
```

### 4. 权重后处理 (Post-processing)

**文件位置**: [vllm/model_executor/model_loader/utils.py:99-128](vllm/model_executor/model_loader/utils.py#L99-L128)

#### 4.1 量化权重处理

```python
def process_weights_after_loading(model, model_config, target_device):
    for name, module in model.named_modules():
        quant_method = getattr(module, "quant_method", None)

        if quant_method:
            # 重新打包权重为内核友好格式
            quant_method.process_weights_after_loading(module)
```

**示例: AWQ 量化**:
```python
class AWQQuantMethod:
    def process_weights_after_loading(self, module):
        # 1. 反量化权重
        weight = self.dequantize(module.weight)

        # 2. 应用量化优化
        qweight = self.optimize_for_kernel(weight)

        # 3. 替换参数
        module.weight = torch.nn.Parameter(qweight)
```

#### 4.2 注意力权重优化

```python
# 初始化注意力层专用权重
for module in model.modules():
    if isinstance(module, Attention):
        module.process_weights_after_loading(dtype)
```

### 5. 高级特性

#### 5.1 Expert Parallelism (EP) 权重过滤

**文件位置**: [vllm/model_executor/model_loader/default_loader.py:318-379](vllm/model_executor/model_loader/default_loader.py#L318-L379)

对于 MoE 模型，只加载当前 rank 需要的专家权重：

```python
def _init_ep_weight_filter(model_config):
    """计算当前 rank 需要的专家 ID"""
    if model_config.is_moe and enable_expert_parallel:
        num_experts = model_config.get_num_experts()
        ep_size = dp_size * tp_size
        ep_rank = compute_ep_rank()

        # 计算本地专家 ID
        self.local_expert_ids = compute_local_expert_ids(
            num_experts, ep_size, ep_rank
        )
```

在权重加载时过滤：
```python
def _get_weights_iterator(source):
    for name, tensor in weights_iterator:
        # EP 权重过滤
        if self.local_expert_ids and is_expert_weight(name):
            expert_id = extract_expert_id(name)
            if expert_id not in self.local_expert_ids:
                continue  # 跳过非本地专家权重

        yield name, tensor
```

#### 5.2 LoRA 模型加载

```python
if self.lora_config:
    self.model = self.load_lora_model(
        self.model, self.vllm_config, self.device
    )
```

#### 5.3 推测解码 (Speculative Decoding)

```python
if hasattr(self, "drafter"):
    # 加载 drafter 模型（如 EAGLE, Medusa）
    self.drafter.load_model(self.model)

    # EAGLE3 辅助层配置
    if self.use_aux_hidden_state_outputs:
        aux_layers = self._get_eagle3_aux_layers_from_config()
        self.model.set_aux_hidden_state_layers(aux_layers)
```

### 6. 模型优化包装

**文件位置**: [vllm/v1/worker/gpu_model_runner.py:5130-5148](vllm/v1/worker/gpu_model_runner.py#L5130-L5148)

#### 6.1 CUDAGraph 包装

```python
if cudagraph_mode.has_full_cudagraphs():
    self.model = CUDAGraphWrapper(
        self.model,
        self.vllm_config,
        runtime_mode=CUDAGraphMode.FULL
    )
```

**优势**:
- 减少 CPU 开销
- 加速小批量推理
- 固定计算图优化

#### 6.2 Micro-batching 包装

```python
elif self.parallel_config.use_ubatching:
    self.model = UBatchWrapper(
        self.model,
        self.vllm_config,
        CUDAGraphMode.FULL,
        self.device
    )
```

**用途**: 动态批处理优化

### 7. 内存管理与错误处理

**文件位置**: [vllm/v1/worker/gpu_model_runner.py:5074-5084](vllm/v1/worker/gpu_model_runner.py#L5074-L5084)

```python
try:
    with DeviceMemoryProfiler() as m:
        time_before_load = time.perf_counter()

        # 加载模型
        self.model = model_loader.load_model(...)

        time_after_load = time.perf_counter()

    self.model_memory_usage = m.consumed_memory

except torch.cuda.OutOfMemoryError as e:
    logger.error(
        "Failed to load model - not enough GPU memory. "
        "Try lowering --gpu-memory-utilization"
    )
    raise e
```

**内存优化建议**:
- 降低 `--gpu-memory-utilization`
- 增加 `--tensor-parallel-size`
- 使用量化 `--quantization`

## 完整调用链路示例

以加载 Llama-3-70B 为例：

```
1. 用户调用 vllm serve meta-llama/Llama-3-70B

2. GPUModelRunner.load_model()
   └─> get_model_loader(LoadConfig(load_format="auto"))
       └─> DefaultModelLoader()

3. BaseModelLoader.load_model()
   ├─> initialize_model()
   │   ├─> get_model_architecture()
   │   │   └─> registry.resolve_model_cls(["LlamaForCausalLM"])
   │   │       └─> 返回: (LlamaForCausalLM, "LlamaForCausalLM")
   │   └─> LlamaForCausalLM(vllm_config, prefix="")
   │       └─> 实例化模型结构（权重为空）
   │
   ├─> load_weights()
   │   ├─> _prepare_weights()
   │   │   └─> 下载/定位 model-00001-of-00030.safetensors
   │   │       到 model-00030-of-00030.safetensors
   │   │
   │   ├─> get_all_weights()
   │   │   └─> multi_thread_safetensors_weights_iterator()
   │   │       └─> 并行读取 30 个 safetensors 文件
   │   │
   │   └─> model.load_weights(weights)
   │       └─> 逐层加载权重到 GPU
   │
   └─> process_weights_after_loading()
       └─> 无量化，跳过

4. 返回到 GPUModelRunner.load_model()
   ├─> 记录内存使用: 140 GiB
   ├─> prepare_communication_buffer_for_model()
   └─> CUDAGraphWrapper 包装模型

5. 模型加载完成，准备推理
```

## 关键数据结构

### ModelConfig
```python
@dataclass
class ModelConfig:
    model: str                    # 模型名称或路径
    revision: str                 # 版本
    architectures: List[str]      # 模型架构列表
    dtype: torch.dtype           # 数据类型
    quantization: str            # 量化方法
    is_moe: bool                 # 是否为 MoE 模型
```

### LoadConfig
```python
@dataclass
class LoadConfig:
    load_format: str              # 加载格式
    device: str                   # 目标设备
    safetensors_load_strategy    # 加载策略
    model_loader_extra_config    # 额外配置
```

### VllmConfig
```python
@dataclass
class VllmConfig:
    model_config: ModelConfig
    load_config: LoadConfig
    parallel_config: ParallelConfig
    cache_config: CacheConfig
    quant_config: QuantConfig
```

## 性能优化技巧

### 1. 多线程加载
```bash
# 启用多线程权重加载
--load-format safetensors \
--model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 16}'
```

### 2. 内存映射
```bash
# 使用 safetensors 内存映射模式
--load-format safetensors \
--model-loader-extra-config '{"mmap": true}'
```

### 3. 预下载
```python
from vllm.model_executor.model_loader import get_model_loader

loader = get_model_loader(load_config)
loader.download_model(model_config)  # 预先下载
```

## 常见问题

### Q1: 为什么加载大模型很慢？
**A**: 使用多线程加载和更快的存储：
```bash
--load-format safetensors \
--model-loader-extra-config '{"enable_multithread_load": true}'
```

### Q2: 如何减少 GPU 内存占用？
**A**: 使用量化或张量并行：
```bash
--quantization awq  # 或 gptq, fp8
--tensor-parallel-size 4
```

### Q3: 权重加载失败怎么办？
**A**: 检查权重完整性：
```python
from safetensors import safe_open
with safe_open("model.safetensors", framework="pt") as f:
    print(f.keys())  # 查看权重名称
```

## 总结

`GPUModelRunner.load_model()` 的核心流程：

1. **选择加载器** - 根据格式选择合适的 ModelLoader
2. **实例化模型** - 从注册表获取模型类并创建实例
3. **加载权重** - 下载/读取权重文件并注入模型
4. **后处理** - 量化优化、注意力权重处理
5. **优化包装** - CUDAGraph、Micro-batching 等优化

整个设计采用了分层架构，支持多种权重格式、量化方法和并行策略，是 vLLM 高性能推理的基础。
