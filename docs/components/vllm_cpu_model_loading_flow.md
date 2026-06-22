# vLLM CPU 版本模型加载流程详解

## 📋 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                    API Server (主进程)                        │
│                  PID: 815636                                  │
│              vllm.entrypoints.openai.api_server              │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                  EngineCore 进程                             │
│                  PID: 815951                                  │
│              调度器 + KV Cache 管理                           │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                   Worker 进程                                │
│                  PID: 816085                                  │
│            模型加载 + 推理执行                                │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔍 模型加载详细流程

### 第 1 步: Worker 初始化
**文件**: `vllm/v1/worker/cpu_worker.py`

```python
class CPUWorker(Worker):
    def __init__(self, vllm_config, ...):
        # 1. 检查 CPU 内存
        memory_status = get_memory_node_info(cpu_core.numa_node)

        # 2. 设置内存利用率
        memory_fraction = vllm_config.cache_config.gpu_memory_utilization
        self.requested_cpu_memory = ceil(memory_status.total_memory * memory_fraction)

        # 3. 检查是否有足够内存
        if self.requested_cpu_memory > available_memory:
            raise ValueError("内存不足...")

        # 4. 初始化父类 Worker
        super().__init__(...)
```

---

### 第 2 步: 初始化设备
**文件**: `vllm/v1/worker/cpu_worker.py`

```python
def init_device(self):
    # 1. 检查预加载的库 (tcmalloc, libiomp)
    check_preloaded_libs("libtcmalloc")
    check_preloaded_libs("libiomp")

    # 2. 设置 OpenMP 线程数
    torch.set_num_threads = skip_set_num_threads

    # 3. 初始化分布式环境
    init_worker_distributed_environment(
        self.vllm_config,
        self.rank,
        self.distributed_init_method,
        self.local_rank,
        current_platform.dist_backend,  # "gloo" for CPU
    )

    # 4. 设置随机种子
    set_random_seed(self.model_config.seed)

    # 5. 创建 ModelRunner
    self.model_runner = CPUModelRunner(
        self.vllm_config, torch.device("cpu")
    )
```

---

### 第 3 步: CPUModelRunner 初始化
**文件**: `vllm/v1/worker/cpu_model_runner.py`

```python
class CPUModelRunner(GPUModelRunner):
    def __init__(self, vllm_config, device):
        # 1. 调用父类初始化（GPU 版本）
        with _torch_cuda_wrapper():
            super().__init__(vllm_config, device)

        # 2. 确保设备是 CPU
        assert device == torch.device("cpu")

        # 3. 禁用 CUDA graph
        self.use_cuda_graph = False
        self.cascade_attn_enabled = False

        # 4. 后处理张量：将 GPU 张量替换为 CPU 张量
        self._postprocess_tensors()

        # 5. 替换 Triton 内核为 CPU 实现
        self._postprocess_triton()
```

**关键步骤 - 替换张量**:
```python
def _postprocess_tensors(self):
    # 将所有 GPU buffer 替换为 CPU buffer
    for v in vars(self).values():
        if isinstance(v, CpuGpuBuffer):
            v.gpu = v.cpu  # GPU 指针指向 CPU 数据

    # 替换 input_batch 中的张量
    for k, v in vars(self.input_batch).items():
        if k.endswith("_cpu_tensor"):
            replace_tensor(self.input_batch, k, k[:-11])
```

**关键步骤 - 替换内核**:
```python
def _postprocess_triton(self):
    # 替换 Triton GPU 内核为 CPU 实现
    vllm.v1.worker.block_table._compute_slot_mapping_kernel = \
        cpu_tl.compute_slot_mapping_kernel

    # 替换其他 GPU 内核...
```

---

### 第 4 步: 加载模型
**文件**: `vllm/v1/worker/cpu_model_runner.py`

```python
@instrument(span_name="Loading (CPU)")
def load_model(self, load_dummy_weights=False):
    if load_dummy_weights:
        raise ValueError("CPU 不支持加载虚拟权重")

    logger.info("Starting to load model %s...", self.model_config.model)

    # 🔑 关键调用：使用 get_model 加载模型
    self.model = get_model(vllm_config=self.vllm_config)

    # 加载 LoRA（如果有）
    if self.lora_config:
        self.model = self.load_lora_model(...)

    # 加载 drafter（如果有）
    if hasattr(self, "drafter"):
        self.drafter.load_model(self.model)
```

---

### 第 5 步: get_model 函数
**文件**: `vllm/model_executor/model_loader/__init__.py`

```python
def get_model(*, vllm_config, model_config=None, prefix="", load_config=None):
    # 1. 获取模型加载器
    loader = get_model_loader(load_config or vllm_config.load_config)

    # 2. 加载模型
    return loader.load_model(
        vllm_config=vllm_config,
        model_config=model_config,
        prefix=prefix
    )

def get_model_loader(load_config):
    # 根据 load_format 选择加载器
    # "auto" -> DefaultModelLoader
    # "safetensors" -> DefaultModelLoader
    # "gguf" -> GGUFModelLoader
    # ...
    return _LOAD_FORMAT_TO_MODEL_LOADER[load_format](load_config)
```

---

### 第 6 步: DefaultModelLoader.load_model
**文件**: `vllm/model_executor/model_loader/base_loader.py`

```python
@instrument(span_name="Load model")
def load_model(self, vllm_config, model_config, prefix=""):
    # 1. 确定加载设备
    load_device = vllm_config.device_config.device  # "cpu"

    # 2. 设置默认数据类型
    with set_default_torch_dtype(model_config.dtype):  # float16
        with target_device:  # torch.device("cpu")
            # 3. 初始化模型架构
            model = initialize_model(
                vllm_config=vllm_config,
                model_config=model_config,
                prefix=prefix,
            )

        # 4. 加载权重
        logger.debug("Loading weights on %s ...", load_device)
        self.load_weights(model, model_config)

        # 5. 处理权重（量化、打包等）
        process_weights_after_loading(model, model_config, target_device)

    return model.eval()
```

---

### 第 7 步: 初始化模型架构
**文件**: `vllm/model_executor/model_loader/utils.py`

```python
@instrument(span_name="Initialize model")
def initialize_model(vllm_config, *, prefix="", model_class=None, model_config=None):
    # 1. 获取模型类（根据 architecture）
    if model_class is None:
        model_class, _ = get_model_architecture(model_config)

    # 2. 配置量化（如果有）
    if vllm_config.quant_config is not None:
        configure_quant_config(vllm_config.quant_config, model_class)

    # 3. 实例化模型
    with set_current_vllm_config(vllm_config, check_compile=True, prefix=prefix):
        model = model_class(vllm_config=vllm_config, prefix=prefix)

    return model
```

**模型类查找过程**:
```python
def _get_model_architecture(model_config):
    # 从 HuggingFace config 获取 architecture
    architectures = model_config.hf_config.architectures
    # 例如: ["OPTForCausalLM"]

    # 根据 architecture 找到对应的 vLLM 模型类
    model_cls, arch = model_config.registry.resolve_model_cls(
        architectures,
        model_config=model_config,
    )
    # 返回: (OPTForCausalLM, "OPTForCausalLM")

    return model_cls, arch
```

---

### 第 8 步: 加载权重
**文件**: `vllm/model_executor/model_loader/default_loader.py`

```python
@instrument(span_name="Load weights")
def load_weights(self, model, model_config):
    # 1. 初始化 EP 权重过滤（如果启用）
    self._init_ep_weight_filter(model_config)

    # 2. 获取所有需要加载的权重
    weights_to_load = {name for name, _ in model.named_parameters()}

    # 3. 加载权重
    loaded_weights = model.load_weights(
        self.get_all_weights(model_config, model)
    )

    logger.info("Loading weights took %.2f seconds", ...)
```

**获取权重迭代器**:
```python
def get_all_weights(self, model_config, model):
    # 1. 准备主权重源
    primary_weights = DefaultModelLoader.Source(
        model_config.model,      # 模型名称或路径
        model_config.revision,   # 版本
        prefix="",
    )

    # 2. 下载/准备权重文件
    hf_folder, hf_weights_files, use_safetensors = \
        self._prepare_weights(...)

    # 3. 返回权重迭代器
    if use_safetensors:
        weights_iterator = safetensors_weights_iterator(
            hf_weights_files,
            self.load_config.use_tqdm_on_load,
        )
    else:
        weights_iterator = pt_weights_iterator(
            hf_weights_files,
            self.load_config.use_tqdm_on_load,
        )

    return weights_iterator
```

---

### 第 9 步: 处理权重
**文件**: `vllm/model_executor/model_loader/utils.py`

```python
def process_weights_after_loading(model, model_config, target_device):
    # 1. 处理量化权重
    for name, module in model.named_modules():
        quant_method = getattr(module, "quant_method", None)
        if isinstance(quant_method, QuantizeMethodBase):
            # 将权重移到设备，处理后移回
            with device_loading_context(module, target_device):
                quant_method.process_weights_after_loading(module)

    # 2. 初始化 attention 权重
    for name, module in model.named_modules():
        if isinstance(module, (Attention, MLAAttention)):
            if hasattr(module, "process_weights_after_loading"):
                with device_loading_context(module, target_device):
                    module.process_weights_after_loading(model_config.dtype)
```

**CPU 特殊处理**:
```python
@contextmanager
def device_loading_context(module, target_device):
    if target_device.type == "cpu":
        # 如果目标设备是 CPU，无需移动
        yield module
        return

    # GPU 情况：CPU -> GPU -> 处理 -> GPU -> CPU
    # ... 省略 GPU 处理逻辑 ...
```

---

## 🎯 CPU 版本的关键差异

### 1. **设备选择**
- GPU: `torch.device("cuda")`
- CPU: `torch.device("cpu")`

### 2. **内存管理**
- GPU: 使用 `torch.cuda.memory_allocated()`
- CPU: 使用 `psutil` 检查系统内存

### 3. **分布式后端**
- GPU: `nccl`
- CPU: `gloo`

### 4. **内核替换**
- GPU: Triton CUDA 内核
- CPU: C++/OpenMP 实现

### 5. **张量存储**
- GPU: 在 GPU 内存中
- CPU: 在系统 RAM 中

---

## 📝 实际加载流程示例（distilgpt2）

```
1. CPUWorker.__init__
   └─ 检查内存: 15.47 GiB 总内存，使用 20% = 3.09 GiB

2. CPUWorker.init_device
   └─ 设置分布式后端: gloo
   └─ 创建 CPUModelRunner

3. CPUModelRunner.__init__
   └─ 替换 GPU tensors -> CPU tensors
   └─ 替换 Triton kernels -> CPU implementations

4. CPUModelRunner.load_model
   └─ get_model(vllm_config)

5. DefaultModelLoader.load_model
   └─ initialize_model -> OPTForCausalLM(vllm_config)
   └─ load_weights -> 从 HuggingFace 下载权重
   └─ process_weights_after_loading -> (CPU: 无操作)

6. 模型已加载到 CPU 内存
   └─ 权重类型: torch.float16
   └─ 设备: torch.device("cpu")
   └─ 内存占用: ~300 MB (distilgpt2)
```

---

## 🔧 调试技巧

### 1. 查看模型加载日志
```bash
export VLLM_LOGGING_LEVEL=DEBUG
python -m vllm.entrypoints.openai.api_server ...
```

### 2. 查看模型结构
```bash
export VLLM_LOG_MODEL_INSPECTION=1
```

### 3. 设置断点
在以下位置设置断点：
- `vllm/v1/worker/cpu_worker.py:141` (创建 ModelRunner)
- `vllm/v1/worker/cpu_model_runner.py:102` (load_model)
- `vllm/model_executor/model_loader/base_loader.py:43` (load_model)
- `vllm/model_executor/model_loader/default_loader.py:368` (load_weights)

### 4. 检查内存使用
```python
import psutil
print(psutil.virtual_memory())
```

---

## 📚 相关文件清单

| 文件 | 作用 |
|------|------|
| `vllm/v1/worker/cpu_worker.py` | CPU Worker 主类 |
| `vllm/v1/worker/cpu_model_runner.py` | CPU 模型运行器 |
| `vllm/model_executor/model_loader/__init__.py` | 模型加载入口 |
| `vllm/model_executor/model_loader/base_loader.py` | 基础加载器 |
| `vllm/model_executor/model_loader/default_loader.py` | 默认加载器 |
| `vllm/model_executor/model_loader/utils.py` | 加载工具函数 |
| `vllm/model_executor/model_loader/weight_utils.py` | 权重加载工具 |
| `vllm/model_executor/models/` | 模型实现目录 |
