# vLLM-Ascend 算子集成架构详解

> 本文档详细解答 vLLM 与 vLLM-Ascend 之间的算子调用关系，阐述 vLLM-Ascend 如何通过 **Patch 机制**、**OOT 注册机制（PluggableLayer/CustomOp）** 来替换 vLLM 中的算子，实现昇腾 NPU 适配。

---

## 一、核心结论

**Q: vLLM 中的算子是否都需要在 vLLM-Ascend 中重新实现？**

**A: 不是！** 主要通过 3 种机制实现适配：

| 机制 | 说明 | 替换方式 | 比例 |
|------|------|---------|------|
| **Patch 机制** | Monkey Patch 直接替换 | 函数级别替换 | 27 Worker + 21 Platform |
| **PluggableLayer.register_oot** | 装饰器注册替换 | 类级别替换 | 核心 Layer（Linear, Attention, MoE） |
| **CustomOp.register_oot** | 自定义算子注册替换 | 兼容层 | 量化、激活函数等 |

**Q: 模型适配只在 vLLM 中吗？**

**A: 主要在 vLLM 中（281 个模型），vLLM-Ascend 有 3 个特有模型**（DeepSeek V4、DeepSeek V4 MTP）

**Q: vLLM 如何调用 vLLM-Ascend 中的算子？**

**A: 通过 3 种机制**：
1. **Patch 机制**: `import vllm.module; 替换函数` → vLLM 代码运行时自动使用 Ascend 实现
2. **PluggableLayer OOT**: `@PluggableLayer.register_oot` → vLLM 创建 Layer 时自动替换
3. **CustomOp OOT**: `@CustomOp.register_oot` → vLLM 创建 CustomOp 时自动替换

---

## 二、3 种集成机制详解

### 2.1 机制一：Patch 机制（Monkey Patch）

**原理**: 直接替换 vLLM 模块中的函数/方法

**工作流程**:
```
vLLM-Ascend 启动
  → NPUPlatform.pre_register_and_update()
    → adapt_patch(is_global_patch=True)
      → 导入 path/platform/*.py  → 全局 Monkey Patch

Worker 启动
  → NPUWorker.__init__()
    → adapt_patch(is_global_patch=False)
      → 导入 path/worker/*.py  → Worker 级 Monkey Patch
```

**典型 Patch 示例**（文件: `patch/worker/patch_triton.py`）:

```python
import vllm.model_executor.layers.mamba.ops.causal_conv1d
from vllm_ascend.ops.triton.mamba.causal_conv1d import causal_conv1d_fn

# 直接替换 vLLM 的 causal_conv1d 函数
vllm.model_executor.layers.mamba.ops.causal_conv1d.causal_conv1d_fn = causal_conv1d_fn
```

**说明**: vLLM 代码中使用 `causal_conv1d_fn` 时自动调用 Ascend 实现，无需修改 vLLM 源代码。

---

### 2.2 机制二：PluggableLayer.register_oot

**原理**: 替换 vLLM 中注册的 Layer 类

**核心代码**（文件: `vllm/model_executor/custom_op.py`）:

```python
class PluggableLayer(nn.Module):
    def __new__(cls, *args, **kwargs):
        layer_class_name = cls.__name__

        # 检查是否有 OOT 替换
        if layer_class_name not in op_registry_oot:
            layer_cls_to_instantiate = cls          # 使用原始类
        else:
            layer_cls_to_instantiate = op_registry_oot[layer_class_name]  # 使用替换类
        return super().__new__(layer_cls_to_instantiate)

    # 注册 OOT 替换
    @classmethod
    def register_oot(cls, name: str | None = None):
        def decorator(layer_cls):
            reg_name = name if name is not None else cls.__name__
            op_registry_oot[reg_name] = layer_cls
            return layer_cls
        return decorator
```

**示例**: vLLM-Ascend 如何注册自己的 MoE OOT 实现（文件: `vllm_ascend/utils.py`）:

```python
# vLLM-Ascend 启动时执行
from vllm.model_executor.custom_op import PluggableLayer
from vllm_ascend.ops.fused_moe import AscendFusedMoE

# 注册 Ascend 专用 MoE 实现
PluggableLayer.register_oot(AscendFusedMoE, name="FusedMoE")
```

**说明**: 当 vLLM 代码执行 `layer = FusedMoE(...)` 时，`FusedMoE.__new__` 会自动返回 `AscendFusedMoE` 实例。

---

### 2.3 机制三：CustomOp.register_oot

**原理**: 替换 vLLM 中注册的 CustomOp 类

**核心代码**（文件: `vllm/model_executor/custom_op.py`）:

```python
class CustomOp(nn.Module):
    def __new__(cls, *args, **kwargs):
        op_name = cls.__name__

        # 检查是否有 OOT 替换
        if op_name not in op_registry_oot:
            op_cls_to_instantiate = cls
        else:
            op_cls_to_instantiate = op_registry_oot[op_name]
        return super().__new__(op_cls_to_instantiate)

    def forward(self, *args, **kwargs):
        return self._forward_method(*args, **kwargs)

    def forward_cuda(self, *args, **kwargs):
        raise NotImplementedError

    def forward_oot(self, *args, **kwargs):
        return self.forward_native(*args, **kwargs)
```
