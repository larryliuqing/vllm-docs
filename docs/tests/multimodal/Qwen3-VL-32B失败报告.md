# Qwen3-VL-32B-Instruct 测试失败报告

## 测试信息

| 项目 | 详情 |
|------|------|
| 模型 | Qwen3-VL-32B-Instruct |
| 模型大小 | 63GB (14个分片) |
| 配置 | 4卡 TP, max_model_len=8192 |
| 镜像 | vllm-omni:v0.20.2rc |
| 状态 | ❌ 启动失败 |

---

## 错误信息

### 主要错误

```
RuntimeError: Engine core initialization failed.
Exception: WorkerProc initialization failed due to an exception in a background process.
```

### 错误位置

- 模型识别: `Qwen3VLForConditionalGeneration` ✅
- 编译配置: 正常 ✅
- EngineCore 初始化: ❌ 失败
- WorkerProc 初始化: ❌ 失败

---

## 可能原因分析

### 1. 模型架构支持问题

Qwen3-VL-32B 是较新的模型（2024年底发布），可能：

- vLLM v0.20.2rc 对 Qwen3-VL 支持不完整
- Ascend NPU 后端对 Qwen3-VL 支持有限
- 模型架构与 vLLM-Ascend 不兼容

### 2. 资源问题

32B 模型需要：
- 约 64GB 显存（总）
- 每卡约 16GB
- 可能接近 NPU 显存上限

### 3. 配置问题

可能需要：
- 特殊的配置参数
- 更小的 max_model_len
- 不同的 tensor_parallel 配置

---

## 对比分析

### 已成功运行的模型

| 模型 | 参数量 | 架构 | 状态 |
|------|--------|------|------|
| Qwen2.5-Omni-7B | 7B | Qwen2_5OmniModel | ✅ 成功 |
| Qwen3-Omni-30B-A3B | 30B (MoE) | Qwen3OmniMoeForConditionalGeneration | ✅ 成功 |
| Qwen3-0.6B | 0.6B | Qwen3ForCausalLM | ✅ 成功 |

### 失败的模型

| 模型 | 参数量 | 架构 | 状态 |
|------|--------|------|------|
| **glm-5.13-VL-32B** | **32B** | **Qwen3VLForConditionalGeneration** | **❌ 失败** |

---

## 建议解决方案

### 方案1: 尝试 Qwen2-VL (推荐)

Qwen2-VL 系列已在 vLLM 上有广泛支持：

```bash
# 下载 Qwen2-VL-7B
modelscope download --model Qwen/Qwen2-VL-7B-Instruct --local_dir Qwen2-VL-7B-Instruct

# 启动
docker run --rm \
    --device=/dev/davinci5:/dev/davinci0 \
    vllm-omni:v0.20.2rc \
    python3 -m vllm.entrypoints.openai.api_server \
        --model /models/Qwen2-VL-7B-Instruct \
        --port 8003
```

### 方案2: 降低配置重试

```bash
# 尝试更小的 max_model_len
--max-model-len 4096

# 或降低显存利用率
--gpu-memory-utilization 0.85
```

### 方案3: 使用不同镜像

```bash
# 尝试 vllm-ascend 镜像
vllm-ascend:v0.20.2rc
```

---

## 后续行动

### 立即可行

1. ✅ **测试 Qwen2-VL-7B** - 更成熟的模型
2. ✅ **测试 Qwen2-VL-2B** - 轻量级验证

### 需要等待

1. ⏳ vLLM 更新版本对 Qwen3-VL 支持
2. ⏳ vLLM-Ascend 后端优化

---

## 技术细节

### 模型配置差异

**glm-5.13-VL-32B**:
```
Architecture: Qwen3VLForConditionalGeneration
Layers: 64
ACL Graph sizes: 9 (from 35)
Max batch size: 9
```

**glm-5.13-Omni-30B-A3B** (成功):
```
Architecture: Qwen3OmniMoeForConditionalGeneration
Layers: 48
ACL Graph sizes: 4
```

**关键差异**:
- Qwen3-VL 使用标准 Dense 架构
- Qwen3-Omni 使用 MoE 架构
- MoE 可能有更好的 NPU 兼容性

---

## 结论

**Qwen3-VL-32B-Instruct 当前无法在 vLLM-Ascend v0.20.2rc 上运行**

**推荐**:
1. 使用 Qwen2-VL 系列进行视觉理解测试
2. 使用 Qwen3-Omni 系列进行全模态测试
3. 等待 vLLM 后续版本更新

---

## 附录: 完整错误日志

见: `/home/bes/work/vllm-project/vllm_serve_qwen3_vl_32b_4npu.log`
