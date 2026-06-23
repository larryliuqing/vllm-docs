# vLLM 测试文档

本目录包含 vLLM 及 vllm-ascend 的测试报告、测试指南和相关文档。

---

## 📁 子目录结构

```
tests/
├── README.md           # 本文件
├── multimodal/         # 多模态模型测试
├── omni/               # Omni 模型测试
└── ascend/             # Ascend NPU 测试
```

---

## 📚 文档分类

### 一、多模态测试 ([multimodal/](multimodal/))

| 文档 | 说明 |
|------|------|
| [Qwen2-VL-7B测试指南.md](multimodal/Qwen2-VL-7B测试指南.md) | Qwen2-VL-7B 模型测试指南 |
| [Qwen2-VL-7B完整测试报告.md](multimodal/Qwen2-VL-7B完整测试报告.md) | Qwen2-VL-7B 完整测试报告 |
| [Qwen3-VL-32B测试指南.md](multimodal/Qwen3-VL-32B测试指南.md) | Qwen3-VL-32B 模型测试指南 |
| [Qwen3-VL-32B失败报告.md](multimodal/Qwen3-VL-32B失败报告.md) | Qwen3-VL-32B 测试失败分析 |
| [vLLM多模态对比测试方案.md](multimodal/vLLM多模态对比测试方案.md) | 多模态对比测试方案 |
| [多模态模型推荐与下载指南.md](multimodal/多模态模型推荐与下载指南.md) | 多模态模型推荐和下载方法 |

---

### 二、Omni 测试 ([omni/](omni/))

| 文档 | 说明 |
|------|------|
| [vllm-ascend_vs_vllm-omni对比测试.md](omni/vllm-ascend_vs_vllm-omni对比测试.md) | vllm-ascend 与 vllm-omni 对比测试 |
| [vllm镜像最终对比报告.md](omni/vllm镜像最终对比报告.md) | vLLM Docker 镜像对比报告 |

---

### 三、Ascend NPU 测试 ([ascend/](ascend/))

| 文档 | 说明 |
|------|------|
| [vllm_昇腾NPU测试指南.md](ascend/vllm_昇腾NPU测试指南.md) | 昇腾 NPU 单卡/多卡测试指南 |

---

## 📊 统计信息

- **文档总数**: 9 个
- **测试模型**: Qwen2-VL-7B, Qwen3-VL-32B
- **测试平台**: NVIDIA GPU, Ascend NPU

---

**返回**: [主文档索引](../README.md)
