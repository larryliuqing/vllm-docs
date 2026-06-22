# vLLM-Ascend 架构文档

本目录包含 vLLM-Ascend 项目的架构分析、功能对比和 Ascend 特有实现文档。

---

## 📚 文档列表

### 1. [vLLM-Ascend 核心组件架构与处理流程深度分析](vllm_ascend_component_architecture_and_workflow.md)
**文件大小**: 150KB+ | **创建时间**: 2026-06-20

**核心内容**:
- vLLM-Ascend 插件化架构（6 大层级）
- NPUPlatform 平台层实现
- 5 种 Attention Backend（标准/MLA/DSA/SFA/FA3）
- 分布式通信（HCCL + KV Transfer + FlashComm）
- 量化实现（14 种方法）
- Patch 机制（48 个文件）
- 8 个 Mermaid 图

**适用人群**: 核心开发者、架构师、Ascend 插件开发者

---

### 2. [vLLM 完整功能列表与 Ascend vs CUDA 实现对比分析](vllm_vllm_ascend_comparison.md)
**文件大小**: 54KB | **创建时间**: 2026-06-20

**核心内容**:
- vLLM 完整功能列表（12 大模块）
- vLLM-Ascend 架构概览和代码统计
- 功能重用清单（60% 重用 + 30% 适配 + 10% 重写）
- Patch 机制详细清单（48 文件）
- Ascend vs CUDA 详细对比
- Ascend 950 (Atlas A5) 支持情况
- 文件路径映射表

**适用人群**: 开发者、架构师、硬件插件开发者

---

### 3. [vLLM-Ascend 310P 专用实现详解](vllm_ascend_310p_implementation.md)
**文件大小**: ~2KB | **创建时间**: 2026-06-20

**核心内容**:
- Ascend 310P 硬件特性
- 310P 专用组件（ModelRunner310P, Worker310P）
- 使用限制

**适用人群**: 310P 部署工程师

---

### 4. [vLLM-Ascend XLite 轻量级推理架构详解](vllm_ascend_xlite_architecture.md)
**文件大小**: ~2KB | **创建时间**: 2026-06-20

**核心内容**:
- XLite 功能定位
- 核心组件（XLite, ModelRunner）
- 使用场景

**适用人群**: openEuler Xlite 用户

---

### 5. [vLLM-Ascend EPLB 专家并行负载均衡架构详解](vllm_ascend_eplb_architecture.md)
**文件大小**: ~4KB | **创建时间**: 2026-06-20

**核心内容**:
- EPLB 功能定位
- 负载均衡流程
- MoE 集成

**适用人群**: MoE 模型开发者

---

## 📊 统计信息

- **文档总数**: 5 个
- **总行数**: ~4,000 行
- **覆盖模块**: NPUPlatform, Attention Backends, Distributed, Quantization, 310P, XLite, EPLB

---

**返回**: [主文档索引](../README.md)
