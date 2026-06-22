# vLLM 核心架构文档

本目录包含 vLLM 核心架构和功能的详细分析文档。

---

## 📚 文档列表

### 1. [vLLM 核心组件架构与处理流程深度分析](vllm_component_architecture_and_workflow.md)
**文件大小**: 120KB+ | **创建时间**: 2026-06-20

**核心内容**:
- vLLM 核心组件抽象（9 大组件）
- 引擎启动流程（5 阶段详解）
- 请求处理流程
- 推理执行流程
- 7 个 Mermaid 图

**适用人群**: 核心开发者、架构师

---

### 2. [vLLM 深度分析：模型适配、权重加载与推理请求处理](vLLM_Analysis.md) ⭐ 新增
**文件大小**: 97KB+ | **创建时间**: 2026-06-20

**核心内容**:
- 模型适配机制（注册表模式）
- 权重加载流程详解
- 推理请求处理流程
- Qwen3ForCausalLM 实例分析
- 模型配置与架构验证

**适用人群**: 模型开发者、架构理解者

---

### 3. [vLLM 完整架构图](vLLM_Architecture_Diagrams.md) ⭐ 新增
**文件大小**: 45KB+ | **创建时间**: 2026-06-20

**核心内容**:
- 整体架构请求处理流程图
- 模型加载流程图
- 推理执行流程图
- KV Cache 管理流程图
- 分布式推理架构图

**适用人群**: 架构可视化理解者、新人入门

---

### 4. [vLLM Multiprocess 架构设计详解](vllm_multiprocess_architecture_design.md)
**文件大小**: 147KB | **创建时间**: 2026-06-13

**核心内容**:
- 多进程架构设计
- WorkerProc 进程管理
- 推理任务调度流程
- CUDA Graph 优化
- KV Offload 配置

**适用人群**: 核心开发者、性能优化工程师

---

### 5. [vLLM 推测解码架构与实现详解](vllm_speculative_decoding_architecture.md)
**文件大小**: ~45KB | **创建时间**: 2026-06-20

**核心内容**:
- 推测解码原理（Draft + Verify + Accept/Reject）
- 8 种推测方法（Ngram, Eagle, Medusa, DFlash 等）
- 详细工作流程（5 个步骤）
- vLLM vs vLLM-Ascend 实现对比
- 性能优化策略

**适用人群**: 推理加速开发者

---

### 6. [vLLM 结构化输出架构与实现详解](vllm_structured_output_architecture.md)
**文件大小**: ~25KB | **创建时间**: 2026-06-20

**核心内容**:
- 结构化输出原理
- 6 种输出类型（JSON, Regex, Grammar 等）
- 4 种后端对比（Guidance, Outlines, XGrammar, LMFormatEnforcer）
- 详细工作流程（6 个步骤）
- 使用示例

**适用人群**: API 开发者

---

### 7. [vLLM KV Offload 架构与实现详解](vllm_kv_offload_architecture.md)
**文件大小**: ~10KB | **创建时间**: 2026-06-20

**核心内容**:
- KV Offload 原理
- 3 种 Offload 策略
- 核心组件详解
- 使用示例

**适用人群**: 长序列处理开发者

---

## 📊 统计信息

- **文档总数**: 7 个
- **总行数**: ~4,400 行
- **覆盖模块**: AsyncLLM, EngineCore, Scheduler, Worker, Speculative Decoding, Structured Output, 模型适配

---

**返回**: [主文档索引](../README.md)