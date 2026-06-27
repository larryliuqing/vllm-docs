# vLLM 组件与机制详解文档

本目录包含 vLLM 各个组件的详细实现文档，涵盖算子、模型适配、PD分离等关键机制。

---

## 📚 文档列表

### 1. [vLLM 算子分类与调用机制详解](vllm_operator_classification.md)
**文件大小**: 15KB | **创建时间**: 2026-06-20

**核心内容**:
- vLLM与vLLM-Ascend算子全面分类
- 算子重新实现策略（决策树）
- 跨项目调用机制（Patch + Custom Op + Backend路由）
- 算子性能优化级别

**适用人群**: 算子开发者、Ascend适配开发者

---

### 2. [vLLM-Ascend 算子集成架构详解](vllm_ascend_operator_integration_arch.md)
**文件大小**: 13KB | **创建时间**: 2026-06-20

**核心内容**:
- 3 种集成机制（Patch/OOT/CustomOp）详解
- vLLM 调用 vLLM-Ascend 算子的完整路径
- 算子重新实现决策流程
- 关键源码节点标注

**适用人群**: 快速理解算子集成机制

---

### 3. [vLLM 模型适配指南](vllm_model_adaptation_guide.md)
**文件大小**: 26KB | **创建时间**: 2026-06-20

**核心内容**:
- 模型适配原理与架构
- 详细适配步骤（7步）
- Ascend NPU适配方法
- 特殊案例（DeepSeek V4、Qwen3-VL、MiniMax-M2）

**适用人群**: 模型开发者、新模型适配者

---

### 4. [vLLM 算子与模型适配架构](vllm_operator_and_model_adaptation.md)
**文件大小**: 27KB | **创建时间**: 2026-06-20

**核心内容**:
- 算子集成架构（Patch/OOT/CustomOp三种机制）
- 模型适配流程
- 跨项目调用关系

**适用人群**: 架构理解、适配开发者

---

### 5. [PD分离（Prefill-Decode Disaggregation）技术文档](pd_separation_architecture.md)
**文件大小**: 42KB | **创建时间**: 2026-06-20

**核心内容**:
- PD分离背景与架构
- vLLM中的KV Transfer实现
- vLLM-Ascend专用Connector体系
- Mooncake/HIXL传输引擎集成
- 测试场景与硬件验证

**适用人群**: 分布式推理开发者、性能优化工程师

---

### 6. [PD分离测试场景分析](PD_Separation_Test_Scenarios.md)
**文件大小**: 103KB+ | **创建时间**: 2026-06-20

**核心内容**:
- 多场景测试配置（1P1D/多P多D/KV Pool等）
- 硬件要求与网络连通性配置
- 各场景数据流图
- 参数化测试表

**适用人群**: 测试工程师、PD部署运维者

---

### 7. [PD分离测试成功报告](PD_TEST_SUCCESS.md)
**文件大小**: 15KB+ | **创建时间**: 2026-06-20

**核心内容**:
- Docker multiprocess模式测试环境
- Qwen3-0.6B 模型测试配置
- MooncakeConnectorV1 参数
- 测试成功记录

**适用人群**: 测试参考、环境搭建参考

---

### 8. [PD分离优化分析报告](PD分离优化分析报告.md)
**文件大小**: ~50KB | **创建时间**: 2026-06-27

**核心内容**:
- PD分离优化方案分析
- 性能瓶颈识别
- 优化建议

**适用人群**: 性能优化工程师

---

### 9. [PD分离源码深度分析](PD分离源码深度分析.md)
**文件大小**: ~50KB | **创建时间**: 2026-06-27

**核心内容**:
- PD分离源码深度剖析
- 关键代码路径追踪

**适用人群**: 核心开发者

---

### 10. [GPU Model Runner Load Model 详细流程](gpu_model_runner_load_model_detailed.md)
**文件大小**: 17KB | **创建时间**: 2026-06-14

**核心内容**:
- GPU Model Runner 初始化流程
- 模型加载完整流程
- KV Cache 初始化机制
- 权重加载和优化

**适用人群**: 模型加载相关开发者

---

### 11. [KVTransfer Workflow](kvtransfer_workflow.md)
**文件大小**: 20KB | **创建时间**: 2026-06-13

**核心内容**:
- KV Cache 传输工作流
- 分布式 KV Cache 管理机制
- KV Transfer 实现细节

**适用人群**: 分布式系统开发者

---

### 12. [vLLM CPU Model Loading Flow](vllm_cpu_model_loading_flow.md)
**文件大小**: 13KB | **创建时间**: 2026-06-13

**核心内容**:
- CPU 设备上的模型加载流程
- CPU 特定的初始化和优化
- CPU 与 GPU 加载流程的差异

**适用人群**: CPU 后端开发者

---

### 13. [vLLM 源码安装与启动指南](vllm_source_install_guide.md)
**文件大小**: 4KB | **创建时间**: 2026-06-14

**核心内容**:
- 源码安装步骤
- 环境配置
- 服务启动方法

**适用人群**: 新入职开发者

---

## 📊 统计信息

- **文档总数**: 13 个
- **覆盖组件**: 算子、模型适配、PD分离（架构/分析/源码）、Model Runner、KV Transfer、CPU Loading、安装指南

---

**返回**: [主文档索引](../README.md)