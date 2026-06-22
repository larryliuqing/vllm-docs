# Mooncake 项目文档

本目录包含 Mooncake（KVCache 解耦架构）的详细分析和集成文档。

---

## 📚 文档列表

### 1. [Mooncake 功能架构与业务流程深度解析](mooncake_architecture_and_workflow.md)
**文件大小**: 180KB+

**核心内容**:
- Mooncake 项目定位（KVCache 解耦架构）
- Transfer Engine 传输引擎详解
- Mooncake Store 分布式 KVCache
- P2P Store 节点间对象共享
- PD 解耦集成
- 多框架集成（vLLM, SGLang, LMCache, xLLM, LMDeploy）

**适用人群**: 分布式系统开发者、KVCache 优化工程师

---

### 2. [Mooncake 与 HIXL 集成详解](mooncake_hixl_integration.md)
**文件大小**: ~14KB | **创建时间**: 2026-06-20

**核心内容**:
- HIXL 概述（单边零拷贝、多链路）
- Mooncake 与 HIXL 集成架构
- 4 个零拷贝接口详解
- 4 种传输类型（D2D, D2H, H2D, H2H）
- 完整使用示例
- 性能对比（HCCS 119GB/s vs RDMA 22GB/s）

**适用人群**: 昇腾 NPU 用户、PD 分离推理开发者

---

### 3. [Mooncake 文档审查报告](mooncake_documentation_review.md)
**文件大小**: ~9KB | **创建时间**: 2026-06-20

**核心内容**:
- 文档现状分析
- 文档质量评价
- 8 大改进方向
- 与官方文档对比

**适用人群**: 文档维护者、项目管理者

---

### 4. [Mooncake 文档索引](mooncake_docs_index.md)
**文件大小**: ~14KB | **创建时间**: 2026-06-20

**核心内容**:
- 完整文档路径索引
- 官方文档链接

**适用人群**: 所有开发者

---

## 📊 统计信息

- **文档总数**: 4 个
- **总行数**: ~4,000 行
- **覆盖组件**: Transfer Engine, Mooncake Store, P2P Store, HIXL 集成

---

**返回**: [主文档索引](../README.md)
