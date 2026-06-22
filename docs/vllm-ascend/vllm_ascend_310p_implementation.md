# vLLM-Ascend 310P 专用实现详解

> 本文档深度解析 vLLM-Ascend 针对 Ascend 310P NPU 的专用实现，阐述其架构设计、核心组件、优化策略和使用限制。

---

## 一、310P 概述

### 1.1 硬件特性

**Ascend 310P** 是华为昇腾系列的推理专用 NPU，具有以下特点：

| 特性 | 说明 |
|------|------|
| **定位** | 推理专用 NPU |
| **算力** | 8 TOPS (INT8) |
| **功耗** | 8W |
| **应用场景** | 边缘推理、轻量级部署 |
| **优化重点** | 低功耗、高能效比 |

### 1.2 源码规模

| 模块 | 文件数 | 总行数 | 主要文件 |
|------|--------|--------|---------|
| **_310p/** | 10 | ~50KB | `model_runner_310p.py` (42KB), `worker_310p.py` (8KB) |

---

## 二、核心组件

### 2.1 目录结构

```
vllm-ascend/vllm_ascend/_310p/
├── __init__.py
├── model_runner_310p.py          # 310P 专用 ModelRunner (42KB)
├── worker_310p.py                # 310P 专用 Worker (8KB)
├── block_table.py                # Block Table 实现 (9KB)
├── npu_input_batch.py            # NPU Input Batch (2KB)
├── attention/                    # 310P 专用 Attention
├── quantization/                 # 310P 专用量化
├── ops/                          # 310P 专用算子
└── sample/                       # 310P 专用采样
```

### 2.2 核心实现

#### **ModelRunner310P**

**文件**: `model_runner_310p.py` (42KB)

**核心职责**：
- 310P 专用的模型前向传播
- 优化的 KV Cache 管理
- 特定的算子调用

#### **Worker310P**

**文件**: `worker_310p.py` (8KB)

**核心职责**：
- 310P 专用的 Worker 初始化
- 设备配置和优化

---

## 三、使用限制

### 3.1 功能限制

| 功能 | 状态 | 说明 |
|------|------|------|
| **标准 Attention** | ✅ 支持 | 使用 CANN 原生算子 |
| **MLA** | ⚠️ 部分支持 | 部分 MLA 算子受限 |
| **DSA** | ❌ 不支持 | DSA 算子不兼容 |
| **量化** | ✅ 支持 | 支持 INT8 量化 |
| **CUDA Graph** | ❌ 不适用 | 使用 ACL Graph |

---

**文档版本**: v1.0  
**创建时间**: 2026-06-20  
**基于源码**: vllm-ascend/vllm_ascend/_310p/  
**维护者**: vLLM-Ascend 项目团队