# vLLM 测试文档

本目录包含 vLLM 及 vllm-ascend 的测试报告、测试指南和相关文档。

---

## 📁 子目录结构

```
tests/
├── README.md           # 本文件
├── multimodal/         # 多模态模型测试
├── omni/               # Omni 模型测试
├── ascend/             # Ascend NPU 测试
├── moe/                # MOE 参数测试 ⭐
├── test_reports/       # 测试报告（PD分离, DS V4等）⭐
└── test_scripts/       # 测试脚本（PD分离部署, DS V4启动等）⭐
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

### 四、MOE 参数测试 ([moe/](moe/))

| 文档 | 说明 |
|------|------|
| [MOE参数测试指南.md](moe/MOE参数测试指南.md) | MOE 参数测试操作指南 |
| [MOE参数测试计划.md](moe/MOE参数测试计划.md) | MOE 参数测试计划 |
| [4卡部署测试指南.md](moe/4卡部署测试指南.md) | 4卡部署操作指南 |
| [4卡性能测试报告.md](moe/4卡性能测试报告.md) | 4卡性能测试报告 |
| [MOE测试成功报告.md](moe/MOE测试成功报告.md) | MOE 测试成功报告 |
| [MOE测试最终报告.md](moe/MOE测试最终报告.md) | MOE 完整测试总结 |
| [MOE测试进展报告.md](moe/MOE测试进展报告.md) | MOE 测试进展记录 |
| [MOE启动状态.md](moe/MOE启动状态.md) | MOE 启动状态 |
| [Qwen3-Omni-30B测试指南.md](moe/Qwen3-Omni-30B测试指南.md) | Omni 30B 测试操作指南 |
| [测试执行总结.md](moe/测试执行总结.md) | 测试执行总结 |
| [多模态测试完成报告.md](moe/多模态测试完成报告.md) | 多模态测试完成报告 |
| [完整测试总结报告.md](moe/完整测试总结报告.md) | 完整测试总结 |
| [glm-5.13-Omni-30B能力测试总结.md](moe/glm-5.13-Omni-30B能力测试总结.md) | GLM Omni-30B 能力测试 |

---

### 五、测试报告 ([test_reports/](test_reports/))

| 文档 | 说明 |
|------|------|
| [PD分离测试报告（2026-06-24）](test_reports/PD_Separation_Test_Report_20260624.md) | 基于 Qwen3-VL-32B 的 PD 分离 4 卡测试 |
| [PD分离 4 卡测试报告（2026-06-24）](test_reports/PD_Separation_4cards_Test_Report_20260624.md) | Qwen3-VL-32B 4 卡 PD 分离测试详细报告 |
| [PD分离性能报告（2026-06-25）](test_reports/PD_Separation_Performance_Report_20260625.md) | 2+2 卡 PD 分离性能基准测试 |
| [PD分离异构TP报告模板](test_reports/PD_Separation_Heter_TP_Performance_Report_TEMPLATE.md) | 异构 TP 配置性能报告模板 |
| [DeepSeek V4 8GPU性能报告](test_reports/DS_V4_W4A8_MTP_8GPU_Performance_Report.md) | 8 卡 DS V4 推理性能测试 |
| [benchmark_results/](test_reports/benchmark_results/) | 异构 TP 基准测试原始数据 |

---

### 六、测试脚本 ([test_scripts/](test_scripts/))

| 脚本 | 说明 |
|------|------|
| `deploy_pd_separation*.sh` | PD 分离部署脚本（4 卡、异构 TP） |
| `run_prefill_*.sh` / `run_decode_*.sh` | Prefill/Decode 节点启动脚本 |
| `bench_deepseek_v4.sh` | DS V4 快速性能测试 |
| `stress_test_dsv4*.sh` | DeepSeek V4 压力测试 |
| `benchmark_heter_tp.sh` | 异构 TP 基准测试 |
| `warmup_request.sh` | 预热请求 |
| `analyze_logs.py` | 日志分析 |
| `test_vllm_cpu.py` | CPU 快速测试 |
| `test_pd_container_local.py` | PD 分离容器本地测试 |

---

## 📊 统计信息

- **文档总数**: 38+ 个（含测试脚本）
- **测试模型**: Qwen2-VL-7B, Qwen3-VL-32B, DeepSeek V4, Qwen3-Omni-30B
- **测试平台**: NVIDIA GPU, Ascend NPU (910B4)

---

**返回**: [主文档索引](../README.md)
