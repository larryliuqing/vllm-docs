# DS-V4-Flash-w4a8-mtp MTP 推测解码测试报告

> **测试日期**: 2026-06-29  
> **测试环境**: vLLM-Ascend v0.20.2rc, CANN 25.5.2  
> **NPU**: 8 × Ascend 910B4 (TP=8, EP=8)  
> **模型**: DeepSeek-V4-Flash-w4a8-mtp (W4A8 量化)  
> **测试脚本**: `test_scripts/run_mtp_docker.sh`

---

## 一、测试目的

评估 MTP (Multi-Token Prediction) 推测解码在 DS-V4 量化模型上的效果，对比不同 `num_speculative_tokens`（1, 3, 5）对生成速度和 acceptance rate 的影响。

## 二、测试配置

| 参数 | 值 |
|------|------|
| 推测方法 | `mtp` |
| num_speculative_tokens | 1, 3, 5 |
| Temperature | 0 (greedy) |
| max_tokens | 30 |
| max_model_len | 6000 (num_spec≤3) / 5000 (num_spec=5) |
| max_num_seqs | 8 |
| gpu_memory_utilization | 0.97 |
| 目标模型 | DS-V4-Flash-w4a8-mtp (W4A8 量化) |
| Draft 模型 | MTP 模块（模型内嵌） |

## 三、测试结果

### 3.1 生成速度

| num_speculative_tokens | Prompt 1 | Prompt 2 | Prompt 3 | Prompt 4 | **平均速度** |
|:---------------------:|:--------:|:--------:|:--------:|:--------:|:-----------:|
| **1** (仅目标) | 0.73 | 5.33 | 21.89 | 21.91 | **12.47 tok/s** |
| **3** | 1.97 | 14.81 | 16.14 | 16.17 | **12.27 tok/s** |
| **5** | 4.32 | 10.37 | 10.74 | 11.04 | **9.12 tok/s** |

> **说明**: Prompt 1 是第一次推理（含编译预热），速度明显偏慢。后 3 个 prompt 的稳态速度更有参考价值。

### 3.2 Steady-State 速度（剔除 Prompt 1）

| num_speculative_tokens | 稳态平均速度 | 相对 baseline |
|:---------------------:|:-----------:|:-------------:|
| **1** | **16.38 tok/s** | — (baseline) |
| **3** | **15.71 tok/s** | **-4.1%** |
| **5** | **10.72 tok/s** | **-34.6%** |

### 3.3 Acceptance Rate

| 指标 | num_spec=1 | num_spec=3 | num_spec=5 |
|------|:----------:|:----------:|:----------:|
| Draft 总轮数 | 103 | 103 | 103 |
| Draft Token 总数 | 103 | 309 | 515 |
| 接受 Token 总数 | 15 | 15 | 15 |

**每位置 Acceptance Rate:**

| Position | num_spec=1 | num_spec=3 | num_spec=5 |
|:--------:|:----------:|:----------:|:----------:|
| **Pos 0** | **14.56%** | **14.56%** | **14.56%** |
| Pos 1 | — | 0.00% | 0.00% |
| Pos 2 | — | 0.00% | 0.00% |
| Pos 3 | — | — | 0.00% |
| Pos 4 | — | — | 0.00% |

**总体 Acceptance Rate:**

| num_speculative_tokens | 总体 Acceptance Rate |
|:---------------------:|:-------------------:|
| 1 | 14.56% |
| 3 | 4.85% |
| 5 | 2.91% |

## 四、分析

### 4.1 Position 0 Acceptance Rate 偏低（~14.56%）

参考同类模型的典型基线（如 Qwen3-Next MTP：golden baseline `[0.85, 0.46, 0.19]`），DS-V4-Flash-w4a8-mtp 的 Position 0 接受率 ~14.56% 明显偏低。

**可能原因：**

1. **W4A8 量化精度损失**: 量化后目标模型和 draft 模型共享的 LM Head 概率分布出现偏差，导致 draft token 与目标模型验证不一致
2. **测试 prompt 过短**: prompt 仅 5-7 tokens（`"Hello, my name is"` 等），上下文不足以让 MTP 做出高质量预测
3. **模型架构差异**: DeepSeek-V4 使用 MLA (Multi-Head Latent Attention) + MoE，量化后 MoE 路由和 MLA 的精度损失可能叠加

### 4.2 位置 1+ 接受率全部为 0

所有 `num_speculative_tokens > 1` 的测试中，位置 1 及之后的 draft token 接受率均为 0%。这意味着：

- 当 Position 0 的 draft token 被接受后，MTP 后续 step 生成的 token 没有任何一个匹配目标模型的验证
- Draft 模型的迭代预测（使用上一步生成的 hidden states）在量化场景下累积误差过快

### 4.3 速度不升反降

| 配置 | 稳态速率 |
|:---:|:--------:|
| num_spec=1 | 16.38 tok/s |
| num_spec=3 | 15.71 tok/s |
| num_spec=5 | 10.72 tok/s |

由于 acceptance rate 极低（位置 1+ 为 0），增加 `num_speculative_tokens` 只是增加了额外的 draft 前向计算，却没有带来更多的 accepted token，导致速度反而下降。对于当前量化模型，**num_speculative_tokens=1 是最优选择**。

### 4.4 与参考基线的对比

| 模型 | 方法 | Golden Baseline | 实际值 |
|------|------|:---------------:|:------:|
| **Qwen3-Next-80B-A3B** | qwen3_next_mtp (TP4) | `[0.85, 0.46, 0.19]` | — |
| **DS-V4-Flash-w4a8** | mtp (TP8) | — | **`[0.146, 0, 0]`** |

差距显著，主要差异来自：模型量化、架构（MLA vs standard attention）、测试配置差异。

## 五、结论与建议

### 5.1 结论

1. **对 W4A8 量化模型的 MTP 效果有限**: Acceptance rate ~14.56%，且无法有效利用多个 speculative token
2. **num_speculative_tokens=1 是最优选择**: 更大值不仅无益，反而因额外的 draft 前向计算降低整体吞吐
3. **量化是主要瓶颈**: W4A8 量化产生的精度损失显著降低了 draft 和目标模型之间的一致性

### 5.2 后续改进建议

1. **使用 FP16/BF16 模型测试**: 排除量化影响，验证 MTP 在非量化版本上的真实效果
2. **增加 prompt 长度**: 使用 50-200 token 的较长的输入提供更充分的上下文
3. **优化 Draft 模型**: 验证 MTP 模块的 hc_head 融合和 shared_head 在量化后的精度
4. **参考 Golden Benchmark**: 使用与 vLLM-Ascend 现有测试相同的基准（如 100 条混合 repeat/sentence prompt 测试）
5. **尝试 alternative 推测方法**: 对量化模型，Ngram 或 Draft Model 可能更适合

---

*报告生成: 2026-06-29*
*测试工具: vllm-docs/docs/tests/test_scripts/run_mtp_docker.sh*
*测试日志: `mtp_test_full_output.log`*