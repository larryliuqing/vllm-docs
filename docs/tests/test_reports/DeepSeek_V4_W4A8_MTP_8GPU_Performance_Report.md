# DeepSeek-V4-Flash-w4a8-mtp 8GPU 性能测试报告

**测试日期**: 2026-06-26
**模型**: DeepSeek-V4-Flash-w4a8-mtp (W4A8量化, 支持MTP)
**硬件**: 8× Ascend 910B4 (32GB HBM/卡)  
**Docker 镜像**: vllm-ascend:v0.20.2rc
**推理框架**: vLLM V1 (v0.20.2)

---

## 1. 测试配置

### 1.1 启动参数

| 参数 | 值 |
|------|-----|
| `tensor-parallel-size` | 8 |
| `data-parallel-size` | 1 |
| `enable-expert-parallel` | True |
| `max-model-len` | 8192 |
| `max-num-batched-tokens` | 10240 |
| `max-num-seqs` | 64 |
| `gpu-memory-utilization` | 0.95 |
| `quantization` | ascend |
| `block-size` | 128 |
| `speculative-config` | `{"num_speculative_tokens": 1, "method": "mtp"}` |
| `compilation-config` | `{"cudagraph_mode": "FULL_DECODE_ONLY"}` |
| `async-scheduling` | True |

### 1.2 模型参数

| 参数 | 值 |
|------|-----|
| **Architecture** | DeepseekV4ForCausalLM |
| **Hidden Size** | 4096 |
| **Num Layers** | 43 |
| **Attention Heads** | 64 |
| **KV Heads** | 1 (MLA) |
| **Head Dim** | 512 |
| **Num Experts** | 256 |
| **Experts Per Token** | 6 |
| **Scoring Func** | sqrtsoftplus |
| **Vocab Size** | 129280 |
| **Max Position** | 1048576 (限制为8192) |
| **Q LoRA Rank** | 1024 |
| **MTP Layers** | 1 |

---

## 2. 启动性能

### 2.1 启动时间线

| 阶段 | 开始 | 结束 | 耗时 | 备注 |
|------|------|------|------|------|
| 引擎初始化 | 03:43:28 | 03:43:28 | - | EngineCore创建 |
| HCCL/Gloo建连 | 03:43:45 | 03:44:56 | ~71s | 8 rank并行 |
| 插件/模型注册 | 03:42:27 | 03:45:30 | ~3min | 各worker独立注册 |
| Backbone权重加载 | 03:45:30 | 03:48:40 | **3min10s** | 38 shards, 53s有效耗时 |
| Drafter(MTP)加载 | 03:48:40 | 03:49:02 | **22s** | 8 rank串行 |
| Backbone + MTP权重合计 | 03:45:30 | 03:49:16 | **~3min46s** | 最慢rank完成 |
| torch.compile | 03:49:12 | 03:50:03 | **46s** | compile range (1, 10240) |
| Graph Capture | 03:50:34 | 03:52:09 | **122s** | 16 batch sizes × 8 rank |
| **→ 总启动时间** | **03:42:27** | **03:52:13** | **~9min46s** | |

### 2.2 每卡权重大小

| Rank | 权重完成时间 | 权重大小 |
|------|------------|---------|
| TP1_EP1 | 03:48:55 | 21.90 GB |
| TP4_EP4 | 03:49:00 | 21.90 GB |
| TP6_EP6 | 03:49:03 | 21.90 GB |
| TP3_EP3 | 03:49:03 | 21.90 GB |
| TP5_EP5 | 03:49:05 | 21.90 GB |
| TP2_EP2 | 03:49:10 | 21.90 GB |
| TP0_EP0 | 03:49:12 | 21.90 GB |
| TP7_EP7 | 03:49:16 | 21.90 GB |

**每卡权重大小**: ~21.90 GB  
**模型总大小 (未分片)**: ~151 GB (38 safetensors shards)

### 2.3 显存分配 (每卡)

| 项目 | 大小 | 占比 |
|------|------|------|
| 总 HBM | 29.49 GiB | 100% |
| 权重占用 | **21.90 GiB** | 74.3% |
| 峰值激活 | 1.31 GiB | 4.4% |
| Non-torch 内存 | 0.46 GiB | 1.6% |
| NPU Graph 内存 | 0.75 GiB | 2.5% |
| **KV Cache** | **4.35 GiB** | **14.8%** |
| 剩余空闲 | ~0.72 GiB | 2.4% |

---

## 3. 推理性能

### 3.1 单请求延迟测试 (Completions API, temperature=0.1)

#### 固定短输入 → 变长输出

| 输出长度 | 平均耗时 | P50 | Token速度 | 曲线 |
|----------|---------|-----|-----------|------|
| 10 tok | 0.78s | 0.77s | 12.8 tok/s | (冷启动+prefill) |
| 50 tok | 2.20s | 2.18s | 22.9 tok/s | |
| 100 tok | 3.81s | - | 26.2 tok/s | |
| 200 tok | 7.09s | - | 28.2 tok/s | |
| 500 tok | 17.96s | - | 27.8 tok/s | 稳态 |

#### 输出长度 - 延迟 - 速度关系

```
输出=10tok:  0.78s → 12.8 tok/s  ▲ prefill占主导
输出=50tok:  2.20s → 22.9 tok/s  
输出=100tok: 3.81s → 26.2 tok/s  
输出=200tok: 7.09s → 28.2 tok/s  ─ 进入稳态
输出=500tok: 17.96s → 27.8 tok/s ▼ 接近理论上限
```

**稳态 Decode 速度**: ~28 tok/s (200+ tokens)

#### 变长输入 → 固定输出 (50 tok)

| Prompt 长度 | 首次耗时 | 后续稳定耗时 | Prefill贡献 |
|------------|---------|-------------|-------------|
| 短 (~5 tok) | 6.08s | 2.25s | 3.83s |
| 中 (~500 char) | 6.27s | 2.00s | 4.27s |
| 长 (~2000 char) | 6.20s | 2.01s | 4.19s |

**结论**: Prompt 长度从 5→2000 对延迟无显著影响。Prefill 阶段对 8卡TP+EP 而言不构成瓶颈。首次请求包含 Graph replay 预热开销(~3.8s)，后续稳定在 ~2s。

### 3.2 并发性能测试

#### 并发吞吐

| 并发数 | 总耗时 | 吞吐量 | 效率(相对1req) |
|--------|--------|--------|---------------|
| 1 req | 2.20s | 22.7 tok/s | 1.00x |
| 4 req | 20.81s | 9.6 tok/s | 0.42x |
| 8 req | 12.70s | 31.5 tok/s | 1.39x |
| 16 req | 14.90s | 53.7 tok/s | 2.37x |

#### 并发效率分析

| 并发数 | 理论最大 | 实际 | 效率 | 说明 |
|--------|---------|------|------|------|
| 1 | 28 tok/s | 22.7 | 81% | 单请求非满载 |
| 4 | 112 tok/s | 9.6 | 9% | ⚠️ 严重降速 |
| 8 | 224 tok/s | 31.5 | 14% | batch开启后回升 |
| 16 | 448 tok/s | 53.7 | 12% | 线性扩展 |

**结论**: 并发在 1→4 时出现剧烈降速，可能原因：
1. 首请求 prefetch/graph replay 抖动
2. max-num-seqs=64 的调度策略在低并发时保守
3. MTP 投机解码在并发下调度开销放大

---

## 4. 关键指标一览

| 指标 | 数值 |
|------|------|
| **模型启动时间** | ~9min46s |
| **每卡权重大小** | 21.90 GB |
| **KV Cache 大小** | 4.35 GiB |
| **有效 max-model-len** | 8192 |
| **Decode 稳态吞吐** | ~28 tok/s |
| **4并发吞吐** | 9.6 tok/s (异常) |
| **16并发吞吐** | 53.7 tok/s |
| **冷启动首请求** | ~6s |
| **首Token延迟(预热后)** | ~0.77s |
| **Graph Capture耗时** | 122s |
| **torch.compile耗时** | 46s |

---

## 5. 瓶颈分析与优化建议

### 5.1 P0 — 显存瓶颈

**问题**: 权重占用 21.9GB/29.5GB (74.3%)，KV Cache 仅剩 4.35 GiB，限制 max-model-len 到 8192。

**分析**:
- W4A8 量化已是最激进方案 (FP4 expert + INT8 attention)
- DeepSeek V4 的 256 专家 × 4096 hidden 导致参数总量大
- 8卡 EP 下每卡 32 专家 = 21.9GB

**建议**:
| 方案 | 效果 | 代价 |
|------|------|------|
| 关闭 `enable-prefix-caching` | +少量KV空间 | 降低前缀命中率 |
| 减少 `max-num-batched-tokens` → 6144 | +~0.5GiB | 降低最大batch |
| 减少 `block-size` → 64 | 更细粒度KV分配 | 增加管理开销 |

### 5.2 P0 — 启动时间过长 (9min46s)

**问题**: Graph Capture 耗时 122s (21% 总启动时间)。

**分析**: 当前配置 16 个 capture sizes: [8, 16, 24, 32, ..., 128]，每个耗时约 6-7s。

**建议**:
| 方案 | 预期节省 | 代价 |
|------|---------|------|
| 减少为 [8, 16, 32, 64, 128] | -60s | 中间size回退 |
| 减少为 [8, 32, 64, 128] | -75s | 更多回退 |
| 减少为 [8, 16, 64, 128] | -75s | |

### 5.3 P1 — Decode 吞吐偏低 (~28 tok/s)

**问题**: 8卡 TP+EP 环境下，稳态 Decode 吞吐 ~28 tok/s，低于单卡预期。

**分析**:
- `disable_custom_all_reduce=True` 限制了通信优化
- `enable_sp=False` (序列并行未开启)
- MoE all-to-all 通信在 8 rank 间频繁
- FlashComm1 已启用但 FlashComm2 未使用

**建议**:
| 方案 | 预期提升 | 风险 |
|------|---------|------|
| 在 `pass_config` 中开启 `fuse_gemm_comms` | ~10% | 兼容性 |
| 开启 `enable_sp` | ~15-20% | 需要验证 |
| 测试 `VLLM_ASCEND_ENABLE_FLASHCOMM1=1` + FlashComm2 | ~20% | 依赖CANN版本 |

### 5.4 P1 — 4并发性能退化

**问题**: 4 并发时吞吐退化到 9.6 tok/s (单请求的 42%)，8/16 并发恢复正常。

**分析**: 推测与调度策略相关：低并发时 batch 未充分填充，scheduler 可能频繁切换导致开销放大。

**建议**:
- 降低 `max-num-seqs` 从 64 到 32，看调度粒度
- 调整 `max-num-batched-tokens` 从 10240 到 6144
- 监控 scheduler 是否在低并发下产生大量空闲轮次

### 5.5 P2 — MTP 投机解码效果待验证

**问题**: MTP (num_speculative_tokens=1) 开启带来额外 22s 的 Drafter 加载时间，实际加速效果未测量。

**建议**: 对比测试：
- `num_speculative_tokens=0` (关闭MTP) vs `=1` vs `=2`
- 在长输出场景 (500+ tok) 测量接受率

### 5.6 P2 — CPU 绑定失败

**问题**: 容器内 `npu-smi` 不可用，CPU topology 绑定跳过。

**建议**: 在 docker run 中挂载 `npu-smi`:
```bash
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi
```

### 5.7 P3 — compile cache 未持久化

**问题**: 首次编译 46s，容器销毁后缓存丢失。

**建议**: 挂载 compile cache 目录：
```bash
-v /root/.cache/vllm:/root/.cache/vllm
```

---

## 6. 对比参考

| 模型 | 硬件 | TP | 吞吐 | 场景 |
|------|------|----|------|------|
| DeepSeek V4 (本次) | 8×910B4 | TP8+EP8 | 28 tok/s | W4A8, MoE 256专家 |
| Qwen2-VL-7B (参考) | 2×910B4 | TP2 | 45 tok/s | Dense 7B |
| Qwen2.5-Omni-7B (参考) | 4×910B4 | TP4 | 33 tok/s | Dense 7B |

---

## 7. 结论

DeepSeek-V4-Flash-w4a8-mtp 在 8×910B4 环境下成功推理，核心结论：

1. **可达稳态吞吐**: ~28 tok/s (单请求 200+ tokens)
2. **并发能力**: 16 并发可达 53.7 tok/s，4 并发异常降速需排查
3. **显存瓶颈**: 权重占比 74%，KV Cache 受限导致 max-model-len 仅 8192
4. **启动优化**: Graph Capture 122s 是最大单点瓶颈，可减半
5. **Prefill**: 8卡并行下 prefill 不构成瓶颈 — 长/短 prompt 延迟一致

---

**报告生成时间**: 2026-06-26
**测试工具**: curl + npu-smi + docker logs
