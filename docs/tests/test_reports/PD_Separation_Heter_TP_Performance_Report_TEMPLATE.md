# PD分离异构TP性能测试报告模板

**测试日期**: {{DATE}}
**测试环境**: {{HOSTNAME}}
**测试配置**: Prefill (NPU 2-3, TP=2) + Decode (NPU 4-7, TP=4)
**测试模型**: Qwen2-VL-7B-Instruct
**KV Connector**: MooncakeLayerwiseConnector (pd_head_ratio=2)

---

## 1. 测试配置

### 1.1 硬件配置

| 项目 | 配置 |
|------|------|
| **服务器类型** | Atlas 训练服务器 |
| **NPU型号** | 910B4 (32GB HBM/card) |
| **PD分离配置** | 2+4卡 (Prefill: NPU 2-3, Decode: NPU 4-7) |
| **Tensor Parallel** | TP=2 (Prefill), TP=4 (Decode) |

### 1.2 软件配置

| 组件 | 版本/配置 |
|------|-----------|
| **Driver** | {{DRIVER_VERSION}} |
| **CANN** | 9.0.0 |
| **vLLM** | 0.20.2 |
| **vLLM-Ascend** | v0.20.2rc |
| **KV Connector** | MooncakeLayerwiseConnector |
| **pd_head_ratio** | 2 |
| **KV Buffer Device** | npu |
| **Max Model Len** | 4096 |
| **GPU Memory Utilization** | 0.85 |
| **Max Batched Tokens** | 4096 |

### 1.3 异构TP配置详解

| 节点 | NPU | TP | pd_head_ratio | KV Connector | 端口 |
|------|-----|----|---------------|--------------|------|
| Prefill | 2,3 | 2 | 2 | MooncakeLayerwiseConnector | 8100 |
| Decode | 4,5,6,7 | 4 | 2 | MooncakeLayerwiseConnector | 8200 |
| Proxy | - | - | - | - | 8000 |

### 1.4 测试场景矩阵

| 场景 | Prompt (tok) | Output (tok) | 测试目的 |
|------|-------------|-------------|----------|
| 短提示-短输出 | ~20 | 20 | 基线，与2+2报告对比 |
| 短提示-中输出 | ~20 | 128 | Decode吞吐基线 |
| 短提示-长输出 | ~20 | 256 | Decode长序列吞吐 |
| 中提示-中输出 | ~512 | 128 | 中Prompt时Prefill性能 |
| 中提示-长输出 | ~1024 | 256 | 中Prompt + 长Output |
| 长提示-中输出 | ~2048 | 128 | 长Prompt时Prefill性能 |
| 长提示-长输出 | ~2048 | 256 | 长Prompt + 长Output |
| 超长提示-中输出 | ~4096 | 128 | Prefill上限压力 |
| 超长提示-长输出 | ~4096 | 256 | 综合压力测试 |

### 1.5 并发测试场景

| 场景 | 并发数 | Output (tok/req) | 测试目的 |
|------|--------|-----------------|----------|
| 单请求 | 1 | 50 | 基线 |
| 轻并发 | 4 | 50 | 轻负载 |
| 中并发 | 8 | 50 | 中等负载 |
| 重并发 | 16 | 50 | 压力测试 |

---

## 2. 启动性能

| 指标 | Prefill节点 | Decode节点 |
|------|-------------|------------|
| **模型加载时间** | {{PREFILL_LOAD_TIME}} | {{DECODE_LOAD_TIME}} |
| **每卡权重大小** | 7.76 GiB | 7.76 GiB |
| **KV Cache容量** | {{PREFILL_KV_CACHE_CAPACITY}} | {{DECODE_KV_CACHE_CAPACITY}} |
| **编译时间** | {{PREFILL_COMPILE_TIME}} | {{DECODE_COMPILE_TIME}} |
| **启动时间** | {{PREFILL_STARTUP_TIME}} | {{DECODE_STARTUP_TIME}} |

---

## 3. 单请求延迟测试

### 3.1 短提示 (~20 tokens) → 变长输出

| 输出长度 | 平均延迟(ms) | P50(ms) | P95(ms) | Decode吞吐(tok/s) |
|----------|-------------|---------|---------|-------------------|
| 20 | {{SHORT_20_LATENCY}} | {{SHORT_20_P50}} | {{SHORT_20_P95}} | {{SHORT_20_THROUGHPUT}} |
| 128 | {{SHORT_128_LATENCY}} | {{SHORT_128_P50}} | {{SHORT_128_P95}} | {{SHORT_128_THROUGHPUT}} |
| 256 | {{SHORT_256_LATENCY}} | {{SHORT_256_P50}} | {{SHORT_256_P95}} | {{SHORT_256_THROUGHPUT}} |

### 3.2 变长提示 → 固定输出

| 提示长度 | 平均延迟(ms) | P50(ms) | Prefill效费比(ms/tok) |
|----------|-------------|---------|----------------------|
| 20 | {{VARPROMPT_20_LATENCY}} | {{VARPROMPT_20_P50}} | {{VARPROMPT_20_COST}} |
| 512 | {{VARPROMPT_512_LATENCY}} | {{VARPROMPT_512_P50}} | {{VARPROMPT_512_COST}} |
| 2048 | {{VARPROMPT_2048_LATENCY}} | {{VARPROMPT_2048_P50}} | {{VARPROMPT_2048_COST}} |
| 4096 | {{VARPROMPT_4096_LATENCY}} | {{VARPROMPT_4096_P50}} | {{VARPROMPT_4096_COST}} |

### 3.3 完整场景矩阵

| 场景 | Prompt | Output | 平均延迟(ms) | P50(ms) | P95(ms) | 吞吐(tok/s) |
|------|--------|--------|-------------|---------|---------|-------------|
| ... | ... | ... | ... | ... | ... | ... |

---

## 4. 并发性能测试

**提示长度**: ~20 tokens, **输出长度**: 50 tokens each

| 并发数 | 总耗时(ms) | 总吞吐(tok/s) | 单请求等效吞吐 | 成功率 |
|--------|-----------|---------------|---------------|--------|
| 1 | {{CONC1_TIME}} | {{CONC1_TPUT}} | {{CONC1_PER_REQ}} | {{CONC1_SUCCESS}} |
| 4 | {{CONC4_TIME}} | {{CONC4_TPUT}} | {{CONC4_PER_REQ}} | {{CONC4_SUCCESS}} |
| 8 | {{CONC8_TIME}} | {{CONC8_TPUT}} | {{CONC8_PER_REQ}} | {{CONC8_SUCCESS}} |
| 16 | {{CONC16_TIME}} | {{CONC16_TPUT}} | {{CONC16_PER_REQ}} | {{CONC16_SUCCESS}} |

**并发扩展效率**: {{CONCURRENCY_EFFICIENCY}}

---

## 5. KV传输性能分析

### 5.1 总体统计

| 指标 | 数值 |
|------|------|
| **延迟样本数** | {{KV_SAMPLE_COUNT}} |
| **平均延迟** | {{KV_AVG_LATENCY}} |
| **最小延迟** | {{KV_MIN_LATENCY}} |
| **最大延迟** | {{KV_MAX_LATENCY}} |
| **P50延迟** | {{KV_P50_LATENCY}} |
| **传输方式** | Ascend Direct Transport (MooncakeLayerwise) |

### 5.2 不同Prompt长度下的KV传输

| Prompt长度 | 平均KV传输延迟 | 占总体延迟比例 |
|------------|---------------|---------------|
| 20 | {{KV_20_LATENCY}} | {{KV_20_RATIO}} |
| 512 | {{KV_512_LATENCY}} | {{KV_512_RATIO}} |
| 2048 | {{KV_2048_LATENCY}} | {{KV_2048_RATIO}} |
| 4096 | {{KV_4096_LATENCY}} | {{KV_4096_RATIO}} |

---

## 6. 与2+2卡配置对比

| 指标 | 2+2 (TP=2, 历史) | 异构TP (2+4, 本次) | 变化 |
|------|-------------------|-------------------|------|
| **稳态延迟(20tok)** | 500ms | {{CUR_SHORT_20_LATENCY}} | {{CHANGE_SHORT_20}} |
| **Decode吞吐(单请求)** | 45 tok/s | {{CUR_DECODE_TPUT}} | {{CHANGE_DECODE_TPUT}} |
| **并发吞吐(10/16req)** | 73.8 tok/s | {{CUR_CONC_TPUT}} | {{CHANGE_CONC_TPUT}} |
| **KV传输延迟** | 1.5ms | {{CUR_KV_LATENCY}} | {{CHANGE_KV}} |
| **冷启动延迟** | 11749ms | {{CUR_COLD_START}} | {{CHANGE_COLD_START}} |
| **TP效率(每卡tokens)** | ~22.5 tok/s/card | {{CUR_TP_EFFICIENCY}} | {{CHANGE_TP_EFF}} |

---

## 7. 性能瓶颈分析

### 7.1 瓶颈识别

| 瓶颈 | 分析 | 严重程度 |
|------|------|----------|
| **Prefill计算** | 对比不同prompt长度的延迟增长曲线 | {{PREFILL_BOTTLENECK}} |
| **KV传输** | 延迟占总延迟比例 (各类场景) | {{KV_BOTTLENECK}} |
| **Decode计算** | TP=4 vs TP=2 的吞吐对比 | {{DECODE_BOTTLENECK}} |
| **并发串行化** | 并发扩展效率分析 | {{CONC_BOTTLENECK}} |
| **pd_head_ratio开销** | per-card效率 vs 同构TP | {{HEAD_RATIO_BOTTLENECK}} |

### 7.2 延迟分解

```
短提示~20 → 短输出~20 延迟分解 ({{SHORT_LATENCY_TOTAL}}ms):

Prefill阶段: {{SHORT_PREFILL_MS}}ms ({{SHORT_PREFILL_PCT}}%)
  ├── Token处理 + Embedding: {{SHORT_EMBED_MS}}ms
  ├── Multi-head Attention + MLP: {{SHORT_ATTN_MS}}ms
  └── KV Cache计算: {{SHORT_KVCACHE_MS}}ms

KV传输阶段: {{SHORT_KVTRANS_MS}}ms ({{SHORT_KVTRANS_PCT}}%)
  ├── 序列化: {{SHORT_SERIAL_MS}}ms
  ├── P2P传输: {{SHORT_P2P_MS}}ms (Layerwise)
  └── 反序列化: {{SHORT_DESERIAL_MS}}ms

Decode阶段: {{SHORT_DECODE_MS}}ms ({{SHORT_DECODE_PCT}}%)
  └── {{SHORT_DECODE_TOKENS}} tokens × {{SHORT_DECODE_PERTOKEN_MS}}ms/token
```

### 7.3 中长提示延迟分解

```
中提示~512 → 中输出~128 延迟分解 ({{MEDIUM_LATENCY_TOTAL}}ms):

Prefill阶段: ...
...
```

---

## 8. 优化建议

### 8.1 立即执行 (高优先级)

#### 优化1: Decode节点增强

**现状**: {{DECODE_NODE_STATUS}}
**方案**: 
- 增加Decode TP或实例数
- 或优化pd_head_ratio配置

#### 优化2: 请求预热

**现状**: {{COLD_START_STATUS}}
**方案**: 部署后自动预热

### 8.2 短期 (1-2周)

#### 优化3: 缓存感知调度

**现状**: {{CACHE_AWARE_STATUS}}
**方案**: Proxy实现Prefix-hash感知选择Prefiller

#### 优化4: 增大批处理窗口

**现状**: {{BATCH_STATUS}}
**方案**: 调整max-num-batched-tokens

### 8.3 长期 (1月+)

- 动态扩缩容
- KV量化传输
- 多节点PD分离

---

## 9. 测试结论

### 9.1 综合评分

| 维度 | 评分 | 说明 |
|------|------|------|
| **功能完整性** | ⭐⭐⭐⭐⭐ | 异构TP分离功能完备，运行稳定 |
| **KV传输效率** | ⭐⭐⭐⭐⭐ | 稳定 {{KV_AVG_LATENCY}}, 非瓶颈 |
| **单请求延迟** | ⭐⭐⭐ | {{LATENCY_SCORE}} |
| **吞吐量** | ⭐⭐⭐ | {{THROUGHPUT_SCORE}} |
| **并发能力** | ⭐⭐ | {{CONCURRENCY_SCORE}} |
| **TP效率** | ⭐⭐⭐ | {{TP_EFFICIENCY_SCORE}} |

### 9.2 关键数据一览

| 指标 | 数值 |
|------|------|
| 稳态延迟 (20tok) | {{SUMMARY_LATENCY}} |
| Decode吞吐 (单请求) | {{SUMMARY_DECODE_TPUT}} |
| 并发吞吐 | {{SUMMARY_CONC_TPUT}} |
| KV传输延迟 | {{SUMMARY_KV_LATENCY}} |
| 冷启动延迟 | {{SUMMARY_COLD_START}} |
| pd_head_ratio开销 | {{SUMMARY_HEAD_RATIO_COST}} |
| Prefill瓶颈分析 | {{SUMMARY_PREFILL_ANALYSIS}} |

### 9.3 下一步行动

1. **立即**: {{NEXT_STEP_1}}
2. **1-2周**: {{NEXT_STEP_2}}
3. **持续**: {{NEXT_STEP_3}}

---

**报告版本**: {{REPORT_VERSION}}
**最后更新**: {{REPORT_DATE}}
**测试工具**: benchmark_heter_tp.sh + analyze_logs.py
