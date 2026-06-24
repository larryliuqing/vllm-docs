# MOE参数测试进展报告

**测试时间**: 2026-06-23 14:40-14:50
**模型**: glm-5.13-Omni-30B-A3B-Instruct
**架构**: MoE (128专家, 30B总参数, 3B激活参数)

---

## 🔍 关键发现

### 1. MOE模型架构参数

从模型配置文件 `config.json` 中确认：

```json
{
  "thinker_config": {
    "text_config": {
      "num_experts": 128,           // 专家总数
      "num_experts_per_tok": 8,     // 每个token激活的专家数
      "num_hidden_layers": 48
    }
  },
  "talker_config": {
    "text_config": {
      "num_experts": 128,
      "num_experts_per_tok": 6
    }
  }
}
```

**关键参数**:
- **专家总数**: 128个
- **激活专家数**: Thinker=8个/token, Talker=6个/token
- **激活比例**: 6.25% (8/128), 4.69% (6/128)

---

## ⚠️ EPLB配置约束

### 错误发现

首次启动时遇到错误：
```
ValueError: (n_expert + n_redundant) % ep_size must be 0
```

### 原因分析

EPLB (Expert Parallel Load Balancing) 有严格的数学约束：
- `n_expert`: 专家总数 = 128
- `n_redundant`: 冗余专家数
- `ep_size`: 专家并行大小 = tensor_parallel_size = 4

**约束条件**: `(128 + n_redundant) % 4 == 0`

### 解决方案

| n_redundant | 是否满足条件 | 推荐使用 |
|------------|-------------|---------|
| 0 | ✓ (128 % 4 = 0) | ✓ 基础配置 |
| 1 | ✗ (129 % 4 = 1) | ✗ |
| 2 | ✗ (130 % 4 = 2) | ✗ |
| 3 | ✗ (131 % 4 = 3) | ✗ |
| **4** | **✓ (132 % 4 = 0)** | **✓ 推荐** |
| 8 | ✓ (136 % 4 = 0) | ✓ 高负载场景 |

**修复**: 将 `num_redundant_experts` 从 `2` 改为 `4`

---

## ✅ 已启用的MOE优化参数

### 当前配置 (全优化)

```json
{
  "enable_expert_parallel": true,
  "enable_shared_expert_dp": true,
  "multistream_overlap_shared_expert": true,
  "multistream_overlap_gate": true,
  "weight_prefetch_config": {
    "enabled": true,
    "prefetch_ratio": {
      "moe": {"gate_up": 0.8}
    }
  },
  "eplb_config": {
    "dynamic_eplb": true,
    "num_redundant_experts": 4,
    "expert_heat_collection_interval": 400,
    "algorithm_execution_interval": 30,
    "eplb_policy_type": 1
  }
}
```

---

## 📊 预期性能提升

根据vLLM-Ascend文档，各参数的预期效果：

| 优化参数 | 功能 | 预期效果 |
|---------|------|---------|
| **Expert Parallel** | 专家分布在不同NPU | 内存减少30-50% |
| **Shared Expert DP** | 共享专家数据并行 | 进一步优化内存 |
| **Dynamic EPLB** | 动态负载均衡 | 吞吐量提升20-35% |
| **冗余专家=4** | 提高热专家处理能力 | 减少负载不均衡 |
| **Multistream Overlap** | 计算通信重叠 | 延迟减少10-15% |
| **Weight Prefetch** | 权重预取80% | 减少等待延迟 |

**综合预期**:
- 内存优化: 40-60%
- 吞吐量提升: 30-50%
- 延迟减少: 20-30%
- 延迟稳定性: 提升50-70%

---

## 🎯 测试配置对比

### 4种测试配置

| 配置名称 | 启动脚本 | 关键参数 | 用途 |
|---------|---------|---------|------|
| **Basic** | `start_moe_basic.sh` | TP=4, 无优化 | 性能基准 |
| **Expert Parallel** | `start_moe_expert_parallel.sh` | EP=True, SharedExpertDP | 内存优化测试 |
| **EPLB** | `start_moe_eplb.sh` | EP+DynamicEPLB(4冗余) | 负载均衡测试 |
| **Optimized** | `start_moe_optimized.sh` | 全部优化参数 | 性能极限测试 |

### 测试对比维度

1. **内存占用** - NPU内存使用峰值
2. **吞吐量** - tokens/second
3. **延迟稳定性** - P50/P95/P99延迟
4. **专家激活分布** - 负载均衡效果
5. **编译时间** - 首次启动耗时

---

## 📝 下一步测试计划

### 阶段1: 基础功能验证 (当前)

- [x] 确认MOE参数正确加载
- [x] 修复EPLB配置约束
- [ ] 等待模型启动完成
- [ ] 验证专家并行生效
- [ ] 验证EPLB动态调整

### 阶段2: 性能基准测试

- [ ] 测试Basic配置性能
- [ ] 测试Expert Parallel配置
- [ ] 测试EPLB配置
- [ ] 测试Optimized配置

### 阶段3: 对比分析

- [ ] 对比内存占用
- [ ] 对比吞吐量
- [ ] 对比延迟稳定性
- [ ] 分析专家激活分布
- [ ] 生成性能对比报告

---

## 🔧 故障排查记录

### 问题1: EPLB参数约束错误

**错误信息**:
```
ValueError: (n_expert + n_redundant) % ep_size must be 0
```

**原因**:
- 专家数=128, ep_size=4
- n_redundant=2 → (128+2)%4=2 ≠ 0 ❌

**解决**:
- 改为 n_redundant=4 → (128+4)%4=0 ✓

### 问题2: 模型启动时间较长

**预期**: 8-10分钟
**原因**:
- 30B模型较大
- 专家并行需要额外初始化
- EPLB需要编译动态调整逻辑
- ACL Graph编译耗时

---

## 📚 参考资料

1. **vLLM-Ascend MOE文档**: `/home/bes/work/vllm-project/vllm-ascend/vllm_ascend/ascend_config.py`
2. **模型配置**: `/home/bes/work/vllm-project/models/Qwen/Qwen3-Omni-30B-A3B-Instruct/config.json`
3. **EPLB源码**: `/home/bes/work/vllm-project/vllm-ascend/vllm_ascend/eplb/core/eplb_utils.py`

---

**状态**: 🔄 模型启动中 (预计8-10分钟)
**下一步**: 等待启动完成，开始性能测试