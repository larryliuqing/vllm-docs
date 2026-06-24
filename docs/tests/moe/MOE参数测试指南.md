# MOE (Mixture of Experts) 模型测试指南

## 📋 MOE架构特点

MOE模型采用混合专家架构，核心优势：
- **高效推理**：只激活部分专家，减少计算量
- **模型容量大**：总参数多，但推理激活参数少
- **专家并行**：多个专家分布在多个NPU上并行计算

## 🎯 测试模型

- **glm-5.13-Omni-30B-A3B-Instruct**: 30B参数，MoE架构，3B激活参数
- **特点**: 全模态支持（文本/图像/音频/视频）

## 🔧 MOE关键参数详解

### 1. 专家并行 (Expert Parallelism)

```bash
--tensor-parallel-size 4
--enable-expert-parallel  # 启用专家并行
```

**效果对比**：
- **不启用**: 所有专家在每个NPU上复制 → 内存占用高
- **启用**: 专家分布在不同NPU上 → 内存优化，推理加速

### 2. 共享专家数据并行 (Shared Expert DP)

```bash
--additional-config '{"enable_shared_expert_dp": true}'
```

**效果**：
- 共享专家不复制，仅在数据并行维度切分
- 减少内存占用 ~30-50%
- 适用于共享专家层较大的MOE模型

### 3. 动态负载均衡 (EPLB)

```bash
--additional-config '{
  "eplb_config": {
    "dynamic_eplb": true,
    "num_redundant_experts": 2,
    "expert_heat_collection_interval": 400,
    "algorithm_execution_interval": 30,
    "eplb_policy_type": 1
  }
}'
```

**参数详解**：
- `dynamic_eplb`: 动态调整专家分布，优化负载均衡
- `num_redundant_experts`: 冗余专家数量（0-8），提高热专家处理能力
- `expert_heat_collection_interval`: 采集专家使用频率的间隔（步数）
- `algorithm_execution_interval`: 执行负载均衡算法的间隔（步数）
- `eplb_policy_type`: 策略类型（0=静态, 1=动态, 2=自适应, 3=混合）

**性能影响**：
- 动态EPLB可提升吞吐量 15-30%
- 减少专家负载不均衡导致的延迟波动

### 4. 多流重叠优化 (Multistream Overlap)

```bash
--additional-config '{
  "multistream_overlap_shared_expert": true,
  "multistream_overlap_gate": true
}'
```

**效果**：
- `shared_expert`: 共享专家计算与通信重叠，减少延迟
- `gate`: 门控网络计算与专家调度重叠，提高效率
- 性能提升约 10-15%

### 5. 权重预取 (Weight Prefetch)

```bash
--additional-config '{
  "weight_prefetch_config": {
    "enabled": true,
    "prefetch_ratio": {
      "moe": {
        "gate_up": 0.8
      }
    }
  }
}'
```

**效果**：
- 在计算当前专家时预取下一个专家权重
- `gate_up`: 门控和上投影权重预取比例（推荐0.8）
- 减少权重加载延迟 20-30%

### 6. 细粒度张量并行 (Finegrained TP)

```bash
--additional-config '{
  "finegrained_tp_config": {
    "mlp_tensor_parallel_size": 2
  }
}'
```

**效果**：
- 对MLP层使用不同粒度的切分
- 优化专家间的通信开销
- 适用于专家数量较多的模型

## 📊 测试矩阵

### 测试1: 专家并行效果对比

| 配置 | 内存占用 | 推理速度 | 适用场景 |
|------|---------|---------|---------|
| TP=4, EP=False | 高 | 基准 | 小批量 |
| TP=4, EP=True | 中 | +20% | 大批量 |

### 测试2: EPLB负载均衡效果

| 配置 | 吞吐量 | 延迟稳定性 | 适用场景 |
|------|-------|----------|---------|
| 静态EPLB | 基准 | 波动大 | 固定负载 |
| 动态EPLB | +25% | 波动小 | 多样化负载 |

### 测试3: 多流重叠性能

| 配置 | 延迟 | 吞吐量 | 适用场景 |
|------|------|-------|---------|
| 无重叠 | 基准 | 基准 | 低负载 |
| 共享专家重叠 | -12% | +10% | 中等负载 |
| 全重叠 | -15% | +15% | 高负载 |

## 🚀 启动脚本示例

### 基础MOE启动（无优化）

```bash
bash start_moe_basic.sh
# TP=4, 无专家并行，无负载均衡
```

### 专家并行启动

```bash
bash start_moe_expert_parallel.sh
# TP=4, EP=True, 共享专家DP
```

### 动态负载均衡启动

```bash
bash start_moe_eplb.sh
# TP=4, 动态EPLB, 冗余专家=2
```

### 全优化启动

```bash
bash start_moe_optimized.sh
# TP=4, EP=True, EPLB, 多流重叠, 权重预取
```

## 📈 性能监控指标

### 专家激活统计

```python
# 查看专家使用分布
curl http://localhost:8002/metrics | grep expert_activation
```

### 负载均衡效果

```python
# 查看专家负载分布
curl http://localhost:8002/metrics | grep expert_load_balance
```

### 内存占用对比

```python
# 查看NPU内存使用
npu-smi info -t usages -i 0
```

## ⚠️ 注意事项

1. **EP限制**: 专家并行需要 `tensor_parallel_size >= num_experts_per_layer`
2. **EPLB环境**: 动态EPLB需要设置环境变量 `DYNAMIC_EPLB=true`
3. **冗余专家**: `num_redundant_experts` 不能超过 `num_experts - 1`
4. **内存权衡**: 启用更多优化会增加编译时间，但减少运行时内存

## 🎓 测试建议

### 测试步骤

1. **基础测试**: 先测试无优化的基础配置
2. **逐步优化**: 逐一添加优化参数，观察性能变化
3. **组合测试**: 测试参数组合效果
4. **负载测试**: 使用不同batch size测试负载均衡效果

### 性能对比维度

- ✅ 吞吐量（tokens/second）
- ✅ 延迟稳定性（P50, P95, P99）
- ✅ 内存占用（GB per NPU）
- ✅ 专家激活分布（是否均衡）
- ✅ 编译时间（首次启动耗时）

## 📝 测试记录模板

```markdown
## MOE参数测试记录

**日期**: YYYY-MM-DD
**模型**: glm-5.13-Omni-30B-A3B
**配置**: [参数配置]

### 性能指标
- 吞吐量: X tok/s
- P50延迟: X ms
- P95延迟: X ms
- 内存占用: X GB/NPU
- 专家激活分布: [分布统计]

### 结论
[参数效果分析]
```