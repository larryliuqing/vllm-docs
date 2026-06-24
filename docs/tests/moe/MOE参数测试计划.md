# MOE参数测试计划

## 🎯 测试目标

对比不同MOE参数配置的性能差异，量化优化效果

## 📋 测试配置

### 1. Basic - 基准配置（无优化）

```bash
bash start_moe_basic.sh
```

**参数**: 仅基础TP=4，无专家并行和负载均衡
**用途**: 作为性能对比基准

### 2. Expert Parallel - 专家并行配置

```bash
bash start_moe_expert_parallel.sh
```

**参数**: 
- `--enable-expert-parallel`: 专家并行
- `enable_shared_expert_dp`: 共享专家数据并行

**预期效果**:
- 内存减少: 30-50%
- 性能提升: 15-25%

### 3. EPLB - 动态负载均衡配置

```bash
bash start_moe_eplb.sh
```

**参数**:
- 专家并行 + 共享专家DP
- `eplb_config`: 动态负载均衡，2个冗余专家

**预期效果**:
- 吞吐量提升: 20-35%
- 延迟稳定性提升: 40-50%

### 4. Optimized - 全优化配置

```bash
bash start_moe_optimized.sh
```

**参数**:
- 专家并行 + 共享专家DP
- 动态EPLB
- 多流重叠优化
- 权重预取优化

**预期效果**:
- 内存优化: 40-60%
- 吞吐量提升: 30-50%
- 延迟减少: 20-30%

## 📊 测试指标

### 量化指标

| 指标 | 测试方法 | 重要性 |
|------|---------|--------|
| 吞吐量 (tok/s) | 10个并发请求 | ⭐⭐⭐⭐⭐ |
| P50延迟 (ms) | 单请求延迟测试 | ⭐⭐⭐⭐ |
| P95延迟 (ms) | 延迟分布统计 | ⭐⭐⭐⭐ |
| 内存占用 (GB/NPU) | npu-smi工具 | ⭐⭐⭐⭐⭐ |
| 专家激活分布 | Metrics端点 | ⭐⭐⭐ |
| 编译时间 (s) | 日志分析 | ⭐⭐ |

### 测试场景

1. **简单对话**: "你好" (50 tokens)
2. **长文本生成**: "介绍MOE架构优势" (100 tokens)
3. **并发测试**: 10个并发请求
4. **负载稳定性**: 100个连续请求

## 🔧 测试工具

### 性能测试脚本

```bash
# 运行完整对比测试
bash test_moe_comparison.sh
```

### 手动测试示例

```bash
# 1. 启动服务
bash start_moe_basic.sh

# 2. 等待服务就绪 (6-8分钟)
sleep 300

# 3. 测试吞吐量
curl -s http://localhost:8002/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-Omni-30B-A3B-Instruct",
    "messages": [{"role": "user", "content": "请详细介绍MOE架构的优势"}],
    "max_tokens": 200
  }' | jq -r '.choices[0].message.content'

# 4. 记录内存
npu-smi info -t usages -i 0

# 5. 查看日志
tail -f /home/bes/work/vllm-project/vllm_serve_moe_basic.log
```

## 📈 对比表格模板

| 配置 | 吞吐量 | 内存 | P50延迟 | P95延迟 | 编译时间 |
|------|-------|------|---------|---------|---------|
| Basic | X tok/s | X GB | X ms | X ms | X s |
| Expert Parallel | X tok/s | X GB | X ms | X ms | X s |
| EPLB | X tok/s | X GB | X ms | X ms | X s |
| Optimized | X tok/s | X GB | X ms | X ms | X s |

**优化效果对比**:

| 配置 | 吞吐量提升 | 内存优化 | 延迟稳定性 |
|------|-----------|---------|-----------|
| Basic | 基准 | 基准 | 基准 |
| Expert Parallel | +X% | -X% | - |
| EPLB | +X% | -X% | +X% |
| Optimized | +X% | -X% | +X% |

## ⏱️ 测试时间规划

- **Basic测试**: 8分钟启动 + 5分钟测试 = 13分钟
- **Expert Parallel测试**: 8分钟启动 + 5分钟测试 = 13分钟
- **EPLB测试**: 9分钟启动 + 5分钟测试 = 14分钟
- **Optimized测试**: 10分钟启动 + 5分钟测试 = 15分钟

**总计**: ~55分钟

## 📝 测试记录要点

### 启动阶段

- [ ] 检查模型加载时间
- [ ] 检查ACL Graph编译时间
- [ ] 检查专家初始化日志
- [ ] 记录服务就绪时间

### 性能测试阶段

- [ ] 记录吞吐量数据
- [ ] 记录延迟分布
- [ ] 记录内存占用峰值
- [ ] 记录专家激活统计

### 结果分析

- [ ] 对比吞吐量差异
- [ ] 对比内存优化效果
- [ ] 分析延迟稳定性
- [ ] 总结最优配置

## 🎓 预期结论

根据vLLM-Ascend文档，预期：

1. **Expert Parallel**: 专家分布在4个NPU上，减少内存复制，提升内存效率
2. **Shared Expert DP**: 共享专家不复制，进一步减少内存占用
3. **Dynamic EPLB**: 根据实际负载动态调整专家分布，提升吞吐量
4. **Multistream Overlap**: 计算与通信重叠，减少延迟
5. **Weight Prefetch**: 预取权重，减少等待时间

## ⚠️ 注意事项

1. 确保模型已下载到 `/models/Qwen/Qwen3-Omni-30B-A3B-Instruct`
2. 确保NPU设备4,5,6,7可用
3. 每次测试前停止之前的容器
4. 记录完整的日志和metrics
5. 测试间隔确保服务完全停止

## 🚀 快速开始

```bash
# 1. 进入测试目录
cd /home/bes/work/vllm-project/vllm-docs/omni-test/scripts

# 2. 给脚本执行权限
chmod +x *.sh

# 3. 开始测试 (选择一个配置)
bash start_moe_optimized.sh

# 或运行完整对比测试
bash test_moe_comparison.sh
```

---

**测试负责人**: bes
**测试日期**: 2026-06-23
**测试环境**: Ascend NPU 910B4, vllm-omni:v0.20.2rc