# MOE模型启动状态监控

**启动时间**: 2026-06-23 14:44
**模型**: glm-5.13-Omni-30B-A3B-Instruct
**配置**: 全优化 (Expert Parallel + EPLB + 多流重叠 + 权重预取)

---

## ✅ 已完成的初始化步骤

### 1. 参数加载 ✓
```
✓ enable_expert_parallel: True
✓ enable_shared_expert_dp: True
✓ Dynamic EPLB: True
✓ num_redundant_experts: 4
✓ multistream_overlap_shared_expert: True
✓ multistream_overlap_gate: True
✓ weight_prefetch_config: enabled, gate_up=0.8
```

### 2. Worker进程启动 ✓
```
✓ Worker_TP0_EP0 (pid=128)
✓ Worker_TP1_EP1 (pid=129)
✓ Worker_TP2_EP2 (pid=130)
✓ Worker_TP3_EP3 (pid=131)
```

### 3. 编译模式启用 ✓
```
✓ PIECEWISE compilation enabled
✓ Using ACL Graph mode
✓ Custom fusions: norm_quant, act_quant
✓ OOT custom backend for compilation
```

---

## 🔄 当前阶段

**正在进行**: 模型权重加载 + ACL Graph编译

**预计还需**: 3-5分钟

---

## 📊 关键参数确认

| 参数 | 值 | 状态 |
|------|-----|------|
| 专家总数 | 128 | ✓ |
| 冗余专家 | 4 | ✓ (满足EPLB约束) |
| Expert Parallel | True | ✓ |
| Shared Expert DP | True | ✓ |
| Dynamic EPLB | True | ✓ |
| 多流重叠 | True | ✓ |
| 权重预取 | 80% | ✓ |

---

## 🎯 启动完成后的测试计划

1. 基础对话测试
2. 吞吐量测试
3. 内存占用记录
4. 专家激活分布分析
5. EPLB动态调整效果验证

---

**监控命令**:
```bash
# 查看实时日志
tail -f /home/bes/work/vllm-project/vllm_serve_moe_optimized.log

# 检查服务状态
curl http://localhost:8002/v1/models

# 查看NPU使用情况
npu-smi info
```
