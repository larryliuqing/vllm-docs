# Qwen3-Omni-30B-A3B 模型测试指南

## 模型对比

| 指标 | Qwen2.5-Omni-7B | Qwen3-Omni-30B-A3B |
|------|-----------------|-------------------|
| 参数量 | 7B | 30B (A3B MoE) |
| 模型大小 | 16.68 GB | 66 GB |
| 架构 | Dense | MoE (Mixture of Experts) |
| 最少卡数 | 1卡 | 4卡 |
| 每卡显存 | ~17GB (单卡) | ~16.5GB (4卡) |
| 推理质量 | 基准 | 更优 |
| max_model_len | 建议 8192 | 建议 8192-16384 |

## 模型特点

### Qwen3-Omni-30B-A3B 优势

1. **更大容量** - 30B 参数，更强大的理解能力
2. **MoE 架构** - Active Parameters 仅部分激活，推理效率高
3. **更高质量** - 更好的多模态理解和生成质量
4. **更新版本** - Qwen3 系列，性能优化

### 适用场景

- ✅ 高质量多模态理解
- ✅ 复杂场景分析
- ✅ 专业领域应用
- ✅ 更准确的视频/音频理解

## 启动配置

### 最低配置（4卡）

```bash
# 基础配置
bash scripts/start_qwen3_omni_30b_4npu.sh 8192 8002

# 高配置（支持更长上下文）
bash scripts/start_qwen3_omni_30b_4npu.sh 16384 8002
```

### 手动启动命令

```bash
docker run --rm \
    --device=/dev/davinci4:/dev/davinci0 \
    --device=/dev/davinci5:/dev/davinci1 \
    --device=/dev/davinci6:/dev/davinci2 \
    --device=/dev/davinci7:/dev/davinci3 \
    --device=/dev/davinci_manager \
    --device=/dev/devmm_svm \
    --device=/dev/hisi_hdc \
    -v /usr/local/Ascend:/usr/local/Ascend \
    -v /home/bes/work/vllm-project/models:/models \
    -e ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 \
    -e SOC_VERSION=ascend910b4 \
    -w /tmp \
    vllm-omni:v0.20.2rc \
    bash -c "
        source /usr/local/Ascend/cann/set_env.sh
        source /usr/local/Ascend/nnal/atb/set_env.sh

        python3 -m vllm.entrypoints.openai.api_server \
            --model /models/Qwen/Qwen3-Omni-30B-A3B-Instruct \
            --trust-remote-code \
            --port 8002 \
            --host 0.0.0.0 \
            --max-model-len 8192 \
            --tensor-parallel-size 4 \
            --gpu-memory-utilization 0.90
    "
```

## 预期性能

| 指标 | Qwen2.5-Omni-7B | Qwen3-Omni-30B-A3B |
|------|-----------------|-------------------|
| 文本推理速度 | ~30-33 tok/s | ~20-25 tok/s (预估) |
| 图像理解质量 | 良好 | 优秀 |
| 音频理解质量 | 良好 | 优秀 |
| 视频理解质量 | 基础 | 更准确 |
| 启动时间 | ~5分钟 | ~6-8分钟 |

## 注意事项

### 1. 显存需求
- 总显存: 66GB
- 4卡分配: 每卡 ~16.5GB
- 建议使用 `gpu_memory_utilization=0.90`

### 2. 加载时间
- 权重文件: 15个分片
- 加载时间: 预计 60-90 秒
- 编译时间: ~3-4 分钟
- 总启动时间: ~6-8 分钟

### 3. MoE 特性
- Active Experts: 部分激活
- 实际推理计算量 < 30B dense
- 推理速度相对较快

## 测试要点

### 基础功能测试
```bash
# 获取容器 IP
CONTAINER_IP=$(docker inspect $(docker ps -q --filter "ancestor=vllm-omni:v0.20.2rc" | head -1) | grep '"IPAddress"' | head -1 | awk -F'"' '{print $4}')

# 文本对话测试
curl -X POST http://$CONTAINER_IP:8002/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/models/Qwen/Qwen3-Omni-30B-A3B-Instruct",
    "messages": [{"role": "user", "content": "请介绍你自己。"}],
    "max_tokens": 100
  }'
```

### 多模态测试
```bash
# 使用相同的测试脚本
python3 scripts/test_omni_av_local.py $CONTAINER_IP 8002
```

## 对比测试清单

- [ ] 服务启动成功
- [ ] 文本对话质量对比
- [ ] 图像理解质量对比
- [ ] 音频理解质量对比
- [ ] 视频理解质量对比
- [ ] 推理速度对比
- [ ] 显存占用监控

## 预期优势场景

1. **复杂图像理解** - 更多细节识别
2. **长音频处理** - 更准确的语音识别
3. **视频内容分析** - 更深入的场景理解
4. **多模态推理** - 更复杂的跨模态理解任务

## 故障排查

### 问题1: 显存不足
```
解决方案: 降低 gpu_memory_utilization 或减少 max_model_len
```

### 问题2: 加载超时
```
原因: 模型文件大，加载慢
解决: 耐心等待，或检查磁盘IO性能
```

### 问题3: 推理速度慢
```
原因: 30B 参数量大
解决: 这是正常现象，关注质量提升
```

## 结果记录模板

```
启动时间: ___ 分钟
每卡显存: ___ GB
文本推理: ___ tok/s
图像理解质量: ___ (评分1-5)
音频理解质量: ___ (评分1-5)
视频理解质量: ___ (评分1-5)
综合评价: ___
```
