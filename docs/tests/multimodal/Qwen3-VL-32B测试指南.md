# Qwen3-VL-32B-Instruct 测试指南

## 模型信息

| 项目 | 详情 |
|------|------|
| 模型名称 | Qwen3-VL-32B-Instruct |
| 参数量 | 32B |
| 模型大小 | ~60GB |
| 最少卡数 | 4卡 (推荐) |
| 支持功能 | 图像理解、视频理解 |
| 中文支持 | ✅ 优秀 |

---

## 模型下载

```bash
cd /home/bes/work/vllm-project/models
modelscope download --model Qwen/Qwen3-VL-32B-Instruct --local_dir Qwen/Qwen3-VL-32B-Instruct
```

**下载进度**: 后台运行中...

---

## 启动配置

### 4卡配置 (推荐)

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
            --model /models/Qwen/Qwen3-VL-32B-Instruct \
            --trust-remote-code \
            --port 8003 \
            --host 0.0.0.0 \
            --max-model-len 8192 \
            --tensor-parallel-size 4 \
            --gpu-memory-utilization 0.90
    "
```

### 预期性能

| 指标 | 预估值 |
|------|--------|
| 每卡显存 | ~16GB |
| 启动时间 | ~6-8分钟 |
| 推理速度 | ~20-25 tok/s |

---

## 测试用例

### 1. 文本对话

```bash
curl -X POST http://172.17.0.2:8003/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/models/Qwen/Qwen3-VL-32B-Instruct",
    "messages": [{"role": "user", "content": "你好，请介绍你自己。"}],
    "max_tokens": 100
  }'
```

### 2. 图像理解

```bash
curl -X POST http://172.17.0.2:8003/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/models/Qwen/Qwen3-VL-32B-Instruct",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "请详细描述这张图片的内容。"},
        {"type": "image_url", "image_url": {"url": "https://picsum.photos/512/512"}}
      ]
    }],
    "max_tokens": 300
  }'
```

### 3. 视频理解

```bash
curl -X POST http://172.17.0.2:8003/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/models/Qwen/Qwen3-VL-32B-Instruct",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "请描述这个视频的内容。"},
        {"type": "video_url", "video_url": {"url": "video_url"}}
      ]
    }],
    "max_tokens": 300
  }'
```

---

## 与其他模型对比

### Qwen3-VL vs Qwen2-VL

| 特性 | Qwen2-VL | Qwen3-VL |
|------|----------|----------|
| 参数量 | 2B/7B/72B | 32B |
| 图像理解 | ✅ | ✅ 更强 |
| 视频理解 | ✅ | ✅ 更强 |
| 推理质量 | 优秀 | 更优秀 |
| 中文支持 | ✅ | ✅ |

### Qwen3-VL vs Qwen3-Omni

| 特性 | Qwen3-VL | Qwen3-Omni |
|------|----------|------------|
| 视觉理解 | ✅ 专精 | ✅ |
| 音频理解 | ❌ | ✅ |
| 音频生成 | ❌ | ✅ |
| 参数量 | 32B | 30B (MoE) |
| 定位 | 视觉专家 | 全模态 |

---

## 注意事项

### 1. 显存需求

- 32B 模型需要约 60GB 显存
- 4卡每卡约 16GB
- 建议 `gpu_memory_utilization=0.90`

### 2. max_model_len 配置

```bash
# 图像理解
--max-model-len 4096  # 足够

# 视频理解
--max-model-len 8192  # 推荐

# 长视频
--max-model-len 16384  # 需要更多显存
```

### 3. 性能优化

```bash
# 提升通信性能
export HCCL_OP_EXPANSION_MODE=AIV
```

---

## 测试清单

- [ ] 模型下载完成
- [ ] 4卡启动测试
- [ ] 文本对话测试
- [ ] 图像理解测试
- [ ] 视频理解测试
- [ ] 性能基准测试
- [ ] 与 Qwen2-VL 对比
- [ ] 与 Qwen3-Omni 对比

---

## 预期测试结果

### 图像理解质量

Qwen3-VL-32B 作为专门视觉模型，预期在：
- 图像细节描述
- OCR 文字识别
- 图表理解
- 复杂场景分析

方面表现优秀。

### 与 Omni 模型的区别

**Qwen3-VL**: 视觉理解专精
**Qwen3-Omni**: 全模态（视觉+音频+音频生成）

根据任务需求选择合适的模型。
