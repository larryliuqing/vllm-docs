# Qwen2-VL-7B-Instruct 测试指南

## 模型信息

| 项目 | 详情 |
|------|------|
| 模型名称 | Qwen2-VL-7B-Instruct |
| 参数量 | 7B |
| 模型大小 | ~15GB |
| 最少卡数 | 1卡 |
| 支持功能 | 图像理解、视频理解 |
| 中文支持 | ✅ 优秀 |
| 状态 | 🔄 下载中 |

---

## 模型对比

### Qwen2-VL vs Qwen3-VL

| 特性 | Qwen2-VL-7B | Qwen3-VL-32B |
|------|-------------|--------------|
| 参数量 | 7B | 32B |
| 模型大小 | ~15GB | ~63GB |
| 显存需求 | ~17GB (单卡) | ~16GB/卡 (4卡) |
| vLLM支持 | ✅ 成熟稳定 | ⚠️ 较新 |
| Ascend兼容 | ✅ 预期良好 | ❌ 已失败 |
| 图像理解 | ✅ | ✅ |
| 视频理解 | ✅ | ✅ |
| 推理速度 | 快 | 中等 |

---

## 测试配置

### 单卡启动 (推荐)

```bash
docker run --rm \
    --device=/dev/davinci5:/dev/davinci0 \
    --device=/dev/davinci_manager \
    --device=/dev/devmm_svm \
    --device=/dev/hisi_hdc \
    -v /usr/local/Ascend:/usr/local/Ascend \
    -v /home/bes/work/vllm-project/models:/models \
    -e ASCEND_RT_VISIBLE_DEVICES=0 \
    -e SOC_VERSION=ascend910b4 \
    -w /tmp \
    vllm-omni:v0.20.2rc \
    bash -c "
        source /usr/local/Ascend/cann/set_env.sh
        source /usr/local/Ascend/nnal/atb/set_env.sh

        python3 -m vllm.entrypoints.openai.api_server \
            --model /models/Qwen2-VL-7B-Instruct \
            --trust-remote-code \
            --port 8003 \
            --host 0.0.0.0 \
            --max-model-len 8192 \
            --tensor-parallel-size 1 \
            --gpu-memory-utilization 0.85
    "
```

### 4卡启动 (高吞吐)

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
            --model /models/Qwen2-VL-7B-Instruct \
            --trust-remote-code \
            --port 8003 \
            --host 0.0.0.0 \
            --max-model-len 16384 \
            --tensor-parallel-size 4 \
            --gpu-memory-utilization 0.85
    "
```

---

## 测试用例

### 1. 文本对话

```bash
curl -X POST http://172.17.0.2:8003/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/models/Qwen2-VL-7B-Instruct",
    "messages": [{"role": "user", "content": "你好，请介绍你自己。"}],
    "max_tokens": 100
  }'
```

### 2. 图像理解

```bash
curl -X POST http://172.17.0.2:8003/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/models/Qwen2-VL-7B-Instruct",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "请详细描述这张图片。"},
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
    "model": "/models/Qwen2-VL-7B-Instruct",
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

### 4. 本地图片测试 (Base64)

```python
import base64, requests

image_path = '/home/bes/work/vllm-project/vllm-docs/omni-test/images/test_image_512.jpg'
with open(image_path, 'rb') as f:
    image_base64 = base64.b64encode(f.read()).decode('utf-8')

response = requests.post(
    'http://172.17.0.2:8003/v1/chat/completions',
    json={
        'model': '/models/Qwen2-VL-7B-Instruct',
        'messages': [{
            'role': 'user',
            'content': [
                {'type': 'text', 'text': '描述这张图片'},
                {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{image_base64}'}}
            ]
        }]
    }
)
```

---

## 预期性能

| 指标 | 单卡 | 4卡 |
|------|------|-----|
| 启动时间 | ~4分钟 | ~5分钟 |
| 显存占用 | ~17GB | ~4.5GB/卡 |
| 文本推理 | ~35 tok/s | ~30-35 tok/s |
| 图像理解 | ~12 tok/s | ~12 tok/s |
| max_model_len | 8192 | 16384 |

---

## 测试清单

- [ ] 模型下载完成
- [ ] 单卡启动测试
- [ ] 文本对话测试
- [ ] 图像理解测试 (URL)
- [ ] 图像理解测试 (Base64)
- [ ] 视频理解测试
- [ ] 性能基准测试
- [ ] 4卡对比测试 (可选)

---

## 与其他模型对比

### 视觉理解能力

| 模型 | 图像 | 视频 | 质量 | 中文 |
|------|------|------|------|------|
| **Qwen2-VL-7B** | ✅ | ✅ | 优秀 | ✅ |
| Qwen2.5-Omni-7B | ✅ | ✅ | 良好 | ✅ |
| Qwen3-Omni-30B | ✅ | ✅ | 优秀 | ✅ |
| LLaVA-1.6-7B | ✅ | ❌ | 良好 | ⚠️ |
| InternVL2-8B | ✅ | ✅ | 优秀 | ✅ |

### 推荐

- **图像理解专精**: Qwen2-VL-7B
- **全模态**: Qwen2.5-Omni-7B 或 Qwen3-Omni-30B
- **轻量级**: Qwen2-VL-2B

---

## 注意事项

### 1. 模型路径

确认模型文件存在：
```bash
ls /home/bes/work/vllm-project/models/Qwen2-VL-7B-Instruct/
```

### 2. 端口冲突

检查端口占用：
```bash
ss -tlnp | grep 8003
```

### 3. 显存配置

单卡建议：
```bash
--gpu-memory-utilization 0.85
```

4卡建议：
```bash
--gpu-memory-utilization 0.85
```

---

## 故障排查

### 问题1: 启动失败

**检查**:
```bash
tail -100 /home/bes/work/vllm-project/vllm_serve_qwen2_vl_7b.log
```

### 问题2: 图像理解失败

**检查**:
- 图像 URL 是否可访问
- Base64 编码是否正确
- max_model_len 是否足够

### 问题3: 视频理解失败

**检查**:
- 视频文件大小和格式
- max_model_len 配置（建议 >= 8192）
- 视频时长是否过长

---

## 总结

**Qwen2-VL-7B-Instruct 是一个成熟稳定的视觉语言模型**

**优势**:
- ✅ vLLM 支持完善
- ✅ 单卡即可运行
- ✅ 图像+视频理解
- ✅ 中文效果优秀
- ✅ 社区支持广泛

**适合场景**:
- 图像内容分析
- 视频内容理解
- OCR 文字识别
- 视觉问答系统
