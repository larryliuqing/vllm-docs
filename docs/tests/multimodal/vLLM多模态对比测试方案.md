# vLLM 多模态模型测试对比方案

## 测试目的

对比 `vllm-ascend` 和 `vllm-omni` 两个镜像对多模态模型的支持情况。

## 镜像对比

| 镜像 | 版本 | 专长 |
|------|------|------|
| **vllm-ascend** | v0.20.2rc | 通用 LLM + 部分多模态支持 |
| **vllm-omni** | v0.20.2rc | 专为 Omni 多模态模型优化 |

## vLLM 支持的多模态模型

### 图像理解模型

| 模型 | 参数量 | 特点 | vllm-ascend | vllm-omni |
|------|--------|------|-------------|-----------|
| **LLaVA-1.5** | 7B/13B | 流行的视觉语言模型 | ✅ | ❓ |
| **LLaVA-NeXT** | 7B-110B | 改进版本 | ✅ | ❓ |
| **Qwen-VL** | 7B | 阿里通义千问视觉模型 | ✅ | ✅ |
| **InternVL** | 2B-26B | 书生视觉大模型 | ✅ | ❓ |
| **CogVLM** | 17B | 智谱视觉模型 | ✅ | ❓ |
| **Fuyu** | 8B | ADEPT 视觉模型 | ✅ | ❓ |
| **Qwen2.5-Omni** | 7B | 多模态全栈模型 | ❓ | ✅ |
| **Qwen3-Omni** | 30B | MoE多模态模型 | ❓ | ✅ |

### 音频理解模型

| 模型 | 特点 | vllm-ascend | vllm-omni |
|------|------|-------------|-----------|
| **Qwen-Audio** | 音频理解 | ❓ | ✅ |
| **Qwen2.5-Omni** | 音频+图像+文本 | ❓ | ✅ |
| **Qwen3-Omni** | 全模态+音频生成 | ❓ | ✅ |

## 测试计划

### 测试1: 基础文本模型

**模型**: Qwen3-0.6B
**镜像**: vllm-ascend
**目的**: 验证基础功能

```bash
docker run --rm \
    --device=/dev/davinci5:/dev/davinci0 \
    -v /usr/local/Ascend:/usr/local/Ascend \
    -v /home/bes/work/vllm-project/models:/models \
    -e ASCEND_RT_VISIBLE_DEVICES=0 \
    -e SOC_VERSION=ascend910b4 \
    vllm-ascend:v0.20.2rc \
    bash -c "python3 -m vllm.entrypoints.openai.api_server \
        --model /models/Qwen/Qwen3-0.6B \
        --port 8003"
```

### 测试2: 图像理解模型

**模型**: Qwen2.5-Omni-7B 或 Qwen-VL-Chat
**镜像**: vllm-ascend vs vllm-omni
**目的**: 对比两个镜像的多模态支持

### 测试3: 全模态模型

**模型**: Qwen3-Omni-30B-A3B
**镜像**: vllm-omni
**功能**: 文本+图像+音频+视频理解，音频生成

## 关键问题

### 1. vllm-ascend 是否支持 Omni 模型？

需要测试：
- Qwen2.5-Omni-7B 能否在 vllm-ascend 上运行
- 是否需要 vllm-omni 专用镜像
- 两个镜像的功能差异

### 2. 多模态 API 差异

- 图像输入格式
- 音频输入格式
- 视频输入格式
- 多模态混合输入

### 3. 性能差异

- 推理速度
- 内存占用
- 并发能力

## 测试方法

### 单卡测试脚本

```bash
# vllm-ascend 测试
bash test_vllm_ascend.sh <model_path> <port>

# vllm-omni 测试
bash test_vllm_omni.sh <model_path> <port>
```

### 多模态测试

```python
# 图像理解
curl -X POST http://localhost:8003/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "model_name",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "描述图片"},
        {"type": "image_url", "image_url": {"url": "image_url"}}
      ]
    }]
  }'
```

## 预期结果

| 功能 | vllm-ascend | vllm-omni |
|------|-------------|-----------|
| 基础文本模型 | ✅ | ✅ |
| 图像理解模型 | ✅ | ✅ |
| 音频理解 | ❓ | ✅ |
| 视频理解 | ❓ | ✅ |
| 音频生成 | ❌ | ✅ |
| Omni 专用模型 | ❌ | ✅ |

## 结论

- **vllm-ascend**: 适合标准 LLM 和基础多模态模型
- **vllm-omni**: 专为 Omni 系列优化，支持全模态理解+音频生成

**建议**: 
- 文本模型：使用 vllm-ascend
- 多模态模型（尤其是 Omni）：使用 vllm-omni
