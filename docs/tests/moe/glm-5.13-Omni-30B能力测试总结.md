# Qwen3-Omni-30B-A3B 多模态能力测试总结

**测试日期**: 2026-06-23
**模型**: Qwen3-Omni-30B-A3B-Instruct (66GB, 30B MoE)
**环境**: Ascend 910B4 × 4 (Tensor Parallel=4)

---

## 模型能力确认

根据模型配置文件和官方文档，Qwen3-Omni-30B-A3B-Instruct 支持：

### ✅ 理解能力 (Input)
- **图像理解** ✅
- **音频理解** ✅
- **视频理解** ✅
- **文本理解** ✅

### ✅ 生成能力 (Output)
- **文本生成** ✅
- **音频生成 (TTS)** ✅
  - `enable_audio_output: True`
  - 支持 10 种语音输出语言
  - 采样率: 24000 Hz
  - 实时流式语音响应

### ❌ 不支持的能力
- **图像生成** ❌ (无相关配置)
- **视频生成** ❌ (无相关配置)

---

## 架构说明

### Thinker-Talker 设计

1. **Thinker**: 理解模块
   - 处理文本、图像、音频、视频输入
   - 理解多模态内容

2. **Talker**: 生成模块
   - 音频生成 (TTS)
   - 实时语音响应

3. **Code2Wav**: 音频波形生成
   - 将编码转换为音频波形
   - 输出 24kHz 音频

---

## 音频生成测试

### 测试方法

使用 vLLM Python API 直接调用模型生成音频：

```python
import os
os.environ['VLLM_USE_V1'] = '0'  # 必须关闭 v1

from vllm import LLM, SamplingParams
from transformers import Qwen3OmniMoeProcessor
import soundfile as sf

# 初始化
llm = LLM(
    model="/models/Qwen/Qwen3-Omni-30B-A3B-Instruct",
    trust_remote_code=True,
    tensor_parallel_size=4,
    max_model_len=8192,
)

processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL_PATH)

# 准备输入
messages = [{
    "role": "user",
    "content": [{"type": "text", "text": "请用语音说：你好，我是通义千问。"}]
}]

# 生成
outputs = llm.generate([inputs], sampling_params=sampling_params)

# 保存音频
if hasattr(output, 'audio') and output.audio is not None:
    sf.write("output.wav", output.audio, samplerate=24000)
```

### 测试脚本

脚本位置: `/vllm-docs/omni-test/scripts/test_audio_generation.py`

运行命令:
```bash
docker exec <container_id> bash -c "
source /usr/local/Ascend/cann/set_env.sh
export VLLM_USE_V1=0
python3 /test-data/scripts/test_audio_generation.py
"
```

---

## 支持的语言

### 语音输入 (19种)
English, Chinese, Korean, Japanese, German, Russian, Italian, French, Spanish, Portuguese, Malay, Dutch, Indonesian, Turkish, Vietnamese, Cantonese, Arabic, Urdu

### 语音输出 (10种)
English, Chinese, French, German, Russian, Italian, Spanish, Portuguese, Japanese, Korean

---

## 与 Qwen2.5-Omni-7B 对比

| 特性 | Qwen2.5-Omni-7B | Qwen3-Omni-30B-A3B |
|------|----------------|-------------------|
| 参数量 | 7B | 30B (MoE) |
| 模型大小 | 16.68 GB | 66 GB |
| 最少卡数 | 1卡 | 4卡 |
| 音频生成 | ❓ 未测试 | ✅ 支持 |
| 理解质量 | 良好 | 优秀 |
| 语音输出语言 | - | 10种 |

---

## 使用场景

### ✅ 适合场景

1. **语音助手** - 文本转语音实时响应
2. **多语言对话** - 支持10种语音输出
3. **视频理解** - 大模型容量，理解能力强
4. **复杂音频分析** - 音乐、环境音等

### ⚠️ 限制

1. **不支持图像生成** - 需要使用 Stable Diffusion 等专用模型
2. **不支持视频生成** - 需要使用 Sora、Runway 等模型
3. **资源需求高** - 需要4卡才能运行

---

## 测试结果

### 正在进行中
- [ ] 音频生成测试 (脚本已运行)
- [ ] 音频质量评估
- [ ] 多语言语音测试

---

## 下一步

1. 等待音频生成测试完成
2. 检查生成的音频文件
3. 评估音频质量
4. 测试其他语言语音生成

---

## 结论

**Qwen3-Omni-30B-A3B 是一个强大的多模态理解 + 音频生成模型**

**核心优势**:
- ✅ 完整的多模态理解（文本/图像/音频/视频）
- ✅ 原生音频生成能力（TTS）
- ✅ 支持多语言语音输出
- ✅ 实时流式响应

**适用场景**:
- 多模态对话系统
- 语音助手应用
- 视频内容分析
- 多语言交互系统
