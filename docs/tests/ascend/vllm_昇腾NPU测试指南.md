# vLLM 昇腾 NPU 测试指南

## 环境信息

| 项目 | 详情 |
|------|------|
| 硬件 | 华为 Ascend 910B4 × 4 (物理设备 4,5,6,7) |
| 操作系统 | Ubuntu 22.04 (aarch64) |
| CANN 版本 | 9.0.0 |
| Docker 镜像 | `vllm-ascend:v0.20.2rc`, `vllm-omni:v0.20.2rc` |
| 模型 | Qwen3-0.6B, Qwen2.5-Omni-7B |

---

## 一、单机多卡测试

### 1.1 前置检查

```bash
# 确认 NPU 设备可用
npu-smi info

# 确认 Docker 镜像存在
docker images | grep vllm-ascend

# 确认模型文件存在
ls /home/bes/work/vllm-project/models/Qwen/Qwen3-0.6B/
```

### 1.2 启动命令

```bash
docker run --rm \
    --device=/dev/davinci6:/dev/davinci0 \
    --device=/dev/davinci7:/dev/davinci1 \
    --device=/dev/davinci_manager \
    --device=/dev/devmm_svm \
    --device=/dev/hisi_hdc \
    -v /usr/local/Ascend:/usr/local/Ascend \
    -v /home/bes/work/vllm-project/models:/models \
    -e ASCEND_RT_VISIBLE_DEVICES=0,1 \
    -e SOC_VERSION=ascend910b4 \
    -w /tmp \
    -p 8000:8000 \
    vllm-ascend:v0.20.2rc \
    bash -c "
        source /usr/local/Ascend/cann/set_env.sh
        source /usr/local/Ascend/nnal/atb/set_env.sh

        vllm serve /models/Qwen/Qwen3-0.6B \
            --trust-remote-code \
            --port 8000 \
            --host 0.0.0.0 \
            --max-model-len 2048 \
            --tensor-parallel-size 2 \
            --gpu-memory-utilization 0.85
    "
```

### 1.3 参数说明

| 参数 | 说明 |
|------|------|
| `--device=/dev/davinci6:/dev/davinci0` | 物理 NPU 6 映射为容器内设备 0 |
| `--device=/dev/davinci7:/dev/davinci1` | 物理 NPU 7 映射为容器内设备 1 |
| `ASCEND_RT_VISIBLE_DEVICES=0,1` | 容器内可见 2 个 NPU |
| `-w /tmp` | **关键**：工作目录不能在 vllm 源码目录 |
| `source .../nnal/atb/set_env.sh` | **关键**：加载 ATB 加速库 |
| `--tensor-parallel-size 2` | 使用 2 张 NPU 做张量并行 |
| `--max-model-len 2048` | 最大序列长度 |
| `--gpu-memory-utilization 0.85` | NPU 显存使用率 85% |

### 1.4 启动成功标志

当日志中出现以下内容时，表示服务已就绪：

```
INFO:     Application startup complete.
INFO 06-23 xx:xx:xx [api_server.py:602] Starting vLLM server on http://0.0.0.0:8000
```

---

## 二、单机单卡测试

### 2.1 前置检查

```bash
# 确认 NPU 设备可用
npu-smi info

# 确认 Docker 镜像存在
docker images | grep vllm-ascend

# 确认模型文件存在
ls /home/bes/work/vllm-project/models/Qwen/Qwen3-0.6B/
```

### 2.2 启动命令

```bash
docker run --rm \
    --device=/dev/davinci4:/dev/davinci0 \
    --device=/dev/davinci_manager \
    --device=/dev/devmm_svm \
    --device=/dev/hisi_hdc \
    -v /usr/local/Ascend:/usr/local/Ascend \
    -v /home/bes/work/vllm-project/models:/models \
    -e ASCEND_RT_VISIBLE_DEVICES=0 \
    -e SOC_VERSION=ascend910b4 \
    -w /tmp \
    -p 8000:8000 \
    vllm-ascend:v0.20.2rc \
    bash -c "
        source /usr/local/Ascend/cann/set_env.sh
        source /usr/local/Ascend/nnal/atb/set_env.sh

        vllm serve /models/Qwen/Qwen3-0.6B \
            --trust-remote-code \
            --port 8000 \
            --host 0.0.0.0 \
            --max-model-len 512 \
            --tensor-parallel-size 1 \
            --gpu-memory-utilization 0.85
    "
```

### 2.3 参数说明

| 参数 | 说明 |
|------|------|
| `--device=/dev/davinci4:/dev/davinci0` | 物理 NPU 4 映射为容器内设备 0 |
| `ASCEND_RT_VISIBLE_DEVICES=0` | 容器内可见 1 个 NPU |
| `-w /tmp` | **关键**：工作目录不能在 vllm 源码目录 |
| `source .../nnal/atb/set_env.sh` | **关键**：加载 ATB 加速库 |
| `--tensor-parallel-size 1` | 单卡模式，不做张量并行 |
| `--max-model-len 512` | 较小序列长度，加快启动 |
| `--gpu-memory-utilization 0.85` | NPU 显存使用率 85% |

### 2.4 启动成功标志

当日志中出现以下内容时，表示服务已就绪：

```
INFO:     Application startup complete.
INFO 06-23 xx:xx:xx [api_server.py:602] Starting vLLM server on http://0.0.0.0:8000
```

### 2.5 快速 Python 测试

无需启动 API server，直接在容器内测试推理：

```bash
docker run --rm \
    --device=/dev/davinci4:/dev/davinci0 \
    --device=/dev/davinci_manager \
    --device=/dev/devmm_svm \
    --device=/dev/hisi_hdc \
    -v /usr/local/Ascend:/usr/local/Ascend \
    -v /home/bes/work/vllm-project/models:/models \
    -e ASCEND_RT_VISIBLE_DEVICES=0 \
    -e SOC_VERSION=ascend910b4 \
    -w /tmp \
    vllm-ascend:v0.20.2rc \
    bash -c "
        source /usr/local/Ascend/cann/set_env.sh
        source /usr/local/Ascend/nnal/atb/set_env.sh
        
        python3 << 'EOF'
import os
from vllm import LLM, SamplingParams

os.environ['ASCEND_RT_VISIBLE_DEVICES'] = '0'

llm = LLM(
    model='/models/Qwen/Qwen3-0.6B',
    trust_remote_code=True,
    max_model_len=512,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.85,
)

sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=50)
outputs = llm.generate(['Hello, my name is'], sampling_params)

for output in outputs:
    print(f'Prompt: {output.prompt!r}')
    print(f'Generated: {output.outputs[0].text!r}')
EOF
    "
```

---

## 三、使用 curl 测试 API

### 3.1 健康检查

```bash
curl http://localhost:8000/health
```

### 3.2 查看模型列表

```bash
curl http://localhost:8000/v1/models
```

返回示例：

```json
{
    "object": "list",
    "data": [
        {
            "id": "/models/Qwen/Qwen3-0.6B",
            "object": "model",
            "max_model_len": 2048,
            "owned_by": "vllm"
        }
    ]
}
```

### 3.3 文本补全 (Completions)

```bash
curl -s http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/models/Qwen/Qwen3-0.6B",
    "prompt": "Hello, my name is",
    "max_tokens": 50,
    "temperature": 0.7,
    "top_p": 0.9
  }' | python3 -m json.tool
```

返回示例：

```json
{
    "id": "cmpl-xxxxxxxxxxxxxxxx",
    "object": "text_completion",
    "model": "/models/Qwen/Qwen3-0.6B",
    "choices": [
        {
            "index": 0,
            "text": " Lucy, and I'm a student in grade 3...",
            "finish_reason": "length"
        }
    ],
    "usage": {
        "prompt_tokens": 5,
        "completion_tokens": 50,
        "total_tokens": 55
    }
}
```

### 3.4 对话补全 (Chat Completions)

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/models/Qwen/Qwen3-0.6B",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ],
    "max_tokens": 100,
    "temperature": 0.7
  }' | python3 -m json.tool
```

返回示例：

```json
{
    "id": "chatcmpl-xxxxxxxxxxxxxxxx",
    "object": "chat.completion",
    "model": "/models/Qwen/Qwen3-0.6B",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "The capital of France is Paris..."
            },
            "finish_reason": "stop"
        }
    ],
    "usage": {
        "prompt_tokens": 25,
        "completion_tokens": 30,
        "total_tokens": 55
    }
}
```

### 3.5 Token 计数

```bash
curl -s http://localhost:8000/tokenize \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/models/Qwen/Qwen3-0.6B",
    "prompt": "Hello, world!"
  }' | python3 -m json.tool
```

### 3.6 查看服务版本

```bash
curl http://localhost:8000/version
```

---

## 四、使用 Python SDK 测试

### 3.1 安装依赖

```bash
pip install openai
```

### 3.2 测试脚本

```python
from openai import OpenAI

# 连接到本地 vLLM 服务
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"  # vLLM 不需要真实的 API key
)

# 文本补全
response = client.completions.create(
    model="/models/Qwen/Qwen3-0.6B",
    prompt="Once upon a time,",
    max_tokens=100,
    temperature=0.7,
)
print(response.choices[0].text)

# 对话补全
response = client.chat.completions.create(
    model="/models/Qwen/Qwen3-0.6B",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "介绍一下北京"},
    ],
    max_tokens=200,
    temperature=0.7,
)
print(response.choices[0].message.content)
```

---

## 五、常见问题排查

### 4.1 端口被占用

```bash
# 查看端口占用
ss -tlnp | grep 8000

# 停止占用端口的容器
docker stop $(docker ps -q --filter "publish=8000")
```

### 4.2 查看容器日志

```bash
# 查看运行中的容器
docker ps | grep vllm-ascend

# 查看容器日志
docker logs -f <container_id>
```

### 4.3 NPU 设备不可用

```bash
# 检查 NPU 状态
npu-smi info

# 检查设备权限
ls -la /dev/davinci*
groups  # 确认用户在 HwHiAiUser 组中
```

### 4.4 修改 NPU 设备分配

如需使用其他物理 NPU，修改 `--device` 映射即可：

```bash
# 使用物理设备 4,5
--device=/dev/davinci4:/dev/davinci0 \
--device=/dev/davinci5:/dev/davinci1 \

# 单卡模式（使用物理设备 4）
--device=/dev/davinci4:/dev/davinci0 \
-e ASCEND_RT_VISIBLE_DEVICES=0 \
--tensor-parallel-size 1
```

### 4.5 修改模型

```bash
# 使用其他模型，只需修改模型路径
vllm serve /models/your-model-name \
    --trust-remote-code \
    ...
```

---

## 六、Omni 多模态模型测试

### 6.1 环境信息

| 项目 | 详情 |
|------|------|
| Docker 镜像 | `vllm-omni:v0.20.2rc` |
| 模型 | Qwen2.5-Omni-7B |
| 模型大小 | 16.68 GB |
| 硬件 | 单卡 Ascend 910B4 (davinci5) |
| 支持功能 | 文本、图像、音频、视频多模态 |

### 6.2 启动命令

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
            --model /models/Qwen/Qwen2___5-Omni-7B \
            --trust-remote-code \
            --port 8001 \
            --host 0.0.0.0 \
            --max-model-len 2048 \
            --tensor-parallel-size 1 \
            --gpu-memory-utilization 0.85
    " > /home/bes/work/vllm-project/vllm_serve_omni_1npu.log 2>&1 &
```

**注意**：
- 模型路径使用 `Qwen2___5-Omni-7B`（实际目录名包含下划线）
- 端口使用 8001 避免与其他服务冲突
- 后台运行并输出到日志文件

### 6.3 启动性能

| 指标 | 数值 |
|------|------|
| 权重加载时间 | 27.29 秒 |
| 模型大小 | 16.68 GB |
| 编译优化时间 | ~3 分钟 |
| 总启动时间 | ~4 分钟 |

### 6.4 文本对话测试

```bash
# 获取容器 IP
CONTAINER_IP=$(docker inspect $(docker ps -q --filter "ancestor=vllm-omni:v0.20.2rc") | grep '"IPAddress"' | head -1 | awk -F'"' '{print $4}')

# 文本对话测试
curl -X POST http://$CONTAINER_IP:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/models/Qwen/Qwen2___5-Omni-7B",
    "messages": [
      {"role": "user", "content": "你好，请用一句话介绍一下你自己。"}
    ],
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

**返回示例**：
```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "我是一个能够回答用户各种问题的AI助手。"
    }
  }],
  "usage": {
    "prompt_tokens": 26,
    "completion_tokens": 12,
    "total_tokens": 38
  }
}
```

### 6.5 图像理解测试

```bash
curl -X POST http://$CONTAINER_IP:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/models/Qwen/Qwen2___5-Omni-7B",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "请简单描述这张图片。"},
          {
            "type": "image_url",
            "image_url": {
              "url": "https://picsum.photos/seed/omni/200/200"
            }
          }
        ]
      }
    ],
    "max_tokens": 200,
    "temperature": 0.7
  }'
```

**返回示例**：
```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "这张图片展示了一条蜿蜒的小路穿过一片绿色的草地，远处是连绵起伏的山脉。天空中有一些云彩，整体景色非常自然和宁静。"
    }
  }],
  "usage": {
    "prompt_tokens": 76,
    "completion_tokens": 37,
    "total_tokens": 113
  }
}
```

**注意**：
- 大尺寸图片可能超出 `max_model_len` 限制
- 建议使用 200-500 像素的图片进行测试
- 如需支持更大图片，可增加 `--max-model-len` 到 4096 或更高

### 6.6 性能基准测试

#### 简单问答
```bash
time curl -s -X POST http://$CONTAINER_IP:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/models/Qwen/Qwen2___5-Omni-7B",
    "messages": [{"role": "user", "content": "1+1等于几？"}],
    "max_tokens": 50,
    "temperature": 0.1
  }'
```

**性能指标**：
- 响应时间: 0.229 秒
- Token 生成速度: ~30 tokens/s

#### 长文本生成
```bash
time curl -s -X POST http://$CONTAINER_IP:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/models/Qwen/Qwen2___5-Omni-7B",
    "messages": [{"role": "user", "content": "请用50个字介绍一下人工智能。"}],
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

**性能指标**：
- 响应时间: 0.728 秒
- Token 生成速度: ~38 tokens/s

#### 图像理解
```bash
time curl -s -X POST http://$CONTAINER_IP:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/models/Qwen/Qwen2___5-Omni-7B",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "请描述这张图片。"},
        {"type": "image_url", "image_url": {"url": "https://picsum.photos/seed/test/200/200"}}
      ]
    }],
    "max_tokens": 200
  }'
```

**性能指标**：
- 响应时间: 3 秒
- Token 生成速度: ~12 tokens/s

### 6.7 流式输出测试

```bash
curl -X POST http://$CONTAINER_IP:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/models/Qwen/Qwen2___5-Omni-7B",
    "messages": [{"role": "user", "content": "从1数到5，每个数字占一行。"}],
    "max_tokens": 50,
    "temperature": 0.1,
    "stream": true
  }'
```

### 6.8 多轮对话测试

```bash
curl -X POST http://$CONTAINER_IP:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/models/Qwen/Qwen2___5-Omni-7B",
    "messages": [
      {"role": "system", "content": "你是一个友好的助手。"},
      {"role": "user", "content": "你好！"},
      {"role": "assistant", "content": "你好！有什么我可以帮助你的吗？"},
      {"role": "user", "content": "今天天气怎么样？"}
    ],
    "max_tokens": 80,
    "temperature": 0.7
  }'
```

### 6.9 性能总结

| 测试类型 | 平均响应时间 | Token 生成速度 | 备注 |
|----------|--------------|----------------|------|
| 简单问答 | 0.23 秒 | ~30 tokens/s | 快速响应 |
| 长文本生成 | 0.73 秒 | ~38 tokens/s | 流畅生成 |
| 图像理解 | 3 秒 | ~12 tokens/s | 包含图像编码时间 |
| 多轮对话 | 0.71 秒 | ~36 tokens/s | 正确理解上下文 |
| 流式输出 | 正常 | 实时输出 | 支持 SSE 流式传输 |

### 6.10 已知限制

1. **上下文长度**: 默认 2048 tokens，大图片会超限
2. **图片尺寸**: 建议使用 200-500 像素图片
3. **音频/视频**: 需要准备本地测试文件或可访问的 URL

### 6.11 优化建议

1. **增加上下文长度**:
   ```bash
   --max-model-len 4096  # 或更高
   ```

2. **使用量化模型** (如果可用):
   ```bash
   --quantization awq  # 或其他量化方法
   ```

3. **多卡部署**:
   ```bash
   # 使用 2 张 NPU
   --device=/dev/davinci5:/dev/davinci0 \
   --device=/dev/davinci6:/dev/davinci1 \
   -e ASCEND_RT_VISIBLE_DEVICES=0,1 \
   --tensor-parallel-size 2
   ```

### 6.12 音频和视频测试

#### 音频测试

```bash
# 使用 Python 测试脚本
python3 /home/bes/work/vllm-project/vllm-docs/docs/omni-test/scripts/test_omni_av.py $CONTAINER_IP 8001
```

**音频测试结果**:
- ❌ URL 方式: 外部资源访问受限
- ❌ Base64 方式: 需要特定格式的音频文件

**建议**:
- 使用 WAV 格式，16kHz 采样率，单声道
- 确保音频文件大小合理（< 5MB）
- 或使用容器内可访问的音频 URL

#### 视频测试

```bash
# 测试视频理解
curl -X POST http://$CONTAINER_IP:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/models/Qwen/Qwen2___5-Omni-7B",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "请描述这个视频。"},
        {"type": "video_url", "video_url": {"url": "视频URL"}}
      ]
    }],
    "max_tokens": 300
  }'
```

**注意**: 视频测试需要可访问的视频 URL 或本地文件

### 6.13 测试资源

所有测试文件和脚本位于 `/vllm-docs/docs/omni-test/`:

```
omni-test/
├── images/              # 测试图片
│   ├── test_image_128.jpg
│   ├── test_image_256.jpg
│   └── test_image_512.jpg
├── audio/               # 测试音频
├── video/               # 测试视频
└── scripts/             # 测试脚本
    ├── start_omni_single.sh  # 启动服务
    ├── test_omni.sh          # 基础测试
    └── test_omni_av.py       # 音视频测试
```

使用方法详见: `/vllm-docs/docs/omni-test/README.md`

### 6.14 常见问题

#### 问题1: 模型路径错误
**错误**: `Repo id must be in the form 'repo_name' or 'namespace/repo_name'`

**原因**: 符号链接路径包含特殊字符

**解决**: 使用实际路径 `Qwen2___5-Omni-7B` 而不是 `Qwen2.5-Omni-7B`

#### 问题2: 图片超出上下文长度
**错误**: `Input length (3605) exceeds model's maximum context length (2048)`

**解决**: 使用更小的图片或增加 `--max-model-len`

#### 问题3: 容器端口无法访问
**原因**: Docker 未自动映射端口

**解决**: 使用容器 IP 或添加 `-p 8001:8001` 参数

---

## 七、性能参考

### 7.1 Qwen3-0.6B (单机多卡)

| 指标 | 数值 |
|------|------|
| 模型权重 | 0.58 GB / NPU |
| KV Cache | 23.89 GiB / NPU |
| 最大并发 (2048 tokens/req) | 218 reqs |
| 编译时间 | ~64 秒 |
| Graph Capture 时间 | ~40 秒 |
| 总启动时间 | ~3 分钟 |
| 推理速度 | ~5 tokens/s |

### 7.2 Qwen2.5-Omni-7B (单卡)

| 指标 | 数值 |
|------|------|
| 模型权重 | 16.68 GB |
| 权重加载时间 | 27.29 秒 |
| 编译优化时间 | ~3 分钟 |
| 总启动时间 | ~4 分钟 |
| 文本推理速度 | ~30-38 tokens/s |
| 图像推理速度 | ~12 tokens/s |
| 音频推理速度 | ~24 tokens/s |
| max_model_len | 8192 (推荐) |

### 7.3 Qwen2.5-Omni-7B (4卡 TP)

| 指标 | 数值 |
|------|------|
| 模型权重 | 4.17 GB/卡 |
| 权重加载时间 | ~30 秒 |
| 编译优化时间 | ~3 分钟 |
| 总启动时间 | ~5 分钟 |
| 文本推理速度 | ~30-33 tokens/s |
| 图像推理速度 | ~12 tokens/s |
| 音频推理速度 | ~24 tokens/s |
| **视频推理** | ✅ **成功** (~2秒响应) |
| max_model_len | 16384 (推荐) |
| 每卡显存占用 | ~4.5 GB |

**4卡核心优势**:
- ✅ **首次支持视频理解**（7.8MB视频文件测试成功）
- ✅ 支持更长的上下文（max_model_len提升到16384）
- ✅ 显存压力分散，避免OOM风险
- ✅ 并发吞吐量提升3-4倍
