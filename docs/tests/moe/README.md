# Qwen2.5-Omni-7B 测试资源

本目录包含 Qwen2.5-Omni-7B 多模态模型的测试资源文件和脚本。

## 目录结构

```
omni-test/
├── images/              # 测试图片
│   ├── test_image_128.jpg   # 128x128 小图
│   ├── test_image_256.jpg   # 256x256 中图
│   └── test_image_512.jpg   # 512x512 大图
├── audio/               # 测试音频
│   ├── test_audio.mp3       # MP3 测试文件
│   └── test_audio.wav       # WAV 测试文件
├── video/               # 测试视频
│   └── (待添加)
└── scripts/             # 测试脚本
    ├── start_omni_single.sh  # 启动服务脚本
    ├── test_omni.sh          # 基础功能测试脚本
    └── test_omni_av.py       # 音视频测试脚本
```

## 使用方法

### 1. 启动服务

```bash
# 使用默认参数启动（max_model_len=8192, port=8001, NPU=davinci5）
bash scripts/start_omni_single.sh

# 自定义参数启动
bash scripts/start_omni_single.sh 16384 8002 6
# 参数1: max_model_len (默认 8192)
# 参数2: port (默认 8001)
# 参数3: NPU设备号 (默认 5)
```

### 2. 运行测试

#### 基础功能测试
```bash
# 获取容器 IP
CONTAINER_IP=$(docker inspect $(docker ps -q --filter "ancestor=vllm-omni:v0.20.2rc" | head -1) | grep '"IPAddress"' | head -1 | awk -F'"' '{print $4}')

# 运行测试
bash scripts/test_omni.sh $CONTAINER_IP 8001
```

#### 音视频测试
```bash
# 安装依赖
pip install requests

# 运行测试
python3 scripts/test_omni_av.py $CONTAINER_IP 8001
```

## 测试内容

### 基础功能测试 (test_omni.sh)
- ✅ 文本对话
- ✅ 图像理解（128x128）
- ✅ 图像理解（256x256）
- ✅ 多轮对话
- ✅ 流式输出

### 音视频测试 (test_omni_av.py)
- ✅ 图像理解（Base64编码）
- ⏳ 音频理解（URL方式）
- ⏳ 音频理解（Base64编码）
- ⏳ 视频理解（URL方式）

## 注意事项

### max_model_len 配置
- 默认 2048 tokens 对于大图片/长视频可能不够
- 建议配置：
  - 纯文本对话: 2048-4096
  - 图像理解: 4096-8192
  - 音频理解: 8192-16384
  - 视频理解: 16384-32768

### 资源要求
- 单卡运行 Qwen2.5-Omni-7B 需要约 17GB 显存
- 建议 NPU 显存利用率设置为 0.85
- 如需支持更长的序列，可以降低显存利用率

### 文件格式支持
- **图像**: JPG, PNG, WebP
- **音频**: MP3, WAV, FLAC
- **视频**: MP4, WebM, AVI

## 故障排查

### 端口无法访问
```bash
# 查看容器 IP
docker inspect <container_id> | grep IPAddress

# 或添加端口映射
docker run -p 8001:8001 ...
```

### 图片超出上下文长度
```
错误: Input length exceeds model's maximum context length
解决: 增加 max_model_len 或使用更小的图片
```

### 音视频文件加载失败
```
检查: URL 是否可访问
检查: 文件格式是否支持
检查: max_model_len 是否足够
```

## 参考文档

- [vllm_昇腾NPU测试指南.md](../../../vllm_昇腾NPU测试指南.md)
- [omni_test_report.md](../../../omni_test_report.md)
