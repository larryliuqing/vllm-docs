#!/usr/bin/env python3
"""
Qwen2.5-Omni-7B 音频和视频测试脚本（更新版）
使用方法: python3 test_omni_av_local.py [container_ip] [port]
"""

import base64
import sys
import requests
from pathlib import Path

# 配置
CONTAINER_IP = sys.argv[1] if len(sys.argv) > 1 else "172.17.0.2"
PORT = sys.argv[2] if len(sys.argv) > 2 else "8001"
BASE_URL = f"http://{CONTAINER_IP}:{PORT}"
TEST_DATA_DIR = Path("/home/bes/work/vllm-project/vllm-docs/omni-test")

def encode_file_to_base64(file_path):
    """将文件编码为 base64"""
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def test_audio_local():
    """测试音频理解（使用本地文件）"""
    print("【测试】音频理解（本地WAV文件）")
    print("-" * 50)

    audio_path = TEST_DATA_DIR / "audio" / "test_audio.wav"
    if not audio_path.exists():
        print(f"音频文件不存在: {audio_path}")
        return

    print(f"文件大小: {audio_path.stat().st_size / 1024:.1f} KB")
    audio_base64 = encode_file_to_base64(audio_path)

    response = requests.post(
        f"{BASE_URL}/v1/chat/completions",
        json={
            "model": "/models/Qwen/Qwen2___5-Omni-7B",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "请描述这段音频的内容。"},
                    {
                        "type": "audio_url",
                        "audio_url": {
                            "url": f"data:audio/wav;base64,{audio_base64}"
                        }
                    }
                ]
            }],
            "max_tokens": 200
        },
        timeout=60
    )

    if response.status_code == 200:
        data = response.json()
        print(f"✅ 成功!")
        print(f"响应: {data['choices'][0]['message']['content']}")
        print(f"Token使用: {data['usage']}")
    else:
        print(f"❌ 错误: {response.status_code}")
        print(response.text[:500])
    print()

def test_video_local():
    """测试视频理解（使用本地文件）"""
    print("【测试】视频理解（本地AVI文件）")
    print("-" * 50)

    video_path = TEST_DATA_DIR / "video" / "test_video.avi"
    if not video_path.exists():
        print(f"视频文件不存在: {video_path}")
        return

    print(f"文件大小: {video_path.stat().st_size / 1024 / 1024:.1f} MB")
    video_base64 = encode_file_to_base64(video_path)

    response = requests.post(
        f"{BASE_URL}/v1/chat/completions",
        json={
            "model": "/models/Qwen/Qwen2___5-Omni-7B",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "请描述这个视频的内容。"},
                    {
                        "type": "video_url",
                        "video_url": {
                            "url": f"data:video/avi;base64,{video_base64}"
                        }
                    }
                ]
            }],
            "max_tokens": 300
        },
        timeout=120
    )

    if response.status_code == 200:
        data = response.json()
        print(f"✅ 成功!")
        print(f"响应: {data['choices'][0]['message']['content']}")
        print(f"Token使用: {data['usage']}")
    else:
        print(f"❌ 错误: {response.status_code}")
        print(response.text[:500])
    print()

def test_image_local():
    """测试图像理解（使用本地文件）"""
    print("【测试】图像理解（本地JPG文件）")
    print("-" * 50)

    image_path = TEST_DATA_DIR / "images" / "test_image_256.jpg"
    if not image_path.exists():
        print(f"图片文件不存在: {image_path}")
        return

    print(f"文件大小: {image_path.stat().st_size / 1024:.1f} KB")
    image_base64 = encode_file_to_base64(image_path)

    response = requests.post(
        f"{BASE_URL}/v1/chat/completions",
        json={
            "model": "/models/Qwen/Qwen2___5-Omni-7B",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "请描述这张图片。"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    }
                ]
            }],
            "max_tokens": 200
        },
        timeout=30
    )

    if response.status_code == 200:
        data = response.json()
        print(f"✅ 成功!")
        print(f"响应: {data['choices'][0]['message']['content']}")
        print(f"Token使用: {data['usage']}")
    else:
        print(f"❌ 错误: {response.status_code}")
        print(response.text[:500])
    print()

def main():
    print("=" * 50)
    print("Qwen2.5-Omni-7B 音视频测试（本地文件）")
    print("=" * 50)
    print(f"服务地址: {BASE_URL}")
    print("=" * 50)
    print()

    # 测试图像
    test_image_local()

    # 测试音频
    test_audio_local()

    # 测试视频
    test_video_local()

    print("=" * 50)
    print("测试完成")
    print("=" * 50)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
