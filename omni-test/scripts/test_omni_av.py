#!/usr/bin/env python3
"""
Qwen2.5-Omni-7B 音频和视频测试脚本
使用方法: python3 test_omni_av.py [container_ip] [port]
"""

import base64
import sys
import requests
from pathlib import Path

# 配置
CONTAINER_IP = sys.argv[1] if len(sys.argv) > 1 else "172.17.0.4"
PORT = sys.argv[2] if len(sys.argv) > 2 else "8001"
BASE_URL = f"http://{CONTAINER_IP}:{PORT}"
TEST_DATA_DIR = Path("/home/bes/work/vllm-project/vllm-docs/omni-test")

def encode_file_to_base64(file_path):
    """将文件编码为 base64"""
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def test_audio_with_url():
    """测试音频理解（使用URL）"""
    print("【测试】音频理解（URL方式）")
    print("-" * 50)

    # 使用公开的音频文件 - 尝试不同的源
    audio_urls = [
        "https://www.soundhelix.com/examples/mp3/SoundHelix-Song-1.mp3",
        "https://file-examples.com/storage/fe8c7eef0b67998983919d5/2017/11/file_example_MP3_700KB.mp3"
    ]

    for audio_url in audio_urls:
        try:
            response = requests.post(
                f"{BASE_URL}/v1/chat/completions",
                json={
                    "model": "/models/Qwen/Qwen2___5-Omni-7B",
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "这段音频的内容是什么？"},
                            {"type": "audio_url", "audio_url": {"url": audio_url}}
                        ]
                    }],
                    "max_tokens": 200
                },
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                print(f"响应: {data['choices'][0]['message']['content']}")
                print(f"Token使用: {data['usage']}")
                return  # 成功则退出
            else:
                print(f"URL失败 ({response.status_code}): {audio_url[:50]}...")
        except Exception as e:
            print(f"尝试失败: {str(e)[:50]}")

    print("所有音频URL测试失败")
    print()

def test_audio_with_base64():
    """测试音频理解（使用base64编码）"""
    print("【测试】音频理解（Base64方式）")
    print("-" * 50)

    audio_path = TEST_DATA_DIR / "audio" / "test_audio.mp3"
    if not audio_path.exists():
        print(f"音频文件不存在: {audio_path}")
        print("跳过此测试")
        print()
        return

    audio_base64 = encode_file_to_base64(audio_path)

    response = requests.post(
        f"{BASE_URL}/v1/chat/completions",
        json={
            "model": "/models/Qwen/Qwen2___5-Omni-7B",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "请描述这段音频。"},
                    {
                        "type": "audio_url",
                        "audio_url": {
                            "url": f"data:audio/mp3;base64,{audio_base64}"
                        }
                    }
                ]
            }],
            "max_tokens": 200
        }
    )

    if response.status_code == 200:
        data = response.json()
        print(f"响应: {data['choices'][0]['message']['content']}")
        print(f"Token使用: {data['usage']}")
    else:
        print(f"错误: {response.status_code}")
        print(response.text)
    print()

def test_video_with_url():
    """测试视频理解（使用URL）"""
    print("【测试】视频理解（URL方式）")
    print("-" * 50)

    # 使用公开的视频文件
    video_url = "https://file-examples.com/storage/fe8c7eef0b67998983919d5/2017/04/file_example_MP4_480_1_5MG.mp4"

    response = requests.post(
        f"{BASE_URL}/v1/chat/completions",
        json={
            "model": "/models/Qwen/Qwen2___5-Omni-7B",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "请描述这个视频的内容。"},
                    {"type": "video_url", "video_url": {"url": video_url}}
                ]
            }],
            "max_tokens": 300
        }
    )

    if response.status_code == 200:
        data = response.json()
        print(f"响应: {data['choices'][0]['message']['content']}")
        print(f"Token使用: {data['usage']}")
    else:
        print(f"错误: {response.status_code}")
        print(response.text)
    print()

def test_image_with_base64():
    """测试图像理解（使用base64编码的本地图片）"""
    print("【测试】图像理解（Base64方式）")
    print("-" * 50)

    image_path = TEST_DATA_DIR / "images" / "test_image_256.jpg"
    if not image_path.exists():
        print(f"图片文件不存在: {image_path}")
        print("跳过此测试")
        print()
        return

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
        }
    )

    if response.status_code == 200:
        data = response.json()
        print(f"响应: {data['choices'][0]['message']['content']}")
        print(f"Token使用: {data['usage']}")
    else:
        print(f"错误: {response.status_code}")
        print(response.text)
    print()

def main():
    print("=" * 50)
    print("Qwen2.5-Omni-7B 音视频测试")
    print("=" * 50)
    print(f"服务地址: {BASE_URL}")
    print("=" * 50)
    print()

    # 测试图像（base64）
    test_image_with_base64()

    # 测试音频（URL）
    test_audio_with_url()

    # 测试音频（base64）
    test_audio_with_base64()

    # 测试视频（URL）
    test_video_with_url()

    print("=" * 50)
    print("测试完成")
    print("=" * 50)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)
