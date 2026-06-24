#!/usr/bin/env python3
"""
Qwen3-Omni-30B vs Qwen2.5-Omni-7B 模型对比测试
"""

import requests
import time
import sys

# 配置
IP_30B = "172.17.0.2"
PORT_30B = "8002"
IP_7B = "172.17.0.2"
PORT_7B = "8001"

def test_text_generation(base_url, model_name, model_path):
    """测试文本生成"""
    print(f"\n【{model_name}】文本生成测试")
    print("-" * 60)

    start = time.time()
    response = requests.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": model_path,
            "messages": [{"role": "user", "content": "请详细介绍人工智能的发展历史和未来趋势。"}],
            "max_tokens": 300,
            "temperature": 0.7
        },
        timeout=60
    )
    elapsed = time.time() - start

    if response.status_code == 200:
        data = response.json()
        tokens = data['usage']['completion_tokens']
        speed = tokens / elapsed
        print(f"✅ 成功")
        print(f"生成Token: {tokens}")
        print(f"响应时间: {elapsed:.1f}秒")
        print(f"生成速度: {speed:.1f} tokens/s")
        print(f"响应预览: {data['choices'][0]['message']['content'][:100]}...")
        return speed
    else:
        print(f"❌ 失败: {response.status_code}")
        return 0

def test_image_understanding(base_url, model_name, model_path):
    """测试图像理解"""
    print(f"\n【{model_name}】图像理解测试")
    print("-" * 60)

    import base64
    image_path = "/home/bes/work/vllm-project/vllm-docs/omni-test/images/test_image_256.jpg"
    with open(image_path, 'rb') as f:
        image_base64 = base64.b64encode(f.read()).decode('utf-8')

    start = time.time()
    response = requests.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": model_path,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "请详细描述这张图片的内容。"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                ]
            }],
            "max_tokens": 200
        },
        timeout=60
    )
    elapsed = time.time() - start

    if response.status_code == 200:
        data = response.json()
        tokens = data['usage']['completion_tokens']
        speed = tokens / elapsed
        print(f"✅ 成功")
        print(f"生成Token: {tokens}")
        print(f"响应时间: {elapsed:.1f}秒")
        print(f"响应: {data['choices'][0]['message']['content'][:150]}...")
        return speed
    else:
        print(f"❌ 失败: {response.status_code}")
        return 0

def test_audio_understanding(base_url, model_name, model_path):
    """测试音频理解"""
    print(f"\n【{model_name}】音频理解测试")
    print("-" * 60)

    import base64
    audio_path = "/home/bes/work/vllm-project/vllm-docs/omni-test/audio/test_audio.wav"
    with open(audio_path, 'rb') as f:
        audio_base64 = base64.b64encode(f.read()).decode('utf-8')

    start = time.time()
    response = requests.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": model_path,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "请描述这段音频。"},
                    {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{audio_base64}"}}
                ]
            }],
            "max_tokens": 100
        },
        timeout=60
    )
    elapsed = time.time() - start

    if response.status_code == 200:
        data = response.json()
        tokens = data['usage']['completion_tokens']
        speed = tokens / elapsed
        print(f"✅ 成功")
        print(f"生成Token: {tokens}")
        print(f"响应时间: {elapsed:.1f}秒")
        print(f"响应: {data['choices'][0]['message']['content']}")
        return speed
    else:
        print(f"❌ 失败: {response.status_code}")
        return 0

def main():
    print("=" * 60)
    print("Qwen3-Omni-30B vs Qwen2.5-Omni-7B 对比测试")
    print("=" * 60)

    results = {}

    # 测试 7B 模型
    try:
        print("\n" + "=" * 60)
        print("测试 Qwen2.5-Omni-7B (4卡)")
        print("=" * 60)
        base_url_7b = f"http://{IP_7B}:{PORT_7B}"
        model_path_7b = "/models/Qwen/Qwen2___5-Omni-7B"

        results['7b_text'] = test_text_generation(base_url_7b, "7B", model_path_7b)
        results['7b_image'] = test_image_understanding(base_url_7b, "7B", model_path_7b)
        results['7b_audio'] = test_audio_understanding(base_url_7b, "7B", model_path_7b)
    except Exception as e:
        print(f"7B测试失败: {e}")

    # 测试 30B 模型
    try:
        print("\n" + "=" * 60)
        print("测试 Qwen3-Omni-30B-A3B (4卡)")
        print("=" * 60)
        base_url_30b = f"http://{IP_30B}:{PORT_30B}"
        model_path_30b = "/models/Qwen/Qwen3-Omni-30B-A3B-Instruct"

        results['30b_text'] = test_text_generation(base_url_30b, "30B", model_path_30b)
        results['30b_image'] = test_image_understanding(base_url_30b, "30B", model_path_30b)
        results['30b_audio'] = test_audio_understanding(base_url_30b, "30B", model_path_30b)
    except Exception as e:
        print(f"30B测试失败: {e}")

    # 打印对比结果
    print("\n" + "=" * 60)
    print("性能对比总结")
    print("=" * 60)
    print(f"{'测试项':<20} {'7B (tok/s)':<15} {'30B (tok/s)':<15} {'差异':<15}")
    print("-" * 60)

    if results.get('7b_text') and results.get('30b_text'):
        diff = results['30b_text'] / results['7b_text']
        print(f"{'文本生成':<20} {results['7b_text']:<15.1f} {results['30b_text']:<15.1f} {diff:<15.2f}x")

    if results.get('7b_image') and results.get('30b_image'):
        diff = results['30b_image'] / results['7b_image']
        print(f"{'图像理解':<20} {results['7b_image']:<15.1f} {results['30b_image']:<15.1f} {diff:<15.2f}x")

    if results.get('7b_audio') and results.get('30b_audio'):
        diff = results['30b_audio'] / results['7b_audio']
        print(f"{'音频理解':<20} {results['7b_audio']:<15.1f} {results['30b_audio']:<15.1f} {diff:<15.2f}x")

    print("=" * 60)

if __name__ == "__main__":
    main()
