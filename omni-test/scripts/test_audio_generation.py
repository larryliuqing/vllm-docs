#!/usr/bin/env python3
"""
Qwen3-Omni-30B-A3B 音频生成测试脚本
使用 vLLM 直接生成音频
"""

import os
os.environ['VLLM_USE_V1'] = '0'  # 必须关闭 v1 引擎

import torch
from vllm import LLM, SamplingParams
from transformers import Qwen3OmniMoeProcessor
from qwen_omni_utils import process_mm_info
import soundfile as sf

def generate_audio():
    """生成音频"""
    MODEL_PATH = "/models/Qwen/Qwen3-Omni-30B-A3B-Instruct"

    print("========================================")
    print("Qwen3-Omni-30B-A3B 音频生成测试")
    print("========================================")
    print()

    # 初始化 vLLM 引擎
    print("初始化 vLLM 引擎...")
    llm = LLM(
        model=MODEL_PATH,
        trust_remote_code=True,
        gpu_memory_utilization=0.90,
        tensor_parallel_size=4,
        limit_mm_per_prompt={'image': 3, 'video': 3, 'audio': 3},
        max_num_seqs=8,
        max_model_len=8192,
        seed=1234,
    )

    # 采样参数
    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        max_tokens=512,
    )

    # 加载处理器
    print("加载处理器...")
    processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL_PATH)

    # 测试1: 纯文本转语音
    print("\n【测试1】文本转语音：你好，我是通义千问多模态模型。")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "请用语音说：你好，我是通义千问多模态模型。"}
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = {
        'prompt': text,
        'multi_modal_data': {},
        "mm_processor_kwargs": {
            "use_audio_in_video": True,
        },
    }

    # 生成
    print("生成中...")
    outputs = llm.generate([inputs], sampling_params=sampling_params)

    # 保存音频
    if outputs and outputs[0].outputs:
        output = outputs[0].outputs[0]
        print(f"文本输出: {output.text[:100]}...")

        # 检查是否有音频数据
        if hasattr(output, 'audio') and output.audio is not None:
            audio_path = "/test-data/audio/generated_speech_1.wav"
            sf.write(
                audio_path,
                output.audio.reshape(-1).detach().cpu().numpy(),
                samplerate=24000,
            )
            print(f"✅ 音频已保存: {audio_path}")
        else:
            print("⚠️  未检测到音频输出")

    # 测试2: 更长的语音
    print("\n【测试2】文本转语音：更长的内容")
    messages2 = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "请用语音读出：人工智能正在改变我们的生活方式，从智能手机到自动驾驶，从医疗诊断到金融分析，AI无处不在。"}
            ],
        }
    ]

    text2 = processor.apply_chat_template(
        messages2,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs2 = {
        'prompt': text2,
        'multi_modal_data': {},
        "mm_processor_kwargs": {},
    }

    outputs2 = llm.generate([inputs2], sampling_params=sampling_params)

    if outputs2 and outputs2[0].outputs:
        output2 = outputs2[0].outputs[0]
        print(f"文本输出: {output2.text[:100]}...")

        if hasattr(output2, 'audio') and output2.audio is not None:
            audio_path2 = "/test-data/audio/generated_speech_2.wav"
            sf.write(
                audio_path2,
                output2.audio.reshape(-1).detach().cpu().numpy(),
                samplerate=24000,
            )
            print(f"✅ 音频已保存: {audio_path2}")
        else:
            print("⚠️  未检测到音频输出")

    print("\n========================================")
    print("测试完成")
    print("========================================")

if __name__ == "__main__":
    try:
        generate_audio()
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
