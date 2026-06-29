#!/bin/bash
cd /vllm-ascend

python3 << 'PYEOF'
import os, time, gc
from vllm import LLM, SamplingParams

MODEL_PATH = "/home/la/work/vllm-project/models/DeepSeek-V4-Flash-w4a8-mtp"

PROMPTS = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

SPEC_TOKENS_LIST = [1, 3, 5]
MAX_TOKENS = 100

for num_spec in SPEC_TOKENS_LIST:
    print(f"\n{'='*60}")
    print(f"测试: num_speculative_tokens = {num_spec}")
    print(f"{'='*60}")

    llm = LLM(
        model=MODEL_PATH,
        tensor_parallel_size=8,
        max_model_len=8192,
        max_num_batched_tokens=10240,
        max_num_seqs=64,
        gpu_memory_utilization=0.95,
        trust_remote_code=True,
        tokenizer_mode="ds_v4",
        enable_prefix_caching=True,
        safetensors_load_strategy="prefetch",
        quantization="ascend",
        block_size=128,
        speculative_config={
            "num_speculative_tokens": num_spec,
            "method": "mtp",
            "enforce_eager": True,
        },
        enforce_eager=False,
        enable_expert_parallel=True,
        disable_log_stats=False,
    )

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=MAX_TOKENS,
        ignore_eos=False,
    )

    for i, prompt in enumerate(PROMPTS):
        print(f"\n--- Prompt {i+1}: \"{prompt}\" ---")
        start_time = time.time()
        outputs = llm.generate([prompt], sampling_params)
        elapsed = time.time() - start_time

        for output in outputs:
            gen_text = output.outputs[0].text
            prompt_tok = len(output.prompt_token_ids)
            comp_tok = len(output.outputs[0].token_ids)
            tok_per_sec = comp_tok / elapsed if elapsed > 0 else 0
            print(f"  生成: {gen_text[:100]}...")
            print(f"  耗时: {elapsed:.2f}s, prompt: {prompt_tok} tok, 生成: {comp_tok} tok, 速度: {tok_per_sec:.2f} tok/s")

    # 获取 Metrics
    print(f"\n--- 推测解码统计 (num_spec={num_spec}) ---")
    try:
        metrics = llm.llm_engine.get_metrics()
        num_drafts = 0
        num_accepted_per_pos = {}
        num_draft_tokens = 0
        num_accepted_tokens = 0
        for metric in metrics:
            if metric.name == "vllm:spec_decode_num_drafts":
                num_drafts = metric.value
                print(f"  spec_decode_num_drafts: {num_drafts}")
            elif metric.name == "vllm:spec_decode_num_accepted_tokens_per_pos":
                for pos, val in enumerate(metric.values):
                    num_accepted_per_pos[pos] = val
                print(f"  spec_decode_num_accepted_tokens_per_pos: {metric.values}")
            elif metric.name == "vllm:spec_decode_num_draft_tokens":
                num_draft_tokens = metric.value
                print(f"  spec_decode_num_draft_tokens: {num_draft_tokens}")
            elif metric.name == "vllm:spec_decode_num_accepted_tokens":
                num_accepted_tokens = metric.value
                print(f"  spec_decode_num_accepted_tokens: {num_accepted_tokens}")

        if num_drafts > 0:
            print(f"\n  --- 每个位置的 Acceptance Rate ---")
            for pos in sorted(num_accepted_per_pos.keys()):
                rate = num_accepted_per_pos[pos] / num_drafts
                print(f"    Position {pos}: {rate:.4f} ({num_accepted_per_pos[pos]}/{num_drafts})")
            overall_accept_rate = num_accepted_tokens / num_draft_tokens if num_draft_tokens > 0 else 0
            print(f"\n  总体 Acceptance Rate: {overall_accept_rate:.4f}")
    except Exception as e:
        print(f"  获取 metrics 失败: {e}")

    del llm
    print(f"\n  等待资源释放...")
    time.sleep(5)
    gc.collect()

print(f"\n{'='*60}")
print("所有测试完成！")
print(f"{'='*60}")
PYEOF
