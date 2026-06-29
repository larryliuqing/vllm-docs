#!/bin/bash
# 使用 Python SDK 直接测试 MTP 不同 num_speculative_tokens 的效果
# 可以直接获取 vLLM 内部的 spec decode metrics
# 依赖：vLLM 服务不需要启动，本脚本直接用 LLM API 调用

set -e

cd /home/la/work/vllm-project/vllm-ascend

# 设置环境变量
source /usr/local/Ascend/cann-9.0.0/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

export LIBRARY_PATH=/usr/local/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
export HCCL_EXEC_TIMEOUT=204
export HCCL_CONNECT_TIMEOUT=120
export HCCL_IF_IP=127.0.0.1
export HCCL_INTRA_ROCE_ENABLE=0
export HCCL_INTRA_PCIE_ENABLE=1
export GLOO_SOCKET_IFNAME="lo"
export TP_SOCKET_IFNAME="lo"
export HCCL_SOCKET_IFNAME="lo"

export VLLM_ASCEND_APPLY_DSV4_PATCH=1
export USE_MULTI_BLOCK_POOL=1
export USE_MULTI_GROUPS_KV_CACHE=1
export HCCL_BUFFSIZE=1024
export VLLM_ASCEND_ENABLE_FUSED_MC2=0
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"
export ASCEND_LAUNCH_BLOCKING=0

MODEL_PATH="/home/la/work/vllm-project/models/DeepSeek-V4-Flash-w4a8-mtp"

python3 << 'PYEOF'
import os
import time
from vllm import LLM, SamplingParams
from vllm.v1.metrics.reader import Counter, Vector

MODEL_PATH = os.environ.get("MODEL_PATH", "/home/la/work/vllm-project/models/DeepSeek-V4-Flash-w4a8-mtp")

PROMPTS = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

# 测试不同 num_speculative_tokens
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

    # 逐条测试
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
            print(f"  耗时: {elapsed:.2f}s,  prompt: {prompt_tok} tok, 生成: {comp_tok} tok, 速度: {tok_per_sec:.2f} tok/s")

    # 获取 Spec Decode 性能指标
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

        # 计算每个位置的 acceptance rate
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

print(f"\n{'='*60}")
print("所有测试完成！")
print(f"{'='*60}")
PYEOF