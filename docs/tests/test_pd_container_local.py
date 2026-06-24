#!/usr/bin/env python3
"""
PD分离测试 - 容器内使用本地Qwen模型
基于官方 offline_disaggregated_prefill_npu.py，修改模型路径
"""
import multiprocessing as mp
import os
import time
from multiprocessing import Event, Process

os.environ["VLLM_USE_MODELSCOPE"] = "False"  # 使用本地模型
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


def clean_up():
    import gc
    import torch
    from vllm.distributed.parallel_state import destroy_distributed_environment, destroy_model_parallel

    try:
        destroy_model_parallel()
        destroy_distributed_environment()
        gc.collect()
        torch.npu.empty_cache()
    except:
        pass


def run_prefill(prefill_done, process_close):
    """Prefill进程"""
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "0"  # 逻辑ID 0 = 物理NPU 4

    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig

    prompts = [
        "Hello, how are you today?",
        "Hi, what is your name?",
    ]
    sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=1)

    ktc = KVTransferConfig(
        kv_connector="MooncakeConnectorV1",
        kv_role="kv_producer",
        kv_port="30000",
        engine_id="0",
        kv_connector_extra_config={"prefill": {"dp_size": 1, "tp_size": 1}, "decode": {"dp_size": 1, "tp_size": 1}},
    )

    print("[Prefill] Initializing LLM with local Qwen model...")
    llm = LLM(
        model="/models/Qwen/Qwen3-0.6B",  # 使用本地模型
        kv_transfer_config=ktc,
        max_model_len=2000,
        gpu_memory_utilization=0.8,
        tensor_parallel_size=1,
        trust_remote_code=True,
        enforce_eager=True,
    )

    print("[Prefill] Running generation...")
    llm.generate(prompts, sampling_params)
    print("[Prefill] ✓ Prefill finished")
    prefill_done.set()

    try:
        while not process_close.is_set():
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        print("[Prefill] Cleaning up...")
        del llm
        clean_up()


def run_decode(prefill_done):
    """Decode进程"""
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "1"  # 逻辑ID 1 = 物理NPU 5

    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig

    prompts = [
        "Hello, how are you today?",
        "Hi, what is your name?",
    ]
    sampling_params = SamplingParams(temperature=0, top_p=0.95)

    ktc = KVTransferConfig(
        kv_connector="MooncakeConnectorV1",
        kv_role="kv_consumer",
        kv_port="30100",
        engine_id="1",
        kv_connector_extra_config={"prefill": {"dp_size": 1, "tp_size": 1}, "decode": {"dp_size": 1, "tp_size": 1}},
    )

    print("[Decode] Initializing LLM with local Qwen model...")
    llm = LLM(
        model="/models/Qwen/Qwen3-0.6B",  # 使用本地模型
        kv_transfer_config=ktc,
        max_model_len=2000,
        gpu_memory_utilization=0.8,
        tensor_parallel_size=1,
        trust_remote_code=True,
        enforce_eager=True,
    )

    print("[Decode] Waiting for prefill to finish...")
    prefill_done.wait()

    print("[Decode] Starting generation...")
    outputs = llm.generate(prompts, sampling_params)

    print("\n" + "=" * 70)
    print("[Decode] Generation results:")
    print("=" * 70)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}")
        print(f"Generated: {generated_text!r}")
        print("-" * 70)

    del llm
    clean_up()


if __name__ == "__main__":
    mp.get_context("spawn")

    print("=" * 70)
    print("PD分离测试 - 容器环境（本地Qwen模型）")
    print("=" * 70)
    print("模型: /models/Qwen/Qwen3-0.6B")
    print("Prefill: 逻辑设备0 (物理NPU 4)")
    print("Decode: 逻辑设备1 (物理NPU 5)")
    print("=" * 70)

    prefill_done = Event()
    process_close = Event()

    prefill_process = Process(
        target=run_prefill,
        args=(prefill_done, process_close),
    )
    decode_process = Process(target=run_decode, args=(prefill_done,))

    prefill_process.start()
    decode_process.start()

    decode_process.join()

    process_close.set()
    prefill_process.join()
    prefill_process.terminate()

    print("\n" + "=" * 70)
    print("✓ All process done!")
    print("=" * 70)
