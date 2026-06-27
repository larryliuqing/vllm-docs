#!/usr/bin/env python3
"""
Quick test script for vLLM CPU version
"""
from vllm import LLM, SamplingParams

def main():
    print("=" * 60)
    print("vLLM CPU Quick Test")
    print("=" * 60)

    # Initialize model with low memory utilization
    print("\n1. Initializing model...")
    llm = LLM(
        model='distilgpt2',
        gpu_memory_utilization=0.3,
        max_model_len=512,  # Smaller for faster testing
        trust_remote_code=False
    )

    # Set sampling parameters
    print("\n2. Setting up sampling parameters...")
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.95,
        max_tokens=20,
        seed=42  # For reproducibility
    )

    # Test inference
    print("\n3. Running inference test...")
    prompts = [
        "Hello, my name is",
        "The capital of France is",
        "AI is"
    ]

    outputs = llm.generate(prompts, sampling_params)

    # Print results
    print("\n4. Results:")
    print("-" * 60)
    for i, output in enumerate(outputs, 1):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"\n{i}. Prompt: {prompt!r}")
        print(f"   Generated: {generated_text!r}")

    print("\n" + "=" * 60)
    print("✓ Test completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()