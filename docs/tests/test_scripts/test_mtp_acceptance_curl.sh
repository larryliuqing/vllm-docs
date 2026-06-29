#!/bin/bash
# 使用 curl 直接测试 MTP 不同 num_speculative_tokens 的效果
# 依赖：vLLM 服务已经启动（手动启动）
# 用法：
#   1. 先启动服务：bash run_deepseek_v4_8cards.sh
#   2. 再运行本脚本：bash test_mtp_acceptance_curl.sh

API="http://127.0.0.1:8000/v1/completions"
MODEL="/home/la/work/vllm-project/models/DeepSeek-V4-Flash-w4a8-mtp"

echo "======================================"
echo "MTP 推测解码测试 - 多 token 预测效果"
echo "======================================"
echo ""

# 测试 prompts
declare -a PROMPTS=(
    "Hello, my name is"
    "The president of the United States is"
    "The capital of France is"
    "The future of AI is"
    "Once upon a time"
)

# 遍历测试的 speculative_tokens 数量
for NUM_SPEC in 1 3 5; do
    echo ""
    echo "========== num_speculative_tokens=$NUM_SPEC =========="
    echo ""

    for i in "${!PROMPTS[@]}"; do
        PROMPT="${PROMPTS[$i]}"
        echo "--- Prompt $((i+1)): \"$PROMPT\" ---"

        # 发送请求，记录耗时
        START_TIME=$(date +%s%N)
        RESPONSE=$(curl -s "$API" \
            -H "Content-Type: application/json" \
            -d "{
                \"model\": \"$MODEL\",
                \"prompt\": \"$PROMPT\",
                \"max_tokens\": 50,
                \"temperature\": 0,
                \"stream\": false
            }")
        END_TIME=$(date +%s%N)
        ELAPSED_MS=$(( (END_TIME - START_TIME) / 1000000 ))

        # 提取生成文本（前150字符）
        TEXT=$(echo "$RESPONSE" | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    text = data['choices'][0]['text']
    prompt_tok = data['usage']['prompt_tokens']
    comp_tok = data['usage']['completion_tokens']
    tokens_per_sec = comp_tok / (float('$ELAPSED_MS') / 1000)
    print(f'生成: {text[:120]}...')
    print(f'耗时: ${ELAPSED_MS}ms, 生成token: {comp_tok}, 速度: {tokens_per_sec:.2f} tok/s')
except Exception as e:
    print(f'解析失败: {e}')
    print(sys.stdin.read()[:200])
" 2>&1)
        echo "$TEXT"
        echo ""
    done
done

echo "======================================"
echo "测试完成"
echo "======================================"