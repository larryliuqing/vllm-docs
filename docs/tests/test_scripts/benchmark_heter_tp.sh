#!/bin/bash
# PD分离异构TP性能基准测试脚本
# 配置: Prefill (NPU 2-3, TP=2) + Decode (NPU 4-7, TP=4)
# 模型: Qwen2-VL-7B-Instruct
# KV Connector: MooncakeLayerwiseConnector (pd_head_ratio=2)
#
# 用法: bash benchmark_heter_tp.sh [proxy_url]

set -e

# ==========================================
# 配置
# ==========================================
PROXY_URL="${1:-http://127.0.0.1:8000}"
MODEL_PATH="/home/la/work/vllm-project/models/Qwen/Qwen2-VL-7B-Instruct"
OUTPUT_DIR="/home/la/work/vllm-project/vllm-docs/docs/test_reports/benchmark_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${OUTPUT_DIR}/heter_tp_bench_${TIMESTAMP}"
SUMMARY_FILE="${LOG_DIR}/summary.json"
CONTAINER_PREFILL="vllm-p"
CONTAINER_DECODE="vllm-d"

mkdir -p "$LOG_DIR"

echo "============================================="
echo "PD分离异构TP性能基准测试"
echo "============================================="
echo "配置: Prefill (NPU 2-3, TP=2) + Decode (NPU 4-7, TP=4)"
echo "模型: Qwen2-VL-7B-Instruct"
echo "Proxy: $PROXY_URL"
echo "结果目录: $LOG_DIR"
echo "============================================="
echo ""

# ==========================================
# 辅助函数
# ==========================================

# 生成指定token数量的prompt (近似值)
generate_prompt() {
    local target_tokens="$1"
    local sentence="The quick brown fox jumps over the lazy dog. "
    # ~10 tokens per sentence
    local reps=$(( target_tokens / 10 ))
    if [ "$reps" -lt 1 ]; then reps=1; fi
    local result=""
    for ((i=0; i<reps; i++)); do
        result+="$sentence"
    done
    echo "$result"
}

# 发送单个请求并返回JSON响应
send_request() {
    local prompt_text="$1"
    local max_tokens="$2"

    curl -s -w "\n%{http_code}" -X POST "${PROXY_URL}/v1/completions" \
        -H 'Content-Type: application/json' \
        -d "{
            \"model\": \"${MODEL_PATH}\",
            \"prompt\": \"${prompt_text}\",
            \"max_tokens\": ${max_tokens},
            \"temperature\": 0.0
        }"
}

# 提取JSON字段 (使用python3)
extract_json_field() {
    local json="$1"
    local field="$2"
    echo "$json" | python3 -c "import sys,json; d=json.loads(sys.stdin.read().strip()); print(d${field})" 2>/dev/null || echo "0"
}

# 记录测试结果到临时JSON
record_result() {
    local category="$1"
    local name="$2"
    local latencies="$3"  # space-separated ms values
    local prompt_tokens_avg="$4"
    local completion_tokens_avg="$5"

    local json_file="${LOG_DIR}/results.json"

    if [ ! -f "$json_file" ]; then
        echo '{"scenarios":{}, "concurrent":{}, "throughput":{}}' > "$json_file"
    fi

    # Build latencies array
    local latencies_json="["
    local first=true
    for l in $latencies; do
        if [ "$first" = true ]; then
            first=false
        else
            latencies_json+=", "
        fi
        latencies_json+="$l"
    done
    latencies_json+="]"

    python3 -c "
import json
with open('$json_file') as f:
    data = json.load(f)
data['scenarios']['$name'] = {
    'category': '$category',
    'latencies_ms': $latencies_json,
    'avg_prompt_tokens': $prompt_tokens_avg,
    'avg_completion_tokens': $completion_tokens_avg
}
with open('$json_file', 'w') as f:
    json.dump(data, f, indent=2)
"
}

# 运行单场景测试 (5次请求)
run_scenario() {
    local scenario_name="$1"
    local prompt="$2"
    local max_output_tokens="$3"
    local category="${4:-latency}"

    echo ""
    echo "--- 场景: ${scenario_name} ---"
    echo "  Prompt长度(近似): $(echo "$prompt" | wc -c) 字符, Output长度: ${max_output_tokens}"

    local all_latencies=""
    local total_prompt_tokens=0
    local total_completion_tokens=0
    local num_requests=5
    local failed=0

    for i in $(seq 1 $num_requests); do
        start_ms=$(date +%s%3N)
        response=$(send_request "$prompt" $max_output_tokens)
        end_ms=$(date +%s%3N)
        latency_ms=$((end_ms - start_ms))
        all_latencies="$all_latencies $latency_ms"

        http_code=$(echo "$response" | tail -1)
        json_body=$(echo "$response" | sed '$d')

        if [ "$http_code" = "200" ]; then
            prompt_tok=$(extract_json_field "$json_body" "['usage']['prompt_tokens']")
            completion_tok=$(extract_json_field "$json_body" "['usage']['completion_tokens']")
            total_prompt_tokens=$((total_prompt_tokens + prompt_tok))
            total_completion_tokens=$((total_completion_tokens + completion_tok))
            throughput=$(echo "scale=1; $completion_tok * 1000 / $latency_ms" | bc 2>/dev/null || echo "0")
            echo "  请求 #${i}: ${latency_ms}ms (p=${prompt_tok}, c=${completion_tok}, ${throughput} tok/s)"
        else
            echo "  请求 #${i}: ${latency_ms}ms HTTP=${http_code} ✗"
            failed=$((failed + 1))
        fi
        sleep 0.5
    done

    local avg_prompt=0
    local avg_completion=0
    local good=$((num_requests - failed))
    if [ "$good" -gt 0 ]; then
        avg_prompt=$((total_prompt_tokens / good))
        avg_completion=$((total_completion_tokens / good))
    fi

    record_result "$category" "$scenario_name" "$all_latencies" "$avg_prompt" "$avg_completion"

    # 计算平均值并显示
    local total=0
    local count=0
    for l in $all_latencies; do
        total=$((total + l))
        count=$((count + 1))
    done
    if [ "$count" -gt 0 ]; then
        local avg=$((total / count))
        echo "  >>> 平均: ${avg}ms, 平均Prompt: ${avg_prompt}tok, 平均Completion: ${avg_completion}tok"
    fi
}

# 运行并发测试
run_concurrent() {
    local name="$1"
    local num_requests="$2"
    local max_output_tokens="$3"
    local prompt_text="$(generate_prompt 20)"

    echo ""
    echo "--- 并发测试: ${num_requests} 并发, 各${max_output_tokens}输出 ---"

    local req_dir="${LOG_DIR}/concurrent_${num_requests}"
    mkdir -p "$req_dir"

    start_ms=$(date +%s%3N)

    for i in $(seq 0 $((num_requests - 1))); do
        (
            curl -s -w "\n%{http_code}" -X POST "${PROXY_URL}/v1/completions" \
                -H 'Content-Type: application/json' \
                -d "{
                    \"model\": \"${MODEL_PATH}\",
                    \"prompt\": \"${prompt_text}\",
                    \"max_tokens\": ${max_output_tokens},
                    \"temperature\": 0.0
                }" > "${req_dir}/req_${i}.txt" 2>/dev/null
        ) &
    done

    wait
    end_ms=$(date +%s%3N)
    wall_ms=$((end_ms - start_ms))

    # 聚合结果
    local total_completion=0
    local total_prompt=0
    local success_count=0
    local min_latency=9999999
    local max_latency=0

    for f in "${req_dir}/req_"*.txt; do
        if [ -f "$f" ]; then
            http_code=$(tail -1 "$f")
            json_body=$(sed '$d' "$f")

            if [ "$http_code" = "200" ]; then
                prompt_tok=$(extract_json_field "$json_body" "['usage']['prompt_tokens']")
                completion_tok=$(extract_json_field "$json_body" "['usage']['completion_tokens']")
                total_completion=$((total_completion + completion_tok))
                total_prompt=$((total_prompt + prompt_tok))
                success_count=$((success_count + 1))
            fi
        fi
    done

    total_throughput=$(echo "scale=2; $total_completion * 1000 / $wall_ms" | bc)
    avg_latency=$((success_count > 0 ? wall_ms / success_count : 0))
    per_req_throughput=$(echo "scale=2; $total_throughput / $num_requests" | bc 2>/dev/null || echo "0")

    echo "  总耗时: ${wall_ms}ms"
    echo "  成功: ${success_count}/${num_requests}"
    echo "  总吞吐: ${total_throughput} tok/s"
    echo "  单请求等效吞吐: ${per_req_throughput} tok/s/req"

    # 记录到results.json
    local results_json="${LOG_DIR}/results.json"
    python3 -c "
import json
with open('$results_json') as f:
    data = json.load(f)
data['concurrent']['$name'] = {
    'concurrency': $num_requests,
    'wall_time_ms': $wall_ms,
    'success_count': $success_count,
    'total_completion_tokens': $total_completion,
    'total_prompt_tokens': $total_prompt,
    'throughput_tok_s': $total_throughput
}
with open('$results_json', 'w') as f:
    json.dump(data, f, indent=2)
"
}

# ==========================================
# Phase 1: 环境检查
# ==========================================
echo "[Phase 1/7] 环境检查..."

# 检查Proxy是否可用
if curl -s -X GET "${PROXY_URL}/health" > /dev/null 2>&1; then
    echo "  ✓ Proxy服务运行中: ${PROXY_URL}"
else
    echo "  ⚠ Proxy健康检查失败，尝试通过v1/completions验证..."
    # 尝试轻量验证
    test_resp=$(curl -s -w "%{http_code}" -X POST "${PROXY_URL}/v1/completions" \
        -H 'Content-Type: application/json' \
        -d "{\"model\": \"${MODEL_PATH}\", \"prompt\": \"test\", \"max_tokens\": 1}" 2>/dev/null)
    test_code=$(echo "$test_resp" | tail -1)
    if [ "$test_code" = "200" ]; then
        echo "  ✓ Proxy服务响应正常 (HTTP 200)"
    else
        echo "  ✗ Proxy服务无响应 (HTTP ${test_code})，请先部署PD分离服务"
        echo "    执行: bash deploy_pd_separation_heter_tp.sh"
        exit 1
    fi
fi

# 检查容器状态
if docker ps | grep -q "vllm-p"; then
    echo "  ✓ Prefill容器运行中"
else
    echo "  ✗ Prefill容器未运行"
    exit 1
fi

if docker ps | grep -q "vllm-d"; then
    echo "  ✓ Decode容器运行中"
else
    echo "  ✗ Decode容器未运行"
    exit 1
fi

# 保存容器日志快照 (启动阶段)
docker logs "$CONTAINER_PREFILL" 2>&1 > "${LOG_DIR}/docker_logs_prefill.txt" || true
docker logs "$CONTAINER_DECODE" 2>&1 > "${LOG_DIR}/docker_logs_decode.txt" || true

# ==========================================
# Phase 2: 冷启动延迟测试
# ==========================================
echo ""
echo "[Phase 2/7] 冷启动延迟测试..."
echo "  (如果已部署+预热，可能不明显，但仍记录)"

start_ms=$(date +%s%3N)
response=$(send_request "Hello, how are you?" 20)
end_ms=$(date +%s%3N)
cold_latency_ms=$((end_ms - start_ms))

http_code=$(echo "$response" | tail -1)
json_body=$(echo "$response" | sed '$d')

prompt_tok=$(extract_json_field "$json_body" "['usage']['prompt_tokens']")
completion_tok=$(extract_json_field "$json_body" "['usage']['completion_tokens']")

echo "  首次请求延迟: ${cold_latency_ms}ms (p=${prompt_tok}, c=${completion_tok})"
echo "{\"cold_start_ms\": $cold_latency_ms}" > "${LOG_DIR}/cold_start.json"

# ==========================================
# Phase 3: 预热 (虽然部署时已预热,但再确保)
# ==========================================
echo ""
echo "[Phase 3/7] 预热..."
for i in 1 2 3; do
    send_request "Hello, warmup" 5 > /dev/null
    echo "  预热 #${i}"
    sleep 0.5
done

# ==========================================
# Phase 4: 单请求延迟测试矩阵
# ==========================================
echo ""
echo "[Phase 4/7] 单请求延迟测试矩阵..."

# 短提示系列 (20 tokens)
PROMPT_20=$(generate_prompt 20)
run_scenario "short_prompt_20_short_output_20" "$PROMPT_20" 20 "short"
run_scenario "short_prompt_20_medium_output_128" "$PROMPT_20" 128 "short"
run_scenario "short_prompt_20_long_output_256" "$PROMPT_20" 256 "short"

# 中提示系列 (512 tokens)
PROMPT_512=$(generate_prompt 512)
run_scenario "medium_prompt_512_medium_output_128" "$PROMPT_512" 128 "medium"
run_scenario "medium_prompt_512_long_output_256" "$PROMPT_512" 256 "medium"

# 中长提示系列 (1024 tokens)
PROMPT_1024=$(generate_prompt 1024)
run_scenario "midlong_prompt_1024_medium_output_128" "$PROMPT_1024" 128 "long"
run_scenario "midlong_prompt_1024_long_output_256" "$PROMPT_1024" 256 "long"

# 长提示系列 (2048 tokens)
PROMPT_2048=$(generate_prompt 2048)
run_scenario "long_prompt_2048_medium_output_128" "$PROMPT_2048" 128 "long"
run_scenario "long_prompt_2048_long_output_256" "$PROMPT_2048" 256 "long"

# 超长提示 (4096 tokens)
PROMPT_4096=$(generate_prompt 4096)
run_scenario "xlong_prompt_4096_medium_output_128" "$PROMPT_4096" 128 "xlong"
run_scenario "xlong_prompt_4096_long_output_256" "$PROMPT_4096" 256 "xlong"

# ==========================================
# Phase 5: 吞吐量测试 (固定prompt, 变长输出)
# ==========================================
echo ""
echo "[Phase 5/7] 吞吐量测试 (固定prompt=20, 变长输出)..."
run_scenario "throughput_output_50" "$PROMPT_20" 50 "throughput"
run_scenario "throughput_output_100" "$PROMPT_20" 100 "throughput"
run_scenario "throughput_output_200" "$PROMPT_20" 200 "throughput"

# ==========================================
# Phase 6: 并发测试
# ==========================================
echo ""
echo "[Phase 6/7] 并发测试..."
run_concurrent "concurrent_1" 1 50
run_concurrent "concurrent_4" 4 50
run_concurrent "concurrent_8" 8 50
run_concurrent "concurrent_16" 16 50

# ==========================================
# Phase 7: KV传输 & 系统指标收集
# ==========================================
echo ""
echo "[Phase 7/7] 系统指标收集..."

# 再次保存容器日志 (基准测试后)
docker logs "$CONTAINER_PREFILL" 2>&1 > "${LOG_DIR}/docker_logs_prefill_after.txt" || true
docker logs "$CONTAINER_DECODE" 2>&1 > "${LOG_DIR}/docker_logs_decode_after.txt" || true

# NPU状态快照
npu-smi info > "${LOG_DIR}/npu_status.txt" 2>&1 || true

# 从日志提取关键指标
echo "  KV传输指标:"
grep -i "transfer" "${LOG_DIR}/docker_logs_decode_after.txt" 2>/dev/null | grep -i "latency\|ms\|耗时" | head -20 > "${LOG_DIR}/kv_transfer_metrics.txt" 2>/dev/null || echo "  (无KV传输日志)"
echo "  → 已保存至 ${LOG_DIR}/kv_transfer_metrics.txt"

# 提取模型加载时间
echo "  模型加载指标:"
grep -iE "model.*load|权重|weight|加载模型" "${LOG_DIR}/docker_logs_prefill.txt" 2>/dev/null | head -10 > "${LOG_DIR}/model_load_metrics.txt" 2>/dev/null || echo "  (无模型加载日志)"

# 提取KV Cache容量
grep -iE "kv.*cache|KV.*Cache|cache.*tokens" "${LOG_DIR}/docker_logs_prefill.txt" 2>/dev/null | head -5 > "${LOG_DIR}/kvcache_metrics.txt" 2>/dev/null || echo "  (无KV Cache日志)"

# 提取Prefix Cache命中率
grep -iE "prefix.*hit|前缀缓存" "${LOG_DIR}/docker_logs_prefill.txt" 2>/dev/null | head -10 > "${LOG_DIR}/prefix_cache_metrics.txt" 2>/dev/null || echo "  (无Prefix Cache日志)"

# 提取Generation吞吐
grep -iE "generation.*throughput|gen.*tok/s|吞吐" "${LOG_DIR}/docker_logs_decode_after.txt" 2>/dev/null | head -10 > "${LOG_DIR}/generation_metrics.txt" 2>/dev/null || echo "  (无Generation日志)"

echo ""

# ==========================================
# 生成Summary
# ==========================================
echo "============================================="
echo "基准测试完成!"
echo "结果目录: ${LOG_DIR}"
echo "============================================="

# 生成Summary JSON
python3 << PYEOF
import json, os, re

results_file = "${LOG_DIR}/results.json"
if not os.path.exists(results_file):
    print("WARNING: results.json not found")
    exit(0)

with open(results_file) as f:
    data = json.load(f)

# 计算每个场景的统计
summary = {
    "timestamp": "${TIMESTAMP}",
    "config": {
        "prefill_npus": "2,3",
        "prefill_tp": 2,
        "decode_npus": "4,5,6,7",
        "decode_tp": 4,
        "model": "Qwen2-VL-7B-Instruct",
        "kv_connector": "MooncakeLayerwiseConnector",
        "pd_head_ratio": 2,
        "proxy_url": "${PROXY_URL}"
    },
    "scenarios": {},
    "concurrent": {},
    "cold_start_ms": None
}

# Cold start
cold_file = "${LOG_DIR}/cold_start.json"
if os.path.exists(cold_file):
    with open(cold_file) as f:
        summary["cold_start_ms"] = json.load(f).get("cold_start_ms")

# Scenario statistics
for sname, sdata in data.get("scenarios", {}).items():
    lats = sorted(sdata.get("latencies_ms", []))
    n = len(lats)
    if n == 0:
        continue
    mean = sum(lats) / n
    avg_completion = sdata.get("avg_completion_tokens", 0)
    avg_prompt = sdata.get("avg_prompt_tokens", 0)
    decode_tok_s = (avg_completion * 1000 / mean) if mean > 0 and avg_completion > 0 else 0

    summary["scenarios"][sname] = {
        "category": sdata.get("category", ""),
        "mean_latency_ms": round(mean, 1),
        "min_latency_ms": lats[0],
        "max_latency_ms": lats[-1],
        "p50_latency_ms": lats[n // 2],
        "p95_latency_ms": lats[int(n * 0.95)],
        "avg_prompt_tokens": avg_prompt,
        "avg_completion_tokens": avg_completion,
        "decode_throughput_tok_s": round(decode_tok_s, 1),
        "num_samples": n
    }

summary["concurrent"] = data.get("concurrent", {})

with open("${SUMMARY_FILE}", "w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print(f"Summary saved to ${SUMMARY_FILE}")

# 打印汇总表
print("")
print("=" * 80)
print("性能汇总")
print("=" * 80)
print(f"{'场景':<40} {'延迟(ms)':<12} {'吞吐(tok/s)':<12} {'Prompt':<8} {'Output':<8}")
print("-" * 80)

for sname, sinfo in sorted(summary["scenarios"].items()):
    iname = sname[:38]
    print(f"{iname:<40} {sinfo['p50_latency_ms']:<12} {sinfo['decode_throughput_tok_s']:<12} {sinfo['avg_prompt_tokens']:<8} {sinfo['avg_completion_tokens']:<8}")

print("-" * 80)
print("")
for cname, cinfo in summary["concurrent"].items():
    print(f"并发 {cinfo['concurrency']} req: 总耗时={cinfo['wall_time_ms']}ms, 吞吐={cinfo['throughput_tok_s']} tok/s, 成功={cinfo['success_count']}")

PYEOF

echo ""
echo "下一步: 运行 analyze_logs.py 生成详细报告"
echo "  python3 ${SCRIPTS_DIR}/analyze_logs.py ${LOG_DIR}"
echo ""