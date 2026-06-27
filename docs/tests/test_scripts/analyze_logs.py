#!/usr/bin/env python3
"""
PD分离异构TP性能测试 - 日志收集和分析脚本

用法:
    python3 analyze_logs.py <results_dir>

功能:
    - 解析 summary.json 中的基准测试结果
    - 解析 docker logs 提取 KV 传输延迟、模型加载时间等
    - 计算统计指标 (mean, median, p95, p99)
    - 生成 markdown 格式的性能分析报告
    - 与之前的 2+2 卡报告进行对比分析
"""

import json
import re
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple


def parse_docker_logs(log_text: str, role: str) -> Dict:
    """解析docker日志中的性能指标"""
    metrics = {
        "role": role,
        "model_load_time_s": 0.0,
        "compilation_time_s": 0.0,
        "graph_capture_time_s": 0.0,
        "weight_memory_gib": 0.0,
        "kv_cache_capacity_tokens": 0,
        "kv_cache_memory_gib": 0.0,
        "max_batch_size": 0,
        "kv_transfer_latencies_ms": [],
        "prefix_cache_hit_rates": [],
        "generation_throughputs": [],
        "prompt_throughputs": [],
    }

    # 模型加载时间
    load_patterns = [
        r'(?:model loading|load model|模型加载)[:\s]*([\d.]+)\s*([s秒])',
        r'model execution is (?:added|ready)[^，。]*?([\d.]+)\s*s',
        r'(?:total|共)[^，。]*?([\d.]+)\s*[s秒][^，。]*?(?:load|加载)',
    ]
    for pat in load_patterns:
        match = re.search(pat, log_text, re.IGNORECASE)
        if match:
            metrics["model_load_time_s"] = float(match.group(1))
            break

    # 编译时间
    compile_patterns = [
        r'(?:compilation|编译)[:\s]*([\d.]+)\s*([s秒])',
        r'compile[^，。]*?([\d.]+)\s*[s秒]',
    ]
    for pat in compile_patterns:
        match = re.search(pat, log_text, re.IGNORECASE)
        if match:
            metrics["compilation_time_s"] = float(match.group(1))
            break

    # 图捕获时间
    graph_patterns = [
        r'(?:graph capture|graph_capture|图捕获)[:\s]*([\d.]+)\s*(?:[s秒])',
    ]
    for pat in graph_patterns:
        match = re.search(pat, log_text, re.IGNORECASE)
        if match:
            metrics["graph_capture_time_s"] = float(match.group(1))
            break

    # 权重内存
    weight_patterns = [
        r'(?:weight|权重)[:\s]*([\d.]+)\s*(?:GiB|GB)',
        r'([\d.]+)\s*GiB[^，。]*?(?:weight|权重)',
    ]
    for pat in weight_patterns:
        match = re.search(pat, log_text, re.IGNORECASE)
        if match:
            metrics["weight_memory_gib"] = float(match.group(1))
            break

    # KV Cache容量
    kvcache_patterns = [
        r'(?:KV cache|kv cache)[^，。]*?(?:capacity|容量|size|大小)[:\s]*([\d,]+)\s*(?:tokens?|tok)',
        r'([\d,]+)\s*(?:tokens?|tok)[^，。]*?(?:KV|kv)',
        r'(?:kv_cache|KV Cache)[:\s]*?([\d.]+)\s*GiB',
    ]
    for pat in kvcache_patterns:
        match = re.search(pat, log_text, re.IGNORECASE)
        if match:
            val = match.group(1).replace(",", "")
            if "." in val:
                metrics["kv_cache_memory_gib"] = float(val)
            else:
                metrics["kv_cache_capacity_tokens"] = int(val)
            break

    # KV传输延迟
    kv_transfer_patterns = [
        r'(?:KV transfer|transfer_engine|MooncakeLayerwise|TransferBlock)[^，。]*?'
        r'(?:latency|time|耗时|用时)[:\s]*([\d.]+)\s*(ms|毫秒)',
        r'transfer[^，。]*?latency[:\s]*([\d.]+)\s*ms',
    ]
    for pat in kv_transfer_patterns:
        matches = re.findall(pat, log_text, re.IGNORECASE)
        for m in matches:
            if isinstance(m, tuple):
                metrics["kv_transfer_latencies_ms"].append(float(m[0]))
            else:
                metrics["kv_transfer_latencies_ms"].append(float(m))

    # Prefix Cache命中率
    prefix_patterns = [
        r'(?:prefix.*?hit|前缀缓存|prefix cache hit)[:\s]*([\d.]+)\s*%',
        r'(?:hit_rate|命中率)[:\s]*([\d.]+)',
    ]
    for pat in prefix_patterns:
        matches = re.findall(pat, log_text, re.IGNORECASE)
        metrics["prefix_cache_hit_rates"] = [float(m) for m in matches if m]

    # Generation吞吐
    gen_patterns = [
        r'(?:generation.*?throughput|gen.*?tok/s|decode.*?吞吐)[:\s]*([\d.]+)',
    ]
    for pat in gen_patterns:
        matches = re.findall(pat, log_text, re.IGNORECASE)
        metrics["generation_throughputs"] = [float(m) for m in matches if m]

    return metrics


def compute_statistics(latencies_ms: List[float]) -> Dict:
    """计算延迟统计数据"""
    if not latencies_ms:
        return {
            "mean_ms": 0, "median_ms": 0, "min_ms": 0, "max_ms": 0,
            "p95_ms": 0, "p99_ms": 0
        }
    sorted_lats = sorted(latencies_ms)
    n = len(sorted_lats)
    return {
        "mean_ms": round(sum(sorted_lats) / n, 1),
        "median_ms": round(sorted_lats[n // 2], 1),
        "min_ms": round(sorted_lats[0], 1),
        "max_ms": round(sorted_lats[-1], 1),
        "p95_ms": round(sorted_lats[int(n * 0.95)], 1),
        "p99_ms": round(sorted_lats[int(n * 0.99)], 1),
        "num_samples": n,
    }


def categorize_scenario(name: str) -> Tuple[str, int, int]:
    """从场景名中提取分类、prompt长度和output长度

    返回: (category, prompt_tokens_est, output_tokens_est)
    """
    prompt_map = {
        "short_prompt_20": 20,
        "medium_prompt_512": 512,
        "midlong_prompt_1024": 1024,
        "long_prompt_2048": 2048,
        "xlong_prompt_4096": 4096,
    }
    output_map = {
        "short_output_20": 20,
        "medium_output_128": 128,
        "long_output_256": 256,
    }
    throughput_map = {
        "throughput_output_50": (20, 50),
        "throughput_output_100": (20, 100),
        "throughput_output_200": (20, 200),
    }

    # 尝试匹配吞吐量测试
    for k, (p, o) in throughput_map.items():
        if k in name:
            return ("throughput", p, o)

    # 尝试匹配常规场景
    prompt_len = 0
    output_len = 0
    for k, v in prompt_map.items():
        if k in name:
            prompt_len = v
            break
    for k, v in output_map.items():
        if k in name:
            output_len = v
            break

    if "short" in name:
        return ("short", prompt_len, output_len)
    elif "medium" in name:
        return ("medium", prompt_len, output_len)
    elif "midlong" in name or "long" in name:
        return ("long", prompt_len, output_len)
    elif "xlong" in name:
        return ("xlong", prompt_len, output_len)
    else:
        return ("other", prompt_len, output_len)


def generate_report(
    summary: Dict,
    prefill_metrics: Dict,
    decode_metrics: Dict,
    results_dir: Path,
) -> str:
    """生成Markdown格式的性能分析报告"""
    lines = []
    now = datetime.now()

    config = summary.get("config", {})

    lines.append("# PD分离异构TP性能测试报告\n")
    lines.append(f"**测试日期**: {now.strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"**测试配置**: Prefill (NPU 2-3, TP=2) + Decode (NPU 4-7, TP=4)")
    lines.append(f"**模型**: Qwen2-VL-7B-Instruct")
    lines.append(f"**KV Connector**: MooncakeLayerwiseConnector (pd_head_ratio=2)")
    lines.append(f"**Proxy URL**: {config.get('proxy_url', 'http://127.0.0.1:8000')}")
    lines.append("")

    # ============================================
    # 1. 测试配置
    # ============================================
    lines.append("---\n")
    lines.append("## 1. 测试配置\n")
    lines.append("### 1.1 硬件配置\n")
    lines.append("| 项目 | 配置 |")
    lines.append("|------|------|")
    lines.append("| **服务器类型** | Atlas 训练服务器 |")
    lines.append("| **NPU型号** | 910B4 (32GB HBM/card) |")
    lines.append("| **PD分离配置** | 2+4卡 (Prefill: NPU 2-3, Decode: NPU 4-7) |")
    lines.append("| **Tensor Parallel** | TP=2 (Prefill), TP=4 (Decode) |")
    lines.append("")

    lines.append("### 1.2 软件配置\n")
    lines.append("| 组件 | 版本/配置 |")
    lines.append("|------|-----------|")
    lines.append("| **Driver** | 25.5.2 |")
    lines.append("| **CANN** | 9.0.0 |")
    lines.append("| **vLLM** | 0.20.2 |")
    lines.append("| **vLLM-Ascend** | v0.20.2rc |")
    lines.append("| **KV Connector** | MooncakeLayerwiseConnector |")
    lines.append("| **pd_head_ratio** | 2 |")
    lines.append("| **KV Buffer Device** | npu |")
    lines.append("| **Max Model Len** | 4096 |")
    lines.append("| **GPU Memory Utilization** | 0.85 |")
    lines.append("| **Max Batched Tokens** | 4096 |")
    lines.append("")

    lines.append("### 1.3 异构TP配置详解\n")
    lines.append("| 节点 | NPU | TP | pd_head_ratio | KV Connector |")
    lines.append("|------|-----|----|---------------|--------------|")
    lines.append("| Prefill | 2,3 | 2 | 2 | MooncakeLayerwiseConnector |")
    lines.append("| Decode | 4,5,6,7 | 4 | 2 | MooncakeLayerwiseConnector |")
    lines.append("")

    # ============================================
    # 2. 启动性能
    # ============================================
    lines.append("---\n")
    lines.append("## 2. 启动性能\n")
    lines.append("| 指标 | Prefill节点 (TP=2) | Decode节点 (TP=4) |")
    lines.append("|------|-------------------|-------------------|")

    for role, metrics in [("Prefill", prefill_metrics), ("Decode", decode_metrics)]:
        lines.append(f"| **模型加载时间** | {metrics.get('model_load_time_s', 'N/A'):.2f}s" if metrics.get('model_load_time_s') else f"| **模型加载时间** | N/A")

    # Build a clean startup table
    p_metrics = prefill_metrics
    d_metrics = decode_metrics
    load_p = f"{p_metrics.get('model_load_time_s', 'N/A'):.2f}s" if p_metrics.get('model_load_time_s') else "N/A"
    load_d = f"{d_metrics.get('model_load_time_s', 'N/A'):.2f}s" if d_metrics.get('model_load_time_s') else "N/A"
    comp_p = f"{p_metrics.get('compilation_time_s', 'N/A'):.2f}s" if p_metrics.get('compilation_time_s') else "N/A"
    comp_d = f"{d_metrics.get('compilation_time_s', 'N/A'):.2f}s" if d_metrics.get('compilation_time_s') else "N/A"
    graph_p = f"{p_metrics.get('graph_capture_time_s', 'N/A'):.2f}s" if p_metrics.get('graph_capture_time_s') else "N/A"
    graph_d = f"{d_metrics.get('graph_capture_time_s', 'N/A'):.2f}s" if d_metrics.get('graph_capture_time_s') else "N/A"
    weight_p = f"{p_metrics.get('weight_memory_gib', 'N/A'):.2f} GiB" if p_metrics.get('weight_memory_gib') else "N/A"
    weight_d = f"{d_metrics.get('weight_memory_gib', 'N/A'):.2f} GiB" if d_metrics.get('weight_memory_gib') else "N/A"

    # 修复表格: 使用新行重建
    lines = lines[:-1]  # 删除上面错误插入的行
    lines.append("| 指标 | Prefill节点 | Decode节点 |")
    lines.append("|------|-------------|------------|")
    lines.append(f"| **模型加载时间** | {load_p} | {load_d} |")
    lines.append(f"| **编译时间** | {comp_p} | {comp_d} |")
    lines.append(f"| **图捕获时间** | {graph_p} | {graph_d} |")
    lines.append(f"| **每卡权重大小** | {weight_p} | {weight_d} |")
    lines.append("")

    # ============================================
    # 3. 单请求延迟测试
    # ============================================
    lines.append("---\n")
    lines.append("## 3. 单请求延迟测试\n")

    scenarios = summary.get("scenarios", {})

    # 3.1 短提示变长输出
    lines.append("### 3.1 短提示 (~20 tokens) → 变长输出\n")
    lines.append("| 输出长度 | 平均延迟(ms) | P50(ms) | P95(ms) | Decode吞吐(tok/s) | Prompt | Output |")
    lines.append("|----------|-------------|---------|---------|-------------------|--------|--------|")

    for key in ["short_prompt_20_short_output_20", "short_prompt_20_medium_output_128", "short_prompt_20_long_output_256"]:
        s = scenarios.get(key, {})
        if s:
            lines.append(
                f"| {s.get('avg_completion_tokens', 'N/A')} tok "
                f"| {s.get('mean_latency_ms', 'N/A')} "
                f"| {s.get('p50_latency_ms', 'N/A')} "
                f"| {s.get('p95_latency_ms', 'N/A')} "
                f"| {s.get('decode_throughput_tok_s', 'N/A')} "
                f"| {s.get('avg_prompt_tokens', 'N/A')} "
                f"| {s.get('avg_completion_tokens', 'N/A')} |"
            )
    lines.append("")

    # 3.2 变长提示固定输出
    lines.append("### 3.2 变长提示 → 固定输出 (~128 tokens)\n")
    lines.append("| 提示长度 | 平均延迟(ms) | P50(ms) | P95(ms) | Decode吞吐(tok/s) |")
    lines.append("|----------|-------------|---------|---------|-------------------|")

    for key in ["short_prompt_20_medium_output_128", "medium_prompt_512_medium_output_128",
                 "long_prompt_2048_medium_output_128", "xlong_prompt_4096_medium_output_128"]:
        s = scenarios.get(key, {})
        if s:
            lines.append(
                f"| {s.get('avg_prompt_tokens', 'N/A')} tok "
                f"| {s.get('mean_latency_ms', 'N/A')} "
                f"| {s.get('p50_latency_ms', 'N/A')} "
                f"| {s.get('p95_latency_ms', 'N/A')} "
                f"| {s.get('decode_throughput_tok_s', 'N/A')} |"
            )
    lines.append("")

    # 3.3 完整场景矩阵
    lines.append("### 3.3 完整场景矩阵\n")
    lines.append("| 场景 | Prompt | Output | 平均延迟(ms) | P50(ms) | P95(ms) | 吞吐(tok/s) |")
    lines.append("|------|--------|--------|-------------|---------|---------|-------------|")

    category_order = ["short", "medium", "long", "xlong", "throughput"]
    for cat in category_order:
        for key, s in sorted(scenarios.items()):
            if s.get("category") != cat:
                continue
            label = key.replace("_", " ")[:35]
            lines.append(
                f"| {label} "
                f"| {s.get('avg_prompt_tokens', 'N/A')} "
                f"| {s.get('avg_completion_tokens', 'N/A')} "
                f"| {s.get('mean_latency_ms', 'N/A')} "
                f"| {s.get('p50_latency_ms', 'N/A')} "
                f"| {s.get('p95_latency_ms', 'N/A')} "
                f"| {s.get('decode_throughput_tok_s', 'N/A')} |"
            )
    lines.append("")

    # ============================================
    # 4. 并发性能测试
    # ============================================
    line_continues = True
    concurrent = summary.get("concurrent", {})
    if concurrent:
        lines.append("---\n")
        lines.append("## 4. 并发性能测试\n")
        lines.append(f"**提示长度**: ~20 tokens, **输出长度**: 50 tokens each\n")
        lines.append("| 并发数 | 总耗时(ms) | 总吞吐(tok/s) | 成功率 |")
        lines.append("|--------|-----------|---------------|--------|")

        for name, cdata in sorted(concurrent.items()):
            lines.append(
                f"| {cdata.get('concurrency', 'N/A')} req "
                f"| {cdata.get('wall_time_ms', 'N/A')} "
                f"| {cdata.get('throughput_tok_s', 'N/A')} "
                f"| {cdata.get('success_count', 0)}/{cdata.get('success_count', 0)} |"
            )
        lines.append("")

    # ============================================
    # 5. KV传输性能分析
    # ============================================
    all_kv_latencies = p_metrics.get(
        "kv_transfer_latencies_ms", []
    ) + d_metrics.get("kv_transfer_latencies_ms", [])
    if all_kv_latencies:
        lines.append("---\n")
        lines.append("## 5. KV传输性能分析\n")
        sorted_kv = sorted(all_kv_latencies)
        n = len(sorted_kv)
        lines.append("| 指标 | 数值 |")
        lines.append("|------|------|")
        lines.append(f"| **样本数** | {n} |")
        lines.append(f"| **平均延迟** | {sum(sorted_kv) / n:.2f} ms |")
        lines.append(f"| **最小延迟** | {sorted_kv[0]:.2f} ms |")
        lines.append(f"| **最大延迟** | {sorted_kv[-1]:.2f} ms |")
        lines.append(f"| **P50延迟** | {sorted_kv[n // 2]:.2f} ms |")
        lines.append("")

    # ============================================
    # 6. 与2+2卡配置对比
    # ============================================
    # 使用之前报告的数据进行对比
    prev_baseline = {
        "short_prompt_20_short_output_20": {"latency_ms": 500, "throughput_tok_s": 40},
        "short_prompt_20_long_output_256": {"latency_ms": 2211, "throughput_tok_s": 45.3},
    }

    lines.append("---\n")
    lines.append("## 6. 与2+2卡配置对比\n")
    lines.append("| 指标 | 2+2 (TP=2, 历史) | 异构TP (2+4, 本次) | 变化 |")
    lines.append("|------|-------------------|-------------------|------|")

    for key, baseline in prev_baseline.items():
        current = scenarios.get(key, {})
        if current:
            prev_lat = baseline["latency_ms"]
            cur_lat = current.get("mean_latency_ms", 0)
            change_pct = ((prev_lat - cur_lat) / prev_lat) * 100 if prev_lat > 0 else 0
            direction = "↑ 提升" if change_pct > 0 else "↓ 下降" if change_pct < 0 else "→ 持平"

            lines.append(
                f"| **{key[:35]}** "
                f"| {prev_lat}ms "
                f"| {cur_lat}ms "
                f"| {direction} {abs(change_pct):.0f}% |"
            )

    # Decode吞吐量对比
    lines.append("| **Decode吞吐(短输出)** | 40 tok/s | ... | ... |")
    lines.append("| **并发吞吐(10 req)** | 73.8 tok/s | ... | ... |")
    lines.append("| **KV传输延迟** | 1.5ms | ... | ... |")
    lines.append("| **冷启动延迟** | 11749ms | ... | ... |")
    lines.append("")

    # ============================================
    # 7. 性能瓶颈分析
    # ============================================
    lines.append("---\n")
    lines.append("## 7. 性能瓶颈分析\n")

    # 分析Prefill瓶颈
    lines.append("### 7.1 Prefill阶段瓶颈分析\n")
    prefill_latencies = []
    for key in ["short_prompt_20_medium_output_128", "medium_prompt_512_medium_output_128",
                 "long_prompt_2048_medium_output_128", "xlong_prompt_4096_medium_output_128"]:
        s = scenarios.get(key, {})
        if s:
            prefill_latencies.append((s.get("avg_prompt_tokens", 0), s.get("mean_latency_ms", 0)))

    if prefill_latencies:
        lines.append("| Prompt长度 | 总延迟(ms) | 每Token延迟(ms) |")
        lines.append("|------------|-----------|-----------------|")
        for p_tok, lat in prefill_latencies:
            if p_tok > 0:
                per_token = round(lat / p_tok, 3)
                lines.append(f"| {p_tok} | {lat} | {per_token} |")
        lines.append("")

        # 分析是否超线性增长
        if len(prefill_latencies) >= 3:
            ratios = []
            for i in range(1, len(prefill_latencies)):
                p_ratio = prefill_latencies[i][0] / prefill_latencies[0][0] if prefill_latencies[0][0] > 0 else 1
                l_ratio = prefill_latencies[i][1] / prefill_latencies[0][1] if prefill_latencies[0][1] > 0 else 1
                ratios.append((p_ratio, l_ratio))
            lines.append("**增长分析**:")
            for i, (p_r, l_r) in enumerate(ratios):
                comparison = "超线性" if l_r > p_r * 1.2 else "亚线性" if l_r < p_r * 0.8 else "近似线性"
                lines.append(f"- Prompt {prefill_latencies[0][0]}→{prefill_latencies[i+1][0]} ({p_r:.1f}x), "
                           f"延迟 {l_r:.1f}x → **{comparison}**")
            lines.append("")

    # 分析Decode瓶颈
    lines.append("### 7.2 Decode阶段瓶颈分析\n")
    # 使用短提示同TP系列分析：prompt=20, output=20/128/256
    decode_analysis = []
    for key in ["short_prompt_20_short_output_20", "short_prompt_20_medium_output_128",
                 "short_prompt_20_long_output_256"]:
        s = scenarios.get(key, {})
        if s:
            decode_analysis.append((s.get("avg_completion_tokens", 0), s.get("mean_latency_ms", 0),
                                    s.get("decode_throughput_tok_s", 0)))

    if decode_analysis:
        lines.append("| Output长度 | 延迟(ms) | 吞吐(tok/s) |")
        lines.append("|------------|---------|-------------|")
        for o_tok, lat, tput in decode_analysis:
            lines.append(f"| {o_tok} | {lat} | {tput} |")
        lines.append("")

    lines.append("### 7.3 并发瓶颈分析\n")
    if concurrent:
        concurrency_data = []
        for name, cdata in sorted(concurrent.items()):
            concurrency_data.append((cdata.get("concurrency", 0), cdata.get("throughput_tok_s", 0),
                                     cdata.get("wall_time_ms", 0)))

        if len(concurrency_data) >= 2:
            lines.append("| 并发数 | 总吞吐(tok/s) | 效率(相对单请求) |")
            lines.append("|--------|---------------|-----------------|")
            base_tput = concurrency_data[0][1] if concurrency_data[0][1] > 0 else 1
            for conc, tput, _ in concurrency_data:
                efficiency = round(tput / (base_tput * max(conc, 1)), 2) if base_tput > 0 else 0
                lines.append(f"| {conc} | {tput} | {efficiency:.2f}x |")
            lines.append("")
            # 判断是否存在并发瓶颈
            last_eff = 0
            for i, (conc, tput, _) in enumerate(concurrency_data):
                if conc > 0 and base_tput > 0:
                    eff = tput / (base_tput * conc)
                    last_eff = eff
            if last_eff < 0.5:
                lines.append("**结论**: 存在显著的并发瓶颈。高并发时效率大幅下降，可能是代理串行化或资源争抢所致。\n")
            elif last_eff < 0.8:
                lines.append("**结论**: 并发性能中等，可能存在轻微瓶颈。\n")
            else:
                lines.append("**结论**: 并发性能良好，扩展性较优。\n")

    lines.append("### 7.4 KV传输瓶颈分析\n")
    lines.append(f"KV传输延迟样本数: {len(all_kv_latencies)}")
    if all_kv_latencies:
        avg_kv = sum(all_kv_latencies) / len(all_kv_latencies)
        if avg_kv < 5:
            lines.append("KV传输延迟极低 (<5ms)，**不是主要瓶颈**。")
        elif avg_kv < 20:
            lines.append("KV传输延迟较低 (5-20ms)，对整体性能影响较小。")
        else:
            lines.append("KV传输延迟较高 (>20ms)，**可能是瓶颈**之一，建议优化。")
    else:
        lines.append("*(未能从日志中提取KV传输延迟数据)*")
    lines.append("")

    # ============================================
    # 8. 优化建议
    # ============================================
    lines.append("---\n")
    lines.append("## 8. 优化建议\n")

    lines.append("### 8.1 短期 (高优先级)\n")
    lines.append("1. **增加Decode节点TP或实例数**")
    lines.append("   - 当前TP=4已较TP=2有提升，可对比本次结果与历史2+2报告")
    lines.append("   - 若Decode仍是瓶颈，考虑更多Decode实例\n")

    lines.append("2. **请求预热常态化**")
    lines.append("   - 部署后自动预热，消除冷启动延迟\n")

    lines.append("3. **增大批处理窗口**")
    lines.append("   - 调整 `--max-num-batched-tokens` 可提升并发处理能力\n")

    lines.append("### 8.2 中期 (1-2周)\n")
    lines.append("1. **缓存感知调度**")
    lines.append("   - Proxy实现 Prefix-hash 感知的 Prefiller选择")
    lines.append("   - 提高Prefix Cache命中率\n")
    lines.append("2. **优化pd_head_ratio**")
    lines.append("   - 分析head_ratio=2带来的额外开销")
    lines.append("   - 测试动态头比配置\n")

    lines.append("### 8.3 长期 (1月+)\n")
    lines.append("1. **KV量化传输**")
    lines.append("   - FP8/INT8量化降低传输带宽")
    lines.append("2. **动态扩缩容**")
    lines.append("   - 基于负载自动增删实例")
    lines.append("3. **多节点PD分离**")
    lines.append("   - 跨节点部署支持更大规模")
    lines.append("")

    # ============================================
    # 总结
    # ============================================
    lines.append("---\n")
    lines.append("## 9. 测试总结\n")
    lines.append("### 9.1 关键数据一览\n")
    lines.append("| 指标 | 数值 |")
    lines.append("|------|------|")

    # 找到代表性场景的数据
    ref_scenario = scenarios.get("short_prompt_20_short_output_20", {})
    if ref_scenario:
        lines.append(f"| 稳态延迟 (20tok) | {ref_scenario.get('p50_latency_ms', 'N/A')}ms |")
        lines.append(f"| Decode吞吐 (单请求) | {ref_scenario.get('decode_throughput_tok_s', 'N/A')} tok/s |")

    # 并发数据
    for cname, cdata in concurrent.items():
        if cdata.get("concurrency", 0) >= 4:
            lines.append(f"| 并发吞吐 ({cdata['concurrency']}req) | {cdata.get('throughput_tok_s', 'N/A')} tok/s |")
            break

    if all_kv_latencies:
        lines.append(f"| KV传输延迟 (平均) | {sum(all_kv_latencies)/len(all_kv_latencies):.1f}ms |")
    lines.append(f"| 冷启动延迟 | {summary.get('cold_start_ms', 'N/A')}ms |")
    lines.append("")

    lines.append("### 9.2 与2+2对比结论\n")
    lines.append("对比2026-06-25的2+2卡测试报告:\n")
    lines.append("| 配置 | Prefill | Decode | 预期变化 |")
    lines.append("|------|---------|--------|----------|")
    lines.append("| 2+2 (历史) | TP=2 | TP=2 | 基线 |")
    lines.append("| 异构TP (本次) | TP=2 | TP=4 | Decode吞吐应提升 |")

    has_improvement = any(
        s.get("decode_throughput_tok_s", 0) > 45
        for s in scenarios.values()
    )
    if has_improvement:
        lines.append("\n**初步结论**: 异构TP配置在Decode吞吐上相比2+2配置有提升，但受限于pd_head_ratio开销和代理串行化。")
        lines.append("建议进一步优化点: 缓存感知代理 + 多Decode实例。")
    else:
        lines.append("\n**初步结论**: 异构TP配置的Decode吞吐优势尚未充分发挥，可能需要进一步分析pd_head_ratio开销。")

    lines.append("\n---")
    lines.append("")
    lines.append(f"**报告生成时间**: {now.strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"**数据目录**: `{results_dir}`")
    lines.append("**测试工具**: benchmark_heter_tp.sh + analyze_logs.py")

    return "\n".join(lines)


def main():
    if len(sys.argv) < 2:
        print("用法: python3 analyze_logs.py <results_dir>")
        sys.exit(1)

    results_dir = Path(sys.argv[1])
    if not results_dir.exists():
        print(f"错误: 目录 '{results_dir}' 不存在")
        sys.exit(1)

    summary_file = results_dir / "summary.json"
    if not summary_file.exists():
        print(f"错误: {summary_file} 不存在，请先运行 benchmark_heter_tp.sh")
        sys.exit(1)

    print(f"分析目录: {results_dir}")
    print(f"加载 summary.json...")

    with open(summary_file) as f:
        summary = json.load(f)

    # 解析容器日志
    prefill_metrics = {}
    decode_metrics = {}

    prefill_log = results_dir / "docker_logs_prefill_after.txt"
    decode_log = results_dir / "docker_logs_decode_after.txt"

    if prefill_log.exists():
        print("解析 Prefill 容器日志...")
        prefill_metrics = parse_docker_logs(prefill_log.read_text(), "prefill")
    else:
        # 尝试使用之前的日志
        prefill_log2 = results_dir / "docker_logs_prefill.txt"
        if prefill_log2.exists():
            prefill_metrics = parse_docker_logs(prefill_log2.read_text(), "prefill")

    if decode_log.exists():
        print("解析 Decode 容器日志...")
        decode_metrics = parse_docker_logs(decode_log.read_text(), "decode")
    else:
        decode_log2 = results_dir / "docker_logs_decode.txt"
        if decode_log2.exists():
            decode_metrics = parse_docker_logs(decode_log2.read_text(), "decode")

    # 计算并显示统计数据
    scenarios = summary.get("scenarios", {})
    print(f"\n共 {len(scenarios)} 个测试场景:")
    for sname, sdata in sorted(scenarios.items()):
        stats = compute_statistics(sdata.get("latencies_ms", []))
        print(f"  {sname}: mean={stats['mean_ms']}ms, p50={stats['median_ms']}ms, "
              f"p95={stats['p95_ms']}ms, throughput={sdata.get('decode_throughput_tok_s', 'N/A')} tok/s")

    # 生成报告
    print("\n生成性能报告...")
    report = generate_report(summary, prefill_metrics, decode_metrics, results_dir)

    report_file = results_dir / "performance_report.md"
    with open(report_file, "w") as f:
        f.write(report)
    print(f"报告已写入: {report_file}")

    print("\n=== 完成 ===")
    print(f"结果目录: {results_dir}")
    print(f"摘要: {summary_file}")
    print(f"报告: {report_file}")


if __name__ == "__main__":
    main()
