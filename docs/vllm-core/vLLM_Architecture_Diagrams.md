# vLLM 完整架构图

## 1. 整体架构 - 请求处理流程

```
                           ┌──────────────────────────┐
                           │  vLLM Server (FastAPI)    │
                           │  POST /v1/chat/completions│
                           │  POST /v1/completions     │
                           └─────────┬────────────────┘
                                     │
                           ┌─────────▼────────────────┐
                           │    AsyncLLM / LLM        │
                           │    (vllm/entrypoints/)   │
                           └─────────┬────────────────┘
                                     │ EngineCoreRequest
                           ┌─────────▼────────────────┐
                           │  EngineCoreClient        │
                           │  - InprocClient (同步)    │
                           │  - SyncMPClient (多进程)  │
                           │  - AsyncMPClient (异步)   │
                           └─────────┬────────────────┘
                                     │ ZMQ/In-memory Queue
                           ┌─────────▼────────────────┐
                           │      EngineCore          │
                           │  (vllm/v1/engine/core.py)│
                           │                          │
                           │  ┌────────────────────┐  │
                           │  │    Scheduler       │  │
                           │  │  - Prefill 调度    │  │
                           │  │  - Decode 调度     │  │
                           │  │  - KV Cache 管理   │  │
                           │  │  - 前缀缓存        │  │
                           │  └───────┬────────────┘  │
                           │          │                │
                           │  ┌───────▼────────────┐  │
                           │  │  ModelExecutor     │  │
                           │  │  - UniprocExecutor │  │
                           │  │  - MultiprocExec.  │  │
                           │  │  - RayExecutor     │  │
                           │  └───────┬────────────┘  │
                           │          │                │
                           │  ┌───────▼────────────┐  │
                           │  │  Worker           │  │
                           │  │  - GPUModelRunner │  │
                           │  │  - GPUWorker      │  │
                           │  └───────┬────────────┘  │
                           └──────────┼────────────────┘
                                      │
                           ┌──────────▼──────────────┐
                           │    GPUModelRunner       │
                           │  - Input Preparation    │
                           │  - Model Forward        │
                           │  - Sampling             │
                           └─────────────────────────┘
```

---

## 2. 模型加载流程 - 详细时序图

```
  用户       EngineArgs   EngineCore   ModelRegistry  DefaultModel  AutoWeights  Weight
                                                  Loader      Loader     Utils
  │             │            │              │             │            │         │
  │  LLM(...)-→ │            │              │             │            │         │
  │             │ create_engine_config()     │             │            │         │
  │             │─────────→  │              │             │            │         │
  │             │            │  init()      │             │            │         │
  │             │            │────→ resolve_model_class() │            │         │
  │             │            │              │  resolve() │            │         │
  │             │            │←───────────────────────── │            │         │
  │             │            │────→ get_model_loader() → │            │         │
  │             │            │                           │            │         │
  │             │            │────→ load_model() ───────→│            │         │
  │             │            │                           │ prepare_   │         │
  │             │            │                           │ weights()  │         │
  │             │            │                           │─────────────────────→│
  │             │            │                           │←──────────│ download │
  │             │            │                           │           │from HF   │
  │             │            │                           │           │          │
  │             │            │                           │ get_weights_iterator  │
  │             │            │                           │─────────────────────→│
  │             │            │                           │←──────────│ yield    │
  │             │            │                           │           │(name,    │
  │             │            │                           │           │ tensor)  │
  │             │            │                           │                      │
  │             │            │                           │ ModelClass()         │
  │             │            │                           │────────→ model       │
  │             │            │                           │                      │
  │             │            │                           │ load_weights(iter)   │
  │             │            │                           │──────────────────→   │
  │             │            │                           │           │ _load_   │
  │             │            │                           │           │ module() │
  │             │            │                           │           │          │
  │             │            │                           │           │ 递归加载 │
  │             │            │                           │           │ ├ model  │
  │             │            │                           │           │ ├ layers │
  │             │            │                           │           │ ├ self_  │
  │             │            │                           │           │ │ attn   │
  │             │            │                           │           │ │ ├ qkv  │
  │             │            │                           │           │ │ │ proj │
  │             │            │                           │           │ │ ├ q_   │
  │             │            │                           │           │ │ │ norm │
  │             │            │                           │           │ │ ├ o_   │
  │             │            │                           │           │ │ │ proj │
  │             │            │                           │           │ ├ mlp   │
  │             │            │                           │           │ └ norm  │
  │             │            │                           │←──────────┘         │
  │             │            │                           │                      │
  │             │            │                           │ return model          │
  │             │            │←──────────────────────────┘                      │
  │             │            │                                                  │
  │             │            │ init_kv_caches()                                 │
  │             │            │   ├ profile memory                               │
  │             │            │   ├ allocate KV blocks                           │
  │             │            │   └ warmup model                                 │
  │             │            │                                                  │
  │  ←──────────┘            │                                                  │
  │  return LLM              │                                                  │
```

---

## 3. 推理执行流程 - 单步详细图

```
Scheduler.schedule()
         │
         ▼
┌──────────────────────┐
│ SchedulerOutput      │
│ - scheduled_new_reqs │  ← 新请求 (prefill)
│ - scheduled_cached   │  ← KV cache hit 请求
│ - scheduled_running  │  ← decode 请求
│ - num_tokens_per_req │
│ - blocks_to_swap_in  │
│ - blocks_to_swap_out │
│ - blocks_to_copy     │
└──────┬───────────────┘
       │
       ▼
GPUModelRunner.execute_model(scheduler_output)
       │
       ├────────────────────────────────────────────┐
       │ 1. Prepare Input Batch                     │
       │                                            │
       │ input_ids:    [seq_0, seq_1, ..., seq_N]   │
       │ positions:    [R0: [0..L0],                │
       │                R1: [p, p+L1-1],            │
       │                ...]                        │
       │ token_types:  [types per token]            │
       │                                            │
       │ ├── Prefill requests:                      │
       │ │   - Full sequence tokens                 │
       │ │   - Position IDs: 0..L-1                 │
       │ │                                          │
       │ └── Decode requests:                       │
       │     - Single new token                     │
       │     - Position: L                          │
       └────────────┬───────────────────────────────┘
                    │
       ┌────────────▼───────────────────────────────┐
       │ 2. Prepare Sampling Metadata               │
       │                                            │
       │ - temperature, top_p, top_k per request    │
       │ - logit_bias / penalty                      │
       │ - max_tokens / stop_strings                │
       │ - bad_words / stop_token_ids               │
       │ - grammar constraints                      │
       └────────────┬───────────────────────────────┘
                    │
       ┌────────────▼───────────────────────────────┐
       │ 3. Prepare Attention Metadata              │
       │                                            │
       │ - Block tables (PagedAttention)            │
       │ - Slot mapping                              │
       │ - Sequence lengths                          │
       │ - Query/Context lengths                     │
       │ - Block sizes                               │
       └────────────┬───────────────────────────────┘
                    │
       ┌────────────▼───────────────────────────────┐
       │ 4. Model Forward                           │
       │                                            │
       │ ┌──────────────────────────────────────┐   │
       │ │ Embedding (Embed_tokens)             │   │
       │ │ input_ids → hidden_states            │   │
       │ │ [B, seq] → [B, seq, hidden_size]     │   │
       │ └───────────────┬──────────────────────┘   │
       │                 │                           │
       │ ┌───────────────▼──────────────────────┐   │
       │ │ Layer 0: Qwen3DecoderLayer           │   │
       │ │ ┌────────────────────────────────┐   │   │
       │ │ │ Input LayerNorm (RMSNorm)      │   │   │
       │ │ │ A = LN(h)                     │   │   │
       │ │ └───────────────┬────────────────┘   │   │
       │ │                 │                     │   │
       │ │ ┌───────────────▼────────────────┐   │   │
       │ │ │ Qwen3Attention                 │   │   │
       │ │ │ ┌──────────────────────────┐   │   │   │
       │ │ │ │ QKV Projection           │   │   │   │
       │ │ │ │ [B,T,H] → [B,T,heads*3] │   │   │   │
       │ │ │ └──────────┬───────────────┘   │   │   │
       │ │ │            │                    │   │   │
       │ │ │ ┌──────────▼───────────────┐   │   │   │
       │ │ │ │ QK-Norm (Qwen3 特有)     │   │   │   │
       │ │ │ │ RMSNorm per head         │   │   │   │
       │ │ │ └──────────┬───────────────┘   │   │   │
       │ │ │            │                    │   │   │
       │ │ │ ┌──────────▼───────────────┐   │   │   │
       │ │ │ │ Rotary Embedding         │   │   │   │
       │ │ │ │ RoPE on Q and K          │   │   │   │
       │ │ │ └──────────┬───────────────┘   │   │   │
       │ │ │            │                    │   │   │
       │ │ │ ┌──────────▼───────────────┐   │   │   │
       │ │ │ │ PagedAttention           │   │   │   │
       │ │ │ │ - FlashAttention kernel  │   │   │   │
       │ │ │ │ - KV Cache read/write    │   │   │   │
       │ │ │ │ - Softmax + Matmul       │   │   │   │
       │ │ │ └──────────┬───────────────┘   │   │   │
       │ │ │            │                    │   │   │
       │ │ │ ┌──────────▼───────────────┐   │   │   │
       │ │ │ │ Output Projection        │   │   │   │
       │ │ │ │ [B,T,heads] → [B,T,H]   │   │   │   │
       │ │ │ └──────────────────────────┘   │   │   │
       │ │ └────────────────┬───────────────┘   │   │
       │ │                  │                    │   │
       │ │ ┌────────────────▼───────────────┐   │   │
       │ │ │ Post-Attn LayerNorm (RMSNorm)  │   │   │
       │ │ └────────────────┬───────────────┘   │   │
       │ │                  │                    │   │
       │ │ ┌────────────────▼───────────────┐   │   │
       │ │ │ Qwen3MLP (SwiGLU)             │   │   │
       │ │ │ ┌──────────────────────────┐   │   │   │
       │ │ │ │ gate_up_proj             │   │   │   │
       │ │ │ │ [B,T,H] → [B,T,2*I]     │   │   │   │
       │ │ │ │ Gate: SiLU activation    │   │   │   │
       │ │ │ │ Up: linear transform     │   │   │   │
       │ │ │ │ SwiGLU = SiLU(G)*U      │   │   │   │
       │ │ │ └──────────┬───────────────┘   │   │   │
       │ │ │ ┌──────────▼───────────────┐   │   │   │
       │ │ │ │ down_proj                │   │   │   │
       │ │ │ │ [B,T,I] → [B,T,H]       │   │   │   │
       │ │ │ └──────────────────────────┘   │   │   │
       │ │ └────────────────────────────────┘   │   │
       │ │                                      │   │
       │ │ return hidden_states, residual       │   │
       │ └──────────────────┬───────────────────┘   │
       │                    │                        │
       │ ... 重复 x28 layers ...                     │
       │                    │                        │
       │ ┌──────────────────▼───────────────────┐   │
       │ │ Final LayerNorm (model.norm)         │   │
       │ └──────────────────┬───────────────────┘   │
       │                    │                        │
       │ ┌──────────────────▼───────────────────┐   │
       │ │ LM Head (ParallelLMHead)             │   │
       │ │ [B,T,H] → [B,T, vocab_size]         │   │
       │ │ 同时计算 logits                       │   │
       │ └──────────────────┬───────────────────┘   │
       └────────────────────┼──────────────────────┘
                            │
       ┌────────────────────▼───────────────────────┐
       │ 5. Sampling                                │
       │                                            │
       │ ┌──────────────────────────────────────┐   │
       │ │ Sampler                              │   │
       │ │ - Temperature scaling                │   │
       │ │ - Top-k / Top-p filtering            │   │
       │ │ - Gumbel softmax (if needed)         │   │
       │ │ - Multinomial sampling               │   │
       │ │ - Greedy/Beam search (if needed)     │   │
       │ └──────────────────┬───────────────────┘   │
       │                    │                        │
       │ sampled_token_ids: [B, 1]                   │
       └────────┬───────────────────────────────────┘
                │
       ┌────────▼───────────────────────────────────┐
       │ 6. Build Output                            │
       │                                            │
       │ ModelRunnerOutput                          │
       │ - sampled_token_ids                        │
       │ - logprobs (if requested)                   │
       │ - hidden_states (if draft model)            │
       │ - num_tokens_per_req                       │
       │ - request_ids                              │
       │ - finish_reasons                           │
       └────────┬───────────────────────────────────┘
                │
                ▼
       EngineCore._process_engine_step()
                │
                ▼
       OutputProcessor.process_outputs()
                │
                ▼
       Detokenizer: token_ids → text
                │
                ▼
       RequestOutput → User
```

---

## 4. 内存布局 - PagedAttention KV Cache

```
虚拟内存 (逻辑视角, per request)
┌────────────────────────────────────────┐
│ Request: "Hello, how are you?"        │
│                                        │
│ Block 0: "Hello,"       [slot 0-3]    │
│ Block 1: " how"         [slot 4-7]    │
│ Block 2: " are"         [slot 8-11]   │
│ Block 3: " you"         [slot 12-15]  │
│ Block 4: "?"            [slot 16-19]  │
└────────────────────────────────────────┘

物理内存 (GPU 视角)
┌─────────────┬─────────────┬─────────────┬─────────────┐
│  Block 0    │  Block 1    │  Block 2    │  Block 3    │
│ (free)→─┐   │ (Req A)│←─  │ (Req A)│←─  │ (Req B)    │
│          │   │        │    │        │    │             │
│          │   │ "Hello,"│    │ " how" │    │ "What"      │
├──────────┼───┼─────────┼────┼─────────┼────┼─────────────┤
│  Block 4    │  Block 5    │  Block 6    │  Block 7    │
│ (Req A)│←─  │ (free)     │ (free)     │ (free)      │
│          │   │            │            │             │
│ " are"  │   │            │            │             │
├──────────┼───┼────────────┼────────────┼─────────────┤
│  Block 8    │  Block 9    │  Block 10   │  Block 11   │
│ (Req A)│←─  │ (Req C)    │ (Req C)    │ (free)      │
│          │   │            │            │             │
│ " you"  │   │ "Paris is" │ " the"     │             │
├──────────┼───┼────────────┼────────────┼─────────────┤
│  Block 12   │  Block 13   │  Block 14   │  Block 15   │
│ (Req A)│←─  │ (Req C)    │ (Req C)    │ (free)      │
│          │   │            │            │             │
│ "?"     │   │ " capital" │ " of"      │             │
└──────────┴───┴────────────┴────────────┴─────────────┘

Block Table (per request)
┌───────────────────────────────────────────────────┐
│ BlockTable[Req A] = [1, 2, 4, 8, 12]              │
│ BlockTable[Req B] = [3]                            │
│ BlockTable[Req C] = [9, 10, 13, 14]               │
└───────────────────────────────────────────────────┘

Free Block List: [0, 5, 6, 7, 11, 15, ...]
```

---

## 5. 张量并行分切示意图

```
原始模型 (单一 GPU)
┌─────────────────────────────────────────────────┐
│           Qwen3-7B (28 layers)                  │
│                                                 │
│  QKV Projection: W[3584, 4608]                  │
│  ├ Q: W[3584, 3584]                            │
│  ├ K: W[3584, 512]                             │
│  └ V: W[3584, 512]                             │
│                                                 │
│  Output Projection: W[3584, 3584]               │
│                                                 │
│  MLP:                                           │
│  ├ gate_up: W[3584, 37888]                     │
│  └ down: W[18944, 3584]                        │
└─────────────────────────────────────────────────┘

TP=2 张量并行 (2 GPU)
┌───────────────────────────┐  ┌───────────────────────────┐
│        GPU 0              │  │        GPU 1              │
│                           │  │                           │
│ QKV: (Column Parallel)    │  │ QKV: (Column Parallel)    │
│ W_q[3584, 1792] heads 0-13│  │ W_q[3584, 1792] heads14-27│
│ W_k[3584, 256] kv_0-1     │  │ W_k[3584, 256] kv_2-3     │
│ W_v[3584, 256]            │  │ W_v[3584, 256]            │
│       ↓ all-reduce        │  │       ↓ all-reduce        │
│                           │  │                           │
│ Output: (Row Parallel)    │  │ Output: (Row Parallel)    │
│ W_o[1792, 3584]           │  │ W_o[1792, 3584]           │
│       ↓ reduce-scatter    │  │       ↓ reduce-scatter    │
│                           │  │                           │
│ MLP: (Column Parallel)    │  │ MLP: (Column Parallel)    │
│ W_gate_up[3584, 18944]    │  │ W_gate_up[3584, 18944]    │
│       ↓ all-reduce        │  │       ↓ all-reduce        │
│ W_down[9472, 3584]        │  │ W_down[9472, 3584]        │
│       ↓ reduce-scatter    │  │       ↓ reduce-scatter    │
│                           │  │                           │
│ Embedding: (Vocab Parallel)│  │ Embedding: (Vocab Parallel)│
│ E[76032, 3584] vocab [0:  │  │ E[76032, 3584] vocab [   │
│          76032]           │  │     76032:152064]         │
└───────────────────────────┘  └───────────────────────────┘
```

---

## 6. Decode 循环状态机

```
                    ┌──────────────┐
                    │   IDLE       │
                    │  (等待请求)    │
                    └──────┬───────┘
                           │ add_request()
                    ┌──────▼───────┐
                    │  QUEUED      │
                    │  (请求入队)    │
                    └──────┬───────┘
                           │ schedule()
                    ┌──────▼───────┐
                    │  PREFILL     │
                    │  (处理prompt) │
                    │              │
                    │  ┌────────────────────┐
                    │  │ Chunked Prefill?  │
                    │  │ Yes → 分块处理     │
                    │  │ No → 完整处理      │
                    │  └────────────────────┘
                    └──────┬───────┘
                           │ prompt done
                    ┌──────▼───────┐
                    │  DECODE      │
                    │  (逐token生成)│
                    │              │
                    │  ┌────────────────────┐
                    │  │ Finished?          │
                    │  │ - max_tokens reached│
                    │  │ - EOS token        │
                    │  │ - stop string      │
                    │  └────────────────────┘
                    └──────┬───────┘
                           │ finished
                    ┌──────▼───────┐
                    │  FINISHED    │
                    │  (输出结果)    │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │  DONE        │
                    │  (释放资源)    │
                    └──────────────┘
```
