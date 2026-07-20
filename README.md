# llm-serving-platform

A hands-on **LLM inference / serving optimization** platform: take one 3B model on one
consumer GPU and push it through three serving backends — **HF baseline → vLLM → quantized** —
measuring each against a fixed set of serving metrics. The deliverable is a **benchmark report**
that turns "self-hosting is cheaper/faster than a hosted API" from a claim into numbers.

> **Narrative tie-in (Project 1 → Project 2):** Project 1 (RAGOps Copilot) ended on
> "latency/cost is dominated by the external LLM → self-hosting can flip the economics."
> This project makes that concrete, and on **D5 it replays Project 1's real requests as the
> load** — the two projects bookend each other.

## Scope

| Choice | Value | Why |
|---|---|---|
| **Hardware** | Local **RTX 5080, 16 GB** (WSL2, CUDA 13 / Blackwell) | Free, already set up, and 16 GB is exactly the ceiling that makes batching/VRAM tradeoffs *show up*. |
| **Model** | **Qwen2.5-3B-Instruct** (fp16 ≈ 6 GB) | Leaves ~10 GB for KV cache / batching, so "batch too big → OOM" is demonstrable. Strong, permissive, no gated license. (Alt: `meta-llama/Llama-3.2-3B-Instruct`, but it's HF-gated.) |
| **Backends** | HF Transformers baseline → vLLM → quantized | Three-rung ladder that isolates *where* the wins come from (batching vs paged attention vs precision). |

## Metrics (the fixed yardstick)

Every backend is reported against the **same six metrics** under the **same load**. Full
definitions, the throughput↔latency tradeoff, and the cost model live in
[`docs/metrics.md`](docs/metrics.md).

| Metric | Meaning | Why it matters |
|---|---|---|
| **Throughput (tok/s)** | System tokens/sec across all concurrent requests | Capacity — users per card |
| **TTFT** | Time to first token | Perceived responsiveness (prefill-bound) |
| **TPOT / ITL** | Inter-token interval after the first | Smoothness of generation (decode-bound) |
| **p50 / p95 latency** | End-to-end latency percentiles | p95 = the tail that loses users |
| **Peak VRAM (GB)** | Peak GPU memory | The hard 16 GB wall → max batch/context |
| **$ / 1k tokens** | Imputed cost @ reference cloud GPU price | Makes results portable across hardware |

## Plan

- **W8 — Baseline benchmark report v1**
  - D1 ✅ scope + GPU env + metrics definition + repo scaffold *(this commit)*
  - D2 → HF baseline backend + FastAPI endpoint + **load generator** (configurable concurrency / prompt-length) → first baseline numbers
  - D3 → vLLM backend (continuous batching, paged attention)
  - D4–D5 → quantization + replay Project 1's real load
- **W9 — Optimization + cost matrix**
  - Tuning sweeps, batch/context limits, backend × config cost matrix, write-up.

## Repo layout

```
src/        serving backends + API (env_check.py today)
bench/      load generator (D2)
eval/       correctness / quality checks
docs/       metrics.md and design notes
results/    raw per-request json + summary tables
notes/      running lab notebook (mirrors ../Summer2026/notes/w8.md)
```

## Setup

```bash
uv sync                          # installs pinned deps from uv.lock
uv run python -m src.env_check   # smoke test: expect CUDA True + RTX 5080
```

Expected smoke-test output:

```
torch          2.13.0+cu130
cuda available True
gpu            NVIDIA GeForce RTX 5080
vram           17.1 GB
compute cap    12.0
transformers   5.14.1
matmul smoke   ok (...)
```

> **Environment note:** RTX 5080 (Blackwell, sm_120) needs the CUDA 13 torch wheels —
> `uv add torch --index https://download.pytorch.org/whl/cu130`. vLLM is intentionally
> **not** installed yet (added D3) to avoid dependency conflicts with the baseline stack.
