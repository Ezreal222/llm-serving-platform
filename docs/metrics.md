# Serving Metrics — the fixed yardstick

Every backend in this project (baseline → vLLM → quantized) is reported against the
**same six metrics** under the **same load**. If a change doesn't move one of these, it
didn't matter. This file is the contract; the load generator (D2) emits exactly these fields.

| Metric | Definition | Why it matters |
|---|---|---|
| **Throughput (tok/s)** | System-wide generated tokens per second (all concurrent requests). | The "capacity" of the server — how many users a single card can carry. |
| **TTFT** (Time To First Token) | Latency from request arrival to the first output token. | User-perceived responsiveness, especially for streaming. Dominated by **prefill**. |
| **TPOT / ITL** | Mean inter-token interval for the tokens after the first. | How "smooth" generation feels. Governed by the **decode** step (batch + memory bandwidth). |
| **p50 / p95 latency** | End-to-end per-request latency percentiles. | p95 captures the tail — the experience that actually loses users (echoes Project 1 D4). |
| **Peak VRAM (GB)** | Peak GPU memory during the run. | The hard 16 GB wall on the 5080 — sets the max batch / context. |
| **$ / 1k tokens** | Cost per 1000 generated tokens. | Makes results portable/comparable across hardware. |

## Throughput vs latency — the core tradeoff

They usually trade off. Larger batch → higher **throughput** but higher per-request
**latency**. Serving optimization = *maximize throughput subject to a latency SLA*
(e.g. "p95 TTFT < 500 ms"). Always report the two together; a throughput number without
its latency budget is meaningless.

## The two phases behind the latency metrics

- **Prefill** — process the whole prompt in one compute-bound pass → drives **TTFT**
  (longer prompt ⇒ slower first token).
- **Decode** — generate one token at a time, memory-bandwidth-bound → drives **TPOT**.
  This is where **KV cache** and **batching** pay off.

Because the two phases have different bottlenecks, they're optimized differently — which
is why we measure TTFT and TPOT separately, not just end-to-end latency.

## Cost model ($/1k tok) — how the local number becomes portable

The 5080 has no per-hour bill, so we impute one from a reference cloud GPU price:

```
$/1k tok = (reference_gpu_usd_per_hour / 3600) / measured_throughput_tok_s * 1000
```

- **Reference price:** cloud L4 ≈ **$0.75/hr** (pin the exact figure + source in results).
- Every backend is costed with the *same* reference price, so the comparison stays fair —
  only measured throughput changes the number.
- Report it as "imputed cost @ $X/hr reference", never as a real spend.

## Reporting rules (so backends are comparable)

1. **Same load** for every backend: same prompt-length distribution, same concurrency sweep,
   same output-length cap (defined by the D2 load generator).
2. **Same warmup**: discard the first N requests before measuring (JIT / cache warm).
3. Raw per-request records → `results/<backend>_<load>.json`; summary tables in the report.
4. Always pair a throughput number with the concurrency level and the latency percentiles
   it was measured at.
