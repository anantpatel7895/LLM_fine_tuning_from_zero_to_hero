# Benchmarking — a system-design skill that happens to be useful for GPUs

**Why this note exists.** Phase 00 measured a GPU, but almost nothing in the *method* was GPU-specific. The same discipline applies to a database, an API, a Kafka cluster or a disk array. This file separates the transferable part from the CUDA part, because the transferable part is worth more.

**Read it as:** how to find out what a system can actually do, and how to avoid fooling yourself while doing it.

---

## Part 1 — What benchmarking is for

A benchmark answers **"what can this system actually do?"** as distinct from **"what does the vendor claim?"**

Phase 00, concretely:

| Claim | Measured | Ratio |
|---|---|---|
| 125 TFLOPS | 86 TFLOPS | 69% |
| 600 GB/s | 482 GB/s | 80% |
| "24 GB" | 21.75 GiB | 91% |

None of those gaps are defects. Vendor figures are laboratory maximums under ideal conditions; real systems pay for cooling, reserved memory, protocol overhead and imperfect kernels. **The gap is normal. Not knowing the gap is the problem.**

### The three questions a benchmark exists to answer

1. **Capacity planning** — *will this fit?* (Will an 8B model fit in 21.75 GiB?)
2. **Time estimation** — *how long will this take?* (Two hours or twenty?)
3. **Architecture choice** — *which design wins?* (One GPU or two? Shard the model or replicate it?)

If a measurement answers none of these, it is trivia. **Measure toward a decision**, otherwise you are collecting numbers for their own sake.

> This is the single biggest difference between a benchmark and a stress test. A stress test asks "does it break?" A benchmark asks "which choice should I make?"

---

## Part 2 — The four axes (this is the transferable part)

Every system has the same four performance axes. Only the units change.

| Axis | The question | GPU | Web service | Database |
|---|---|---|---|---|
| **Compute** | How fast can it *do work*? | 86 TFLOPS | requests/sec/core | queries/sec |
| **Bandwidth** | How fast can it *move data*? | 482 GB/s | Gbps in/out | disk read MB/s |
| **Capacity** | How much fits *at once*? | 21.75 GiB VRAM | RAM, connection pool | working set vs buffer pool |
| **Interconnect** | How fast do the *parts talk*? | 15.2 GB/s PCIe | inter-service RTT | replica lag |

**The system is only as fast as its binding axis**, and almost every performance surprise is someone optimizing a non-binding one.

The A10 box is a clean example. It has plenty of compute (86 TFLOPS) and a badly limited interconnect (15.2 GB/s, ~20× slower than NVLink). So full fine-tuning an 8B across both cards spends **1.06 s/step communicating** — the expensive compute sits idle waiting for the slow link. Buying faster cards would change nothing. The interconnect is binding.

> **Corollary:** find the binding axis *before* optimizing. "Make the code faster" is not a plan until you know which axis you are up against.

---

## Part 3 — The roofline model

A tool for predicting *which axis binds*, before building anything.

**Arithmetic intensity** = operations performed ÷ bytes moved. Then:

```
ridge point = peak compute ÷ peak bandwidth
```

For your A10, measured: `86 TFLOPS ÷ 482 GB/s` ≈ **178 FLOP/byte**.

- Job intensity **above** 178 → **compute-bound**. Optimize the maths.
- Job intensity **below** 178 → **memory-bound**. Optimize data movement.

Applied:

| Job | Intensity | Bound by | So you optimize |
|---|---|---|---|
| Training (large batched matmuls) | thousands | compute | bigger matmuls, bf16, packing, higher MFU |
| Text generation, batch 1 | ~1–2 | **memory, by ~100×** | quantization, smaller KV cache, batching |

This explains something that otherwise looks contradictory: **quantization transforms inference speed and barely helps training.** 4-bit weights move 4× fewer bytes — decisive when you are bandwidth-bound, nearly irrelevant when you are compute-bound.

**Generalizes directly.** A web service doing heavy JSON serialization per request is compute-bound; one that proxies large payloads is bandwidth-bound. Same reasoning, same conclusion structure.

---

## Part 4 — How to measure without fooling yourself

### Warm up first

The first run is always slower — lazy initialization, cold caches, JIT compilation, CUDA context creation, an empty connection pool. Discard the first few iterations. Benchmarking a cold start when you care about steady state measures the wrong thing entirely.

*(Unless you care about cold start — serverless, or a model-loading endpoint. Then measure it deliberately and separately.)*

### Repeat, and report the distribution

One measurement is an anecdote. Run many; report **p50 and p95/p99**, not the mean.

Averages hide the tail, and the tail is what users experience. A service averaging 50 ms with a p99 of 3 s is a bad service, and its average looks fine.

### Burst ≠ sustained

**This one caught us.** Your matmul numbers were measured on an idle card at 28 °C. A 150 W A10 throttles under a sustained three-hour run. The honest figure needs `nvidia-smi dmon` **during** real training.

Universal: databases look fast until the buffer cache fills; APIs look fast until connection pools saturate; disks look fast until the SLC write cache is exhausted. **Short benchmarks measure the best case and quietly call it "performance."**

### Measure at realistic scale

Bigger is not automatically faster — and Phase 00 proved it:

| Shape | % of peak |
|---|---|
| 2048³ | 68.4% |
| 4096³ | **68.8%** |
| 8192³ | **63.8%** ← slower |

At 8192³ the working set stops fitting in L2 cache and the kernel becomes bandwidth-starved. The largest size was the *least* efficient.

Every system has these cliff edges — a cache boundary, a page size, an MTU, a thread-pool limit. **Benchmark across a range and look for the cliff.** A single data point cannot show you one.

### Isolate the thing you are measuring

Your first `nvidia-smi` showed 28 GiB already consumed by a vLLM server and another job. Any benchmark run then would have been measuring contention, not capability. Check what else is running. On shared infrastructure this is the most common source of nonsense results.

---

## Part 5 — Five ways a benchmark lies

Three of these are mistakes I actually made in `00_gpu_benchmark.py`, caught only because the measurements disagreed with the derivations.

### 1. Mixing sources in one ratio

The script computed the ridge point as **spec** FLOPS ÷ **measured** bandwidth → a meaningless 259. Measured/measured gives 178; spec/spec gives 208. Both are defensible; the mixture is not.

> **Rule: both terms of a ratio must come from the same source.**

### 2. Measuring the wrong quantity and labelling it right

The script reported "CUDA context cost: 2.0 MiB." The real cost is ~277 MiB. It allocated a 1-element tensor and measured the delta — but CUDA was **already initialised**, so it measured allocator rounding on a 4-byte tensor.

The number was real. The label was wrong. That is worse than no number, because a labelled number gets believed.

> **Rule: state what you measured, not what you meant to measure.** If a result looks implausibly good, suspect the instrument.

### 3. Trusting a formula past its validity range

The script applied the textbook `2Ψ` all-reduce volume. A ring all-reduce moves `2(N-1)/N × S` per rank:

| N | factor |
|---|---|
| **2** | **1.00** |
| 4 | 1.50 |
| 8 | 1.75 |
| 64 | 1.97 |

`2Ψ` is the **large-N limit**. At N=2 it overstates by exactly 2×. The measured 84 MiB row (5.929 ms) disproved the derived 11.1 ms directly.

> **Rule: every formula has a validity range. Check that yours includes your case.**

### 4. Benchmarking something nobody runs

A pure matmul in a loop is not training. Real training adds data loading, optimizer steps, checkpointing, gradient synchronization, Python overhead. That is why **86 TFLOPS on a microbenchmark coexists with 25–35% MFU end-to-end** — both are true, measuring different things.

Microbenchmarks tell you the ceiling. Only end-to-end tells you the reality. **Report which one you ran.**

### 5. Averaging away the thing that matters

See Part 4. Worth stating twice.

---

## Part 6 — Benchmarks as a regression baseline

The underrated use. Once recorded in `docs/HARDWARE.md`, these numbers become an answer to a question you will definitely ask later:

> *"This run is much slower than last week. Did the machine change, or did I?"*

Re-run the benchmark. If it still says 86 TFLOPS and 15.2 GB/s, the hardware is fine and the problem is your code. If it says 40 TFLOPS, something changed underneath you — thermal throttling, a driver update, a noisy neighbour.

**Sixty seconds to halve the search space.** This is exactly what a health check endpoint or a synthetic canary does in production: an unchanging probe whose *drift* is the signal.

---

## Part 7 — The mapping to system design

| Phase 00 did | System design calls it |
|---|---|
| Measured peak matmul/bandwidth | Capacity planning, headroom analysis |
| Found the interconnect binding | Bottleneck analysis, Amdahl's law |
| Derived the ridge point | Workload characterization |
| Predicted, then measured, then explained the gap | Performance modelling |
| Recorded numbers as a baseline | SLO baselining, regression detection |
| Chose DDP over ZeRO-3 from measurement | Architecture decision, with evidence |

The last row is the whole point. The decision **"treat GPU 1 as a second experiment slot rather than a second worker"** contradicts most multi-GPU advice, which assumes NVLink. It is defensible only because the number was measured on *this* box.

> **Design decisions backed by measurement survive argument. Decisions backed by intuition do not.**

---

## Q&A

**QB.1 ⭐** — What is the difference between a benchmark, a load test and a stress test?

> **Benchmark:** what can it do under controlled conditions? Answers a *design* question.
> **Load test:** does it meet requirements at expected traffic? Answers a *readiness* question.
> **Stress test:** where and how does it break? Answers a *failure-mode* question.
>
> Different goals. A benchmark that "passes" tells you nothing about behaviour at 3× load; a load test that passes tells you nothing about which component is binding.

**QB.2 📐⭐** — What is a ridge point and how do you compute one?

> `peak compute ÷ peak bandwidth`, giving a threshold in operations-per-byte. Work above it is compute-bound, below it memory-bound. A10 measured: `86e12 ÷ 482e9` ≈ **178 FLOP/byte**.
>
> Both terms must come from the same source. It tells you *which axis to optimize* before you write any optimization.

**QB.3 🔧** — Why report p95/p99 instead of the mean?

> The mean hides the tail, and the tail is what users experience. 50 ms average with a 3 s p99 is a bad service that looks fine on a dashboard. Latency distributions are long-tailed, so the mean is not representative of anything.

**QB.4 🔧⭐** — Your benchmark says 86 TFLOPS but training reaches 30% MFU. Both correct — why?

> They measure different things. The microbenchmark is a pure matmul with no data loading, optimizer step, gradient sync, checkpointing or Python overhead. It establishes the **ceiling**. End-to-end MFU is what remains after real work. A gap is expected; an unexplained gap is a bug.

**QB.5 🔧** — A microbenchmark is fast and production is slow. What do you check first?

> Whether you benchmarked burst or sustained behaviour. Thermal throttling, cache filling, pool saturation and buffer exhaustion all appear only after minutes. Then check isolation — was anything else running? Then scale — does a cliff edge sit between your test size and production size?

**QB.6 ⭐** — Why did measuring the interconnect change the multi-GPU architecture?

> At 15.2 GB/s, exchanging gradients costs ~5.9 ms for LoRA (84 MiB) but ~1.06 s for a full 8B fine-tune (16 GiB). Against a ~1 s step, the first is free and the second doubles it.
>
> So LoRA scales across both cards and full fine-tuning does not — on *this* box. With NVLink at ~300 GB/s the same analysis flips. **The architecture follows from the measurement, not from the tutorial.**

**QB.7 🔧** — A result looks much better than expected. What now?

> Suspect the instrument before celebrating. "CUDA context costs 2 MiB" was implausibly good and was in fact measuring the wrong quantity. Implausibly good results are usually measurement bugs: a missing warmup, an optimized-away loop, an async call timed without synchronizing, or a cache serving the answer.

**QB.8 📐** — When does the `2Ψ` all-reduce formula apply, and when not?

> Ring all-reduce moves `2(N-1)/N × S` bytes per rank. `2Ψ` is the `N → ∞` limit. At N=2 the factor is exactly 1.0, so the formula overstates by 2×. It is a good approximation from roughly N=8 (1.75) upward.
>
> General lesson: **asymptotic formulas need a validity range attached.**

---

## See also

- [HARDWARE.md](../HARDWARE.md) — the measured baseline for this box
- [NUMBERS.md](../../NUMBERS.md) §6 (comms volume), §11 (roofline)
- [phase-00.md](../phases/phase-00.md) — prediction-vs-measurement table
- [LESSONS.md](../LESSONS.md) — L1 and L2, both cases of a tool reporting success while doing something else
